
"""
03_train.py
Training script for BraTS2021 segmentation.
Works with:
 - data_pipeline.py
 - final_model.py (EfficientNetLite3DUNet: 3D EfficientNet-Lite encoder + true nnU-Net decoder)

Training recipe follows nnU-Net conventions, which is what this architecture
needs to have a realistic shot at ~85% mean Dice on BraTS:
 - SGD + Nesterov momentum, poly LR decay (not AdamW + cosine)
 - Deep supervision: loss computed at full-res + 3 decoder scales, weighted
   by 1/2^depth and matched against downsampled ground-truth labels
 - Many more epochs than the original 10 (segmentation on 1251 BraTS cases
   needs on the order of hundreds of epochs to converge -- resume-from-
   checkpoint below lets you run this in chunks)
 - Optional gradient accumulation + gradient checkpointing to fit the
   patch/batch size on a 6GB GPU
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import SGD
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from data_pipeline import create_dataloaders
from final_model import EfficientNetLite3DUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
print(f"CUDA Version    : {torch.version.cuda}")
print(f"Using Device    : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU Name        : {torch.cuda.get_device_name(0)}")

print("=" * 60)

DATA_DIR = "dataset/BraTS2021_Training_Data"
EPOCHS = 100
INITIAL_LR = 1e-2
WEIGHT_DECAY = 3e-5
MOMENTUM = 0.99
NUM_CLASSES = 4
SAVE_DIR = "weights"

CHECKPOINT_PATH = os.path.join(SAVE_DIR, "best_model.pth")

# Gradient accumulation: effective batch size = BATCH_SIZE (data_pipeline.py) * ACCUM_STEPS.
# Bump this instead of BATCH_SIZE if you hit CUDA OOM.
ACCUM_STEPS = 1

# Trades compute for memory in the encoder; matters on small (<=8GB) GPUs.
USE_GRAD_CHECKPOINT = True

# main-res weight = 1.0, then halved per decoder level going up
# (ds at 1/4, 1/8, 1/16 resolution), normalized to sum to 1.
DS_LEVEL_WEIGHTS = [1.0, 0.5, 0.25, 0.125]
DS_LEVEL_WEIGHTS = [w / sum(DS_LEVEL_WEIGHTS) for w in DS_LEVEL_WEIGHTS]

os.makedirs(SAVE_DIR, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()

train_loader, val_loader = create_dataloaders(DATA_DIR)

model = EfficientNetLite3DUNet(
    num_classes=NUM_CLASSES,
    deep_supervision=True,
    use_attention=True,
    drop_path_rate=0.2,
    use_checkpoint=USE_GRAD_CHECKPOINT,
).to(DEVICE)

loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)


def deep_supervision_loss(main_out, aux_outputs, labels):
    """nnU-Net-style multi-scale loss: main output + downsampled-label losses
    at each auxiliary decoder resolution, weighted by DS_LEVEL_WEIGHTS."""

    if not aux_outputs:
        return loss_fn(main_out, labels)

    weights = DS_LEVEL_WEIGHTS[: len(aux_outputs) + 1]
    weights = [w / sum(weights) for w in weights]

    total = weights[0] * loss_fn(main_out, labels)

    labels_f = labels.float()
    for w, aux in zip(weights[1:], aux_outputs):
        target = F.interpolate(labels_f, size=aux.shape[2:], mode="nearest").long()
        total = total + w * loss_fn(aux, target)

    return total


class PolyLRScheduler:
    """nnU-Net's default schedule: lr = initial_lr * (1 - epoch / max_epochs) ** power."""

    def __init__(self, optimizer, initial_lr, max_epochs, power=0.9, min_lr=1e-6):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        self.last_epoch = 0

    def step(self, epoch):
        self.last_epoch = epoch
        frac = min(epoch / self.max_epochs, 1.0)
        lr = max(self.initial_lr * (1 - frac) ** self.power, self.min_lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, sd):
        self.last_epoch = sd["last_epoch"]


optimizer = SGD(
    model.parameters(),
    lr=INITIAL_LR,
    momentum=MOMENTUM,
    nesterov=True,
    weight_decay=WEIGHT_DECAY,
)

scheduler = PolyLRScheduler(optimizer, INITIAL_LR, EPOCHS, power=0.9)

scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))

dice_metric = DiceMetric(
    include_background=False,
    reduction="mean"
)

post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
post_label = AsDiscrete(to_onehot=NUM_CLASSES)

best_dice = 0.0
start_epoch = 1
# ==========================================================
# Automatic Resume
# ==========================================================

if os.path.exists(CHECKPOINT_PATH):

    print("=" * 60)
    print("Checkpoint found.")
    print("Loading previous training state...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1
    best_dice = checkpoint["dice"]

    print(f"Resuming from Epoch {start_epoch}")
    print(f"Best Dice : {best_dice:.4f}")
    print("=" * 60)

else:

    print("=" * 60)
    print("No checkpoint found.")
    print("Starting fresh training.")
    print("=" * 60)


def train_one_epoch(epoch):
    model.train()

    epoch_loss = 0

    pbar = tqdm(train_loader)

    optimizer.zero_grad()

    for step, batch in enumerate(pbar):

        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):

            outputs, aux_outputs = model.forward_with_deep_supervision(images)

            loss = deep_supervision_loss(outputs, aux_outputs, labels)
            loss_to_backprop = loss / ACCUM_STEPS

        if not torch.isfinite(loss):
            print("Invalid loss detected:", loss)
            raise RuntimeError("Loss became NaN or Inf")

        scaler.scale(loss_to_backprop).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            # unscale before clipping -- clipping the AMP-scaled gradients directly
            # (as the original script did) clips against the wrong magnitude
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item()

        pbar.set_description(
            f"Epoch {epoch}"
        )

        pbar.set_postfix(
            loss=loss.item()
        )

    return epoch_loss / len(train_loader)


@torch.no_grad()
def validate():

    model.eval()

    dice_metric.reset()

    val_loss = 0

    for batch in tqdm(val_loader):

        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        val_loss += loss.item()

        outputs = [post_pred(i) for i in decollate_batch(outputs)]
        labels = [post_label(i) for i in decollate_batch(labels)]

        dice_metric(outputs, labels)

    dice = dice_metric.aggregate().item()

    dice_metric.reset()

    return val_loss / len(val_loader), dice


for epoch in range(start_epoch, EPOCHS + 1):

    train_loss = train_one_epoch(epoch)

    val_loss, dice = validate()

    current_lr = scheduler.step(epoch)

    print("=" * 60)
    print(f"Epoch        : {epoch}")
    print(f"Train Loss   : {train_loss:.4f}")
    print(f"Valid Loss   : {val_loss:.4f}")
    print(f"Dice Score   : {dice:.4f}")
    print(f"LR           : {current_lr:.6f}")

    if dice > best_dice:

        best_dice = dice

        torch.save(
            {
                "epoch": epoch,
                "dice": dice,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict()
            },
            CHECKPOINT_PATH
        )

        print("Best model saved.")

print("Training Finished.")
print("Best Dice :", best_dice)
