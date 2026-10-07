
"""
04_evaluate.py
Evaluate a trained BraTS segmentation model.
"""

import os
import torch
from tqdm import tqdm
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from data_pipeline import create_dataloaders
from final_model import EfficientNetLite3DUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "dataset/BraTS2021_Training_Data"
WEIGHTS = "weights/best_model.pth"
NUM_CLASSES = 4

_, loader = create_dataloaders(DATA_DIR)

model = EfficientNetLite3DUNet(num_classes=NUM_CLASSES).to(DEVICE)
ckpt = torch.load(WEIGHTS, map_location=DEVICE)
state = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state)
model.eval()

post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
post_label = AsDiscrete(to_onehot=NUM_CLASSES)

dice_metric = DiceMetric(include_background=False, reduction="mean")
hd95_metric = HausdorffDistanceMetric(
    include_background=False,
    percentile=95,
    reduction="mean"
)

@torch.no_grad()
def evaluate():
    dice_metric.reset()
    hd95_metric.reset()

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(images)

        preds = [post_pred(x) for x in decollate_batch(outputs)]
        gts = [post_label(x) for x in decollate_batch(labels)]

        dice_metric(preds, gts)
        hd95_metric(preds, gts)

    dice = dice_metric.aggregate().item()
    hd95 = hd95_metric.aggregate().item()

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean Dice Score : {dice:.4f}")
    print(f"Hausdorff95     : {hd95:.4f}")

    if "epoch" in ckpt:
        print(f"Checkpoint Epoch: {ckpt['epoch']}")
    if "dice" in ckpt:
        print(f"Best Saved Dice : {ckpt['dice']:.4f}")

    dice_metric.reset()
    torch.cuda.empty_cache()
    hd95_metric.reset()


if __name__ == "__main__":
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS}")
    evaluate()
