
"""
05_inference.py
Run inference on a single BraTS2021 patient and save prediction.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.transforms import Compose, NormalizeIntensity, Resize

from final_model import EfficientNetLite3DUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "weights/best_model.pth"
NUM_CLASSES = 4
TARGET_SHAPE = (128,128,128)

transform = Compose([
    NormalizeIntensity(nonzero=True, channel_wise=True),
    Resize(TARGET_SHAPE)
])

def load_case(case_dir):
    case_dir = Path(case_dir)
    pid = case_dir.name

    def read(name):
        return nib.load(case_dir/f"{pid}_{name}.nii.gz").get_fdata().astype(np.float32)

    t1 = read("t1")
    t1ce = read("t1ce")
    t2 = read("t2")
    flair = read("flair")

    image = np.stack([t1,t1ce,t2,flair], axis=0)
    image = transform(image)
    return image

def predict(case_dir):
    model = EfficientNetLite3DUNet(NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    image = load_case(case_dir)
    x = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    return image, pred

def save_prediction(mask, out_path):
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(mask, affine), out_path)

def visualize(image, pred):
    z = pred.shape[-1]//2

    plt.figure(figsize=(12,4))

    plt.subplot(131)
    plt.imshow(image[3,:,:,z], cmap="gray")
    plt.title("FLAIR")

    plt.subplot(132)
    plt.imshow(pred[:,:,z], cmap="jet")
    plt.title("Prediction")

    plt.subplot(133)
    plt.imshow(image[3,:,:,z], cmap="gray")
    plt.imshow(pred[:,:,z], cmap="jet", alpha=0.5)
    plt.title("Overlay")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    CASE = "dataset/BraTS2021_Training_Data/BraTS2021_00000"  # change to a valid case

    image, pred = predict(CASE)

    Path("outputs").mkdir(exist_ok=True)

    out_file = "outputs/prediction.nii.gz"
    save_prediction(pred, out_file)

    print(f"Prediction saved to: {out_file}")
    print("Prediction shape:", pred.shape)

    visualize(image, pred)
