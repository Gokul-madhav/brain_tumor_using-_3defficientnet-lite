
"""
06_export.py
Export trained EfficientNetLite3DUNet model.

Supported formats:
- PyTorch (.pth)
- TorchScript
- ONNX

Optional:
- TensorRT (after ONNX export)
"""

import os
import torch

from final_model import EfficientNetLite3DUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "weights/best_model.pth"
EXPORT_DIR = "exported_models"

NUM_CLASSES = 4

os.makedirs(EXPORT_DIR, exist_ok=True)


def load_model():

    model = EfficientNetLite3DUNet(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def export_torchscript(model):

    print("=" * 60)
    print("Exporting TorchScript...")

    dummy = torch.randn(
        1,
        4,
        128,
        128,
        128
    ).to(DEVICE)

    traced = torch.jit.trace(
        model,
        dummy
    )

    save_path = os.path.join(
        EXPORT_DIR,
        "efficientnet3d_unet.ts"
    )

    traced.save(save_path)

    print("Saved :", save_path)


def export_onnx(model):

    print("=" * 60)
    print("Exporting ONNX...")

    dummy = torch.randn(
        1,
        4,
        128,
        128,
        128
    ).to(DEVICE)

    save_path = os.path.join(
        EXPORT_DIR,
        "efficientnet3d_unet.onnx"
    )

    torch.onnx.export(
        model,
        dummy,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["prediction"],
        dynamic_axes={
            "image": {0: "batch"},
            "prediction": {0: "batch"}
        }
    )

    print("Saved :", save_path)


def benchmark(model):

    print("=" * 60)
    print("Benchmarking")

    dummy = torch.randn(
        1,
        4,
        128,
        128,
        128
    ).to(DEVICE)

    with torch.no_grad():

        for _ in range(10):
            _ = model(dummy)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        import time

        start = time.time()

        for _ in range(50):
            _ = model(dummy)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    latency = (end - start) / 50

    fps = 1 / latency

    print(f"Average Latency : {latency:.4f} sec")
    print(f"Inference FPS  : {fps:.2f}")


def model_information(model):

    print("=" * 60)

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    size = total * 4 / (1024 ** 2)

    print("Model Information")
    print("-" * 60)
    print(f"Parameters         : {total:,}")
    print(f"Trainable          : {trainable:,}")
    print(f"Estimated Size     : {size:.2f} MB")


if __name__ == "__main__":

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            "best_model.pth not found."
        )

    model = load_model()

    model_information(model)

    benchmark(model)

    export_torchscript(model)

    export_onnx(model)

    print("=" * 60)
    print("Export Complete")

    print()
    print("TensorRT Conversion")
    print("-------------------")
    print("trtexec --onnx=exported_models/efficientnet3d_unet.onnx "
          "--saveEngine=exported_models/model.engine --fp16")
