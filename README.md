# Brain Tumor Segmentation

A PyTorch and MONAI project for multi-class 3D brain tumor segmentation from BraTS-style multi-modal MRI volumes.

The project loads four co-registered MRI modalities, preprocesses them into fixed-size 3D volumes, and predicts a voxel-wise segmentation mask for background and three tumor regions.

## Project Structure

| File or directory | Purpose |
| --- | --- |
| `data_pipeline.py` | BraTS case loading, label remapping, preprocessing, augmentation, and train/validation dataloaders |
| `final_model.py` | Active 3D EfficientNet-Lite and nnU-Net-style segmentation model |
| `train.py` | Training loop, mixed precision, deep-supervision loss, checkpoint resume, and validation |
| `evaluate.py` | Evaluation of a saved model on the validation split |
| `inference.py` | Prediction and visualization for one patient case |
| `export.py` | TorchScript and ONNX export plus optional TensorRT command generation |
| `model.py` | Older, simpler model implementation retained for reference |
| `old_version/` | Previous experimental implementations |
| `dataset/` | Local BraTS data; excluded from version control |
| `weights/` | Local model checkpoints; excluded from version control |

## Dataset

The pipeline expects the BraTS training data in:

```text
dataset/
  BraTS2021_Training_Data/
    BraTS2021_XXXXX/
      BraTS2021_XXXXX_t1.nii.gz
      BraTS2021_XXXXX_t1ce.nii.gz
      BraTS2021_XXXXX_t2.nii.gz
      BraTS2021_XXXXX_flair.nii.gz
      BraTS2021_XXXXX_seg.nii.gz
```

The input channels are ordered as `t1`, `t1ce`, `t2`, and `flair`. The original BraTS segmentation labels are `{0, 1, 2, 4}` and are remapped to dense class IDs:

| Class ID | Region |
| --- | --- |
| `0` | Background |
| `1` | Necrotic tumor core, original label `1` |
| `2` | Peritumoral edema, original label `2` |
| `3` | Enhancing tumor, original label `4` |

The dataset is split into training and validation subsets using a deterministic seed of `42`. No separate test split is created by the current pipeline.

## Preprocessing

Each case is processed with MONAI transforms:

1. Ensure the label has a channel dimension.
2. Normalize each MRI channel using nonzero voxels.
3. Crop the image and label to the foreground.
4. Resize images with trilinear interpolation and labels with nearest-neighbor interpolation.
5. Produce a fixed spatial shape of `128 x 128 x 128`.
6. Apply training-only spatial and intensity augmentation.
7. Convert images and labels to PyTorch tensors.

Training augmentation includes random flips across the three spatial axes, random 90-degree rotations, Gaussian noise, Gaussian smoothing, intensity scaling, and intensity shifting. Validation uses the deterministic preprocessing steps only.

## Model Architecture

The active model is `EfficientNetLite3DUNet` from `final_model.py`. It accepts tensors shaped like `(batch, 4, depth, height, width)` and returns segmentation logits shaped like `(batch, 4, depth, height, width)`.

### Encoder

The encoder is a 3D EfficientNet-Lite-inspired feature extractor:

- A strided 3D convolution stem reduces the spatial resolution.
- Five encoder stages use inverted-residual MBConv blocks.
- MBConv blocks contain expansion, depthwise 3D convolution, squeeze-and-excitation recalibration, and pointwise projection.
- Instance normalization is used throughout the convolutional blocks.
- ReLU6 activations are used in the encoder.
- Residual connections are used when input and output channels match and the stride is one.
- Stochastic depth is progressively applied across encoder blocks.
- Optional gradient checkpointing reduces activation memory during training.

The encoder exposes intermediate feature maps for skip connections. Its channel progression is `32 -> 48 -> 80 -> 160 -> 256`.

### Decoder

The decoder follows an nnU-Net-style structure:

- Each decoder block upsamples with a 3D transposed convolution.
- Upsampled features are aligned to the skip feature size when dimensions are not exact multiples.
- Additive 3D attention gates can filter skip features before concatenation.
- Each decoder block uses two convolution, instance-normalization, and activation units.
- A final upsampling block restores the full output resolution.
- A `1 x 1 x 1` 3D convolution produces the four segmentation logits.

### Deep Supervision

During training, auxiliary segmentation heads are attached to intermediate decoder features at progressively lower resolutions. The training script computes a weighted combined loss for the main output and auxiliary outputs. Auxiliary labels are resized with nearest-neighbor interpolation.

Deep supervision is exposed through `forward_with_deep_supervision()`. The standard `forward()` method returns only the main logits tensor so evaluation, inference, TorchScript, ONNX export, and MONAI inference can use a stable output contract.

## Training

The training recipe in `train.py` uses:

- SGD with Nesterov momentum.
- Polynomial learning-rate decay.
- MONAI combined overlap and cross-entropy loss with softmax and one-hot targets.
- Automatic mixed precision when CUDA is available.
- Gradient scaling and gradient-norm clipping.
- Optional gradient accumulation.
- Automatic resume from `weights/best_model.pth`.
- Validation after each training epoch.

Run training from the project root:

```bash
python train.py
```

The script creates the `weights/` directory when needed and stores the best checkpoint there. The dataset path and other training settings are defined near the top of `train.py`.

## Evaluation

Evaluate the validation split with:

```bash
python evaluate.py
```

The evaluation script loads the active model and checkpoint and applies the validation preprocessing to the validation split.

## Inference

Set `CASE` in `inference.py` to a valid patient directory and run:

```bash
python inference.py
```

The script loads the four MRI modalities, predicts the segmentation, writes a NIfTI mask to `outputs/prediction.nii.gz`, and displays a central-slice visualization.

## Export

With a trained checkpoint available, run:

```bash
python export.py
```

The script can create:

- A TorchScript model at `exported_models/efficientnet3d_unet.ts`.
- An ONNX model at `exported_models/efficientnet3d_unet.onnx`.
- An optional TensorRT conversion command printed to the terminal.

The export path uses the standard single-output `forward()` method and a dynamic batch dimension.

## Environment

Create or activate a Python environment, install the dependencies required by the active PyTorch/MONAI scripts, and run commands from the repository root. A CUDA-enabled PyTorch installation is recommended for practical 3D training, but the scripts select CPU automatically when CUDA is unavailable.

Large datasets, checkpoints, generated exports, predictions, logs, text artifacts, Python caches, and the local virtual environment are excluded by `.gitignore`.

## Medical Use Notice

This repository is intended for research and educational use. Predictions are not a medical diagnosis and must not be used as a substitute for review by a qualified clinician.
