# Brain Tumor Segmentation — Project Overview

## Goal

3D semantic segmentation of brain tumors from multi-modal MRI, trained on the
full **BraTS2021** dataset. Target metric: **85% mean Dice** (multi-class,
background excluded).

## Dataset

**BraTS2021** (Brain Tumor Segmentation Challenge 2021), stored locally at
[dataset/BraTS2021_Training_Data/](dataset/BraTS2021_Training_Data/).

- **1,251 patient cases**, one subfolder per case (`BraTS2021_XXXXX/`).
- **4 MRI modalities per case**, each a `.nii.gz` (NIfTI) volume:
  - `t1` — T1-weighted
  - `t1ce` — T1-weighted, contrast-enhanced
  - `t2` — T2-weighted
  - `flair` — T2 Fluid-Attenuated Inversion Recovery
- **1 segmentation label map per case**: `_seg.nii.gz`, voxel-wise annotated by
  radiologists. Original BraTS labels are `{0, 1, 2, 4}`; the pipeline remaps
  them to a dense `{0, 1, 2, 3}` scheme in [data_pipeline.py](data_pipeline.py) (`remap_mask`):
  - `0` = background
  - `1` = NCR (necrotic tumor core) — originally label 1
  - `2` = ED (peritumoral edema) — originally label 2
  - `3` = ET (enhancing tumor) — originally label 4
- Each volume is natively `240×240×155` voxels, 1mm isotropic, skull-stripped
  and co-registered across modalities (standard BraTS preprocessing).
- 5 files per case × 1,251 cases ≈ 6,255 NIfTI volumes on disk.
- Split: 80/20 train/val via `random_split` with a fixed seed (42) — no
  separate held-out test set defined in-repo.

Total dataset footprint is large (NIfTI volumes at this resolution run
several hundred MB per case), so it's stored outside version control (no `.git`
in this directory).

## Pipeline

| Stage | File | Role |
|---|---|---|
| Data loading | [data_pipeline.py](data_pipeline.py) | `BraTSDataset` loads all 4 modalities + label per case, stacks into a `(4, D, H, W)` tensor, applies MONAI transforms |
| Model | [final_model.py](final_model.py) | `EfficientNetLite3DUNet` — current architecture (see below) |
| Training | [train.py](train.py) | nnU-Net-style training loop, checkpoint/resume, deep supervision |
| Evaluation | [evaluate.py](evaluate.py) | Dice + Hausdorff95 on the val split |
| Inference | [inference.py](inference.py) | Single-case prediction + visualization |
| Export | [export.py](export.py) | TorchScript / ONNX export + latency benchmark |

### Preprocessing / augmentation ([data_pipeline.py](data_pipeline.py))

- Per-channel intensity normalization (nonzero-only), foreground cropping,
  resize to a fixed `128³` patch (trilinear for image, nearest for label).
- Train-time augmentation: random flips (all 3 axes), random 90° rotation,
  Gaussian noise/smoothing, random intensity scale/shift.
- Batch size 2, `num_workers=0`.

### Model ([final_model.py](final_model.py)) — `EfficientNetLite3DUNet`

A custom 3D nnU-Net-style network, ~**10.6M parameters**:

- **Encoder**: 3D EfficientNet-Lite (5 stages, residual MBConv blocks with
  Squeeze-and-Excitation, InstanceNorm3d, stochastic depth/DropPath scaled
  across depth).
- **Decoder**: true nnU-Net decoder — transposed-conv upsampling, optional
  additive attention gates on skip connections, double conv blocks.
- **Deep supervision**: auxiliary segmentation heads at 1/4, 1/8, 1/16
  resolution, used only during training via a separate
  `forward_with_deep_supervision()` method (kept out of default `forward()`
  for TorchScript/ONNX-export safety).
- Kaiming init + zero-init on the last norm of each residual branch
  (stabilizes deep residual stacks early in training).
- `forward(x)` always returns a single `(B, 4, D, H, W)` logits tensor,
  matching the retired `model.py`'s contract — `evaluate.py`, `inference.py`,
  and `export.py` all call it directly and are unaffected by encoder/decoder
  internals.

### Training recipe ([train.py](train.py))

Deliberately nnU-Net-style rather than the more common AdamW+cosine setup,
since that's what a ~10M-param 3D model needs for a realistic shot at 85%
Dice on BraTS:

- **Optimizer**: SGD + Nesterov, momentum 0.99, weight decay 3e-5.
- **LR schedule**: poly decay, `lr = 1e-2 * (1 - epoch/EPOCHS)^0.9`.
- **Loss**: `DiceCELoss`, combined across main + 3 auxiliary deep-supervision
  outputs, weighted `[1.0, 0.5, 0.25, 0.125]` (normalized) — aux targets are
  the ground-truth labels downsampled (nearest) to each decoder scale.
- **Mixed precision** (AMP) with gradient scaling; gradients are unscaled
  *before* clipping (`clip_grad_norm_`, max-norm 12.0) — clipping AMP-scaled
  gradients directly was a bug fixed during the July rewrite.
- **Gradient checkpointing** on the encoder by default, to fit on small GPUs.
- **Checkpoint/resume**: every improvement in val Dice is saved to
  `weights/best_model.pth` (model, optimizer, scheduler, scaler, epoch,
  dice) — training is designed to run in many chunks, not one sitting.
- `EPOCHS = 35` currently set in the script (memory notes an original target
  of 300; current value is lower — worth confirming intent before a long run).

### Hardware constraint

Single **NVIDIA RTX 4050 Laptop GPU, 6GB VRAM**. This is the binding
constraint on patch size (currently 128³), batch size (currently 2), and
model width — gradient checkpointing and AMP exist specifically to fit
within this budget.

## Current training status

Per the latest checkpoint at [weights/best_model.pth](weights/best_model.pth):

- **Epoch 30** (of 35 configured), **best val Dice ≈ 0.782**.
- Training is still short of the 85% Dice target; more epochs and/or recipe
  tuning are the likely next step.

## Notable inconsistencies / leftovers in the repo

- [requirements.txt](requirements.txt) lists TensorFlow/Keras/tf2onnx dependencies, but the
  entire active pipeline (`train.py`, `data_pipeline.py`, `final_model.py`,
  `evaluate.py`, `inference.py`, `export.py`) is PyTorch + MONAI. The
  TensorFlow deps appear to be leftovers from an earlier approach and don't
  reflect what's actually needed to run this code.
- [old_version/](old_version/) (`v1.py`–`v4.py`) and the root-level
  [training.log](training.log) are from a prior, unrelated **TensorFlow-based
  binary classification** attempt (not segmentation) — `training.log` shows
  it failed with a shape mismatch (`target.shape=(None,)` vs
  `output.shape=(None, 2)`) and was abandoned in favor of the current
  PyTorch/MONAI segmentation pipeline.
- [model.py](model.py) (a simpler EfficientNet-Lite-ish encoder + basic U-Net decoder) is
  superseded by [final_model.py](final_model.py) but still present on disk, unused by any
  active script.
- A stray **`best_model.pth` in the project root** (epoch 1, Dice ≈ 0.37) is
  a much earlier/stale checkpoint, distinct from the current one in
  [weights/best_model.pth](weights/best_model.pth) (epoch 30, Dice ≈ 0.78) that all scripts
  actually load.

## Export / deployment ([export.py](export.py))

Supports TorchScript tracing and ONNX export (opset 17, dynamic batch axis),
plus a latency/FPS benchmark and a printed `trtexec` command for an optional
TensorRT conversion step.
