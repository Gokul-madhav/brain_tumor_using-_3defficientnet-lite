
"""
final_model.py
3D EfficientNet-Lite Encoder + True nnU-Net Decoder for BraTS-style
multi-class brain tumor segmentation.

Drop-in replacement for model.py: exposes the same class name
(`EfficientNetLite3DUNet`) and the same default forward() contract
(single logits tensor of shape (B, num_classes, D, H, W)), so
train.py / evaluate.py / inference.py / export.py work against it
without changes to their calling code.

Architecture checklist:
  - 3D EfficientNet-Lite encoder             -> EfficientNetLite3DEncoder
  - True nnU-Net decoder                     -> NNUNetDecoder3D
  - Residual MBConv blocks                   -> MBConvBlock3D
  - InstanceNorm3d                           -> ConvInstanceNormAct3D (everywhere)
  - Deep supervision                         -> NNUNetDecoder3D aux heads + forward_with_deep_supervision()
  - Stochastic depth                         -> DropPath3D, linearly scaled across encoder depth
  - Squeeze-and-Excitation                   -> SEBlock3D inside every MBConv block
  - Weight initialization                    -> EfficientNetLite3DUNet._initialize_weights
  - Optional Attention Gates                 -> AttentionGate3D, toggled via use_attention
  - Sliding-window inference compatibility   -> decoder tolerates non-exact skip/upsample size mismatches
  - TorchScript compatibility                -> canonical forward() has a static Tensor return type
                                                 (verified with torch.jit.trace in __main__; deep
                                                 supervision lives in a separate eager-only method so
                                                 it never pollutes the traced/scripted graph)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

class ConvInstanceNormAct3D(nn.Module):
    """Conv3d -> InstanceNorm3d -> activation."""

    def __init__(self, in_c, out_c, k=3, s=1, groups=1, act="leaky_relu"):
        super().__init__()
        layers = [
            nn.Conv3d(in_c, out_c, k, stride=s, padding=k // 2, groups=groups, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
        ]
        if act == "leaky_relu":
            layers.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        elif act == "relu6":
            layers.append(nn.ReLU6(inplace=True))
        elif act == "silu":
            layers.append(nn.SiLU(inplace=True))
        elif act is not None:
            raise ValueError(f"Unknown activation: {act}")
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation, implemented with 1x1x1 convs (no reshape -> TorchScript friendly)."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, reduced, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(reduced, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w


class DropPath3D(nn.Module):
    """Stochastic depth: randomly drops the residual branch per-sample during training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class MBConvBlock3D(nn.Module):
    """Residual (inverted-residual) MBConv block: expand -> depthwise -> SE -> project."""

    def __init__(self, in_c, out_c, expand_ratio=6, stride=1, kernel_size=3,
                 se_reduction=4, drop_path=0.0):
        super().__init__()
        self.use_res = stride == 1 and in_c == out_c
        hidden = in_c * expand_ratio if expand_ratio != 1 else in_c

        layers = []
        if expand_ratio != 1:
            layers.append(ConvInstanceNormAct3D(in_c, hidden, k=1, act="relu6"))

        layers.append(ConvInstanceNormAct3D(
            hidden, hidden, k=kernel_size, s=stride, groups=hidden, act="relu6"
        ))
        layers.append(SEBlock3D(hidden, reduction=se_reduction))
        layers.append(ConvInstanceNormAct3D(hidden, out_c, k=1, act=None))

        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath3D(drop_path) if self.use_res else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = x + self.drop_path(out)
        return out


class AttentionGate3D(nn.Module):
    """Additive attention gate (Oktay et al.) applied to a skip connection before concatenation."""

    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or max(skip_channels // 2, 1)
        self.theta = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True)
        self.phi = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True)
        self.norm = nn.InstanceNorm3d(inter_channels, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        theta_x = self.theta(skip)
        phi_g = self.phi(gate)
        if phi_g.shape[2:] != theta_x.shape[2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode="trilinear", align_corners=False)
        f = self.act(self.norm(theta_x + phi_g))
        alpha = self.psi(f)
        return skip * alpha


# ----------------------------------------------------------------------------
# Encoder: 3D EfficientNet-Lite
# ----------------------------------------------------------------------------

# (out_channels, expand_ratio, repeats, stride, kernel_size)
ENCODER_STAGE_CONFIG = [
    (32, 1, 1, 1, 3),
    (48, 6, 2, 2, 3),
    (80, 6, 2, 2, 3),
    (160, 6, 3, 2, 3),
    (256, 6, 3, 2, 3),
]


class EfficientNetLite3DEncoder(nn.Module):
    def __init__(self, in_channels=4, stem_channels=32, drop_path_rate=0.2, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.stem = ConvInstanceNormAct3D(in_channels, stem_channels, k=3, s=2, act="relu6")

        total_blocks = sum(cfg[2] for cfg in ENCODER_STAGE_CONFIG)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        self.stages = nn.ModuleList()
        in_c = stem_channels
        idx = 0
        for out_c, expand, repeats, stride, k in ENCODER_STAGE_CONFIG:
            blocks = []
            for r in range(repeats):
                s = stride if r == 0 else 1
                blocks.append(MBConvBlock3D(
                    in_c, out_c, expand_ratio=expand, stride=s,
                    kernel_size=k, drop_path=dpr[idx],
                ))
                in_c = out_c
                idx += 1
            self.stages.append(nn.Sequential(*blocks))

        self.out_channels = [cfg[0] for cfg in ENCODER_STAGE_CONFIG]

    def _run_stage(self, stage, x):
        if self.use_checkpoint and self.training:
            return checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        for stage in self.stages:
            x = self._run_stage(stage, x)
            skips.append(x)
        return skips[0], skips[1], skips[2], skips[3], skips[4]


# ----------------------------------------------------------------------------
# Decoder: true nnU-Net style
# ----------------------------------------------------------------------------

class NNUNetDecoderBlock3D(nn.Module):
    def __init__(self, in_c, skip_c, out_c, use_attention=True, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2)
        self.attn = AttentionGate3D(out_c, skip_c) if use_attention else None
        self.conv1 = ConvInstanceNormAct3D(out_c + skip_c, out_c, act="leaky_relu")
        self.conv2 = ConvInstanceNormAct3D(out_c, out_c, act="leaky_relu")
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            # keeps the decoder correct for sliding-window ROI sizes that
            # aren't an exact multiple of the encoder's total stride (32x)
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        if self.attn is not None:
            skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.dropout(x)


class NNUNetDecoder3D(nn.Module):
    def __init__(self, encoder_channels, num_classes=4, use_attention=True, dropout=0.0):
        super().__init__()
        s1_c, s2_c, s3_c, s4_c, b_c = encoder_channels

        self.stage4 = NNUNetDecoderBlock3D(b_c, s4_c, s4_c, use_attention, dropout)
        self.stage3 = NNUNetDecoderBlock3D(s4_c, s3_c, s3_c, use_attention, dropout)
        self.stage2 = NNUNetDecoderBlock3D(s3_c, s2_c, s2_c, use_attention, dropout)
        self.stage1 = NNUNetDecoderBlock3D(s2_c, s1_c, s1_c, use_attention, dropout)

        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(s1_c, s1_c, kernel_size=2, stride=2),
            ConvInstanceNormAct3D(s1_c, s1_c, act="leaky_relu"),
            ConvInstanceNormAct3D(s1_c, s1_c, act="leaky_relu"),
        )

        self.seg_head = nn.Conv3d(s1_c, num_classes, kernel_size=1)

        # deep supervision heads, finest -> coarsest: 1/4, 1/8, 1/16 resolution
        self.ds_head2 = nn.Conv3d(s2_c, num_classes, kernel_size=1)
        self.ds_head3 = nn.Conv3d(s3_c, num_classes, kernel_size=1)
        self.ds_head4 = nn.Conv3d(s4_c, num_classes, kernel_size=1)

    def forward(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        compute_aux: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s1, s2, s3, s4, b = feats

        d4 = self.stage4(b, s4)
        d3 = self.stage3(d4, s3)
        d2 = self.stage2(d3, s2)
        d1 = self.stage1(d2, s1)

        out = self.final_up(d1)
        main_out = self.seg_head(out)

        aux: List[torch.Tensor] = []
        if compute_aux:
            aux.append(self.ds_head2(d2))
            aux.append(self.ds_head3(d3))
            aux.append(self.ds_head4(d4))

        return main_out, aux


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------

class EfficientNetLite3DUNet(nn.Module):
    """3D EfficientNet-Lite encoder + true nnU-Net decoder.

    forward(x) always returns a single (B, num_classes, D, H, W) logits
    tensor, matching model.py's contract exactly -- this keeps evaluate.py,
    inference.py, export.py (incl. torch.jit.trace / ONNX) and MONAI's
    sliding_window_inference working unmodified.

    Deep supervision is only exposed through the explicit, eager-only
    forward_with_deep_supervision(x) method, used by train.py.
    """

    def __init__(
        self,
        num_classes=4,
        in_channels=4,
        stem_channels=32,
        deep_supervision=True,
        use_attention=True,
        drop_path_rate=0.2,
        dropout=0.0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.encoder = EfficientNetLite3DEncoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = NNUNetDecoder3D(
            self.encoder.out_channels,
            num_classes=num_classes,
            use_attention=use_attention,
            dropout=dropout,
        )

        self._initialize_weights()

    def forward(self, x) -> torch.Tensor:
        feats = self.encoder(x)
        main_out, _ = self.decoder(feats, compute_aux=False)
        return main_out

    def forward_with_deep_supervision(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Eager-only: returns (main_output, [ds_1/4, ds_1/8, ds_1/16]).

        Aux outputs are only computed when the model is both in train()
        mode and deep_supervision=True; otherwise aux is an empty list and
        this is equivalent to forward(x).
        """
        feats = self.encoder(x)
        compute_aux = self.training and self.deep_supervision
        main_out, aux = self.decoder(feats, compute_aux=compute_aux)
        return main_out, aux

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # zero-init the last norm of every residual MBConv branch so each
        # block starts as an identity mapping (standard ResNet/EfficientNet
        # trick -- stabilizes early training in deep residual stacks)
        for m in self.modules():
            if isinstance(m, MBConvBlock3D) and m.use_res:
                last_norm = m.block[-1].block[-1]
                if isinstance(last_norm, nn.InstanceNorm3d) and last_norm.affine:
                    nn.init.zeros_(last_norm.weight)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EfficientNetLite3DUNet(num_classes=4).to(device)

    x = torch.randn(1, 4, 128, 128, 128, device=device)

    # --- eval-mode single-tensor forward (matches evaluate/inference/export) ---
    model.eval()
    with torch.no_grad():
        y = model(x)
    total, trainable = count_parameters(model)
    print("Input             :", x.shape)
    print("Output (eval)     :", y.shape)
    print(f"Total Parameters  : {total:,}")
    print(f"Trainable Params  : {trainable:,}")

    # --- train-mode deep supervision forward ---
    model.train()
    main_out, aux_outputs = model.forward_with_deep_supervision(x)
    print("\nDeep supervision (train mode):")
    print("  main :", main_out.shape)
    for i, a in enumerate(aux_outputs):
        print(f"  ds{i} :", a.shape)

    # --- sliding-window-style patch that isn't a multiple of the 32x stride ---
    model.eval()
    patch = torch.randn(1, 4, 96, 112, 96, device=device)
    with torch.no_grad():
        out = model(patch)
    print("\nNon-32-multiple patch:")
    print("  in  :", patch.shape)
    print("  out :", out.shape)

    # --- TorchScript trace (same call pattern export.py uses) ---
    model.eval()
    traced = torch.jit.trace(model, x)
    with torch.no_grad():
        y_traced = traced(x)
    print("\nTorchScript trace output:", y_traced.shape)
    print("Matches eager output    :", torch.allclose(y, y_traced))
