
"""
02_model.py
Research Starter: EfficientNet-Lite 3D Encoder + nnU-Net Style Decoder
PyTorch
NOTE:
This is a clean, extensible baseline architecture. You can progressively
add deep supervision, ASPP, attention gates, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, k, stride=s, padding=k//2, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class MBConv3D(nn.Module):
    def __init__(self, in_c, out_c, expand=6, stride=1):
        super().__init__()
        hidden = in_c * expand
        self.use_res = stride == 1 and in_c == out_c

        self.block = nn.Sequential(
            nn.Conv3d(in_c, hidden, 1, bias=False),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv3d(
                hidden,
                hidden,
                3,
                stride=stride,
                padding=1,
                groups=hidden,
                bias=False
            ),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.SiLU(inplace=True),

            SEBlock(hidden),

            nn.Conv3d(hidden, out_c, 1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = ConvBNReLU(4, 32, s=2)

        self.e1 = MBConv3D(32, 32)
        self.e2 = MBConv3D(32, 48, stride=2)
        self.e3 = MBConv3D(48, 80, stride=2)
        self.e4 = MBConv3D(80, 160, stride=2)
        self.e5 = MBConv3D(160, 256, stride=2)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.e1(s1)
        s3 = self.e2(s2)
        s4 = self.e3(s3)
        s5 = self.e4(s4)
        b = self.e5(s5)
        return s2, s3, s4, s5, b


class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, 2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(out_c + skip_c, out_c),
            ConvBNReLU(out_c, out_c)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 4 -> 8
        self.d5 = UpBlock(256, 160, 160)

        # 8 -> 16
        self.d4 = UpBlock(160, 80, 80)

        # 16 -> 32
        self.d3 = UpBlock(80, 48, 48)

        # 32 -> 64
        self.d2 = UpBlock(48, 32, 32)

        # NEW: 64 -> 128
        self.d1 = nn.Sequential(
            nn.ConvTranspose3d(
                32,
                32,
                kernel_size=2,
                stride=2
            ),
            ConvBNReLU(32, 32),
            ConvBNReLU(32, 32)
        )

    def forward(self, feats):

        s2, s3, s4, s5, b = feats

        x = self.d5(b, s5)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)

        # Final Upsampling
        x = self.d1(x)

        return x


class EfficientNetLite3DUNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.head = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):

        feats = self.encoder(x)

        x = self.decoder(feats)

        x = self.head(x)

        return x


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


if __name__ == "__main__":

    model = EfficientNetLite3DUNet(num_classes=4)

    x = torch.randn(1, 4, 128, 128, 128)

    with torch.no_grad():
        y = model(x)

    total, trainable = count_parameters(model)

    print(model)
    print()
    print("Input :", x.shape)
    print("Output:", y.shape)
    print(f"Total Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")
