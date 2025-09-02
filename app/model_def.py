# app/model_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

class ASPP(nn.Module):
    def __init__(self, in_channels=1280, out_channels=192, dropout=0.2):
        super().__init__()
        def branch3x3(rate):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
            )
        self.branch1 = nn.Sequential(  # 1x1
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )
        self.branch2 = branch3x3(6)
        self.branch3 = branch3x3(12)
        self.branch4 = branch3x3(18)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)
        gp = self.global_pool(x)
        gp = self.global_conv(gp)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([f1, f2, f3, f4, gp], dim=1)
        out = self.project(out)
        return out

class DeepLabV3Head(nn.Module):
    def __init__(self, aspp_out=192, num_classes=4, dropout=0.2, mid=96):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv1 = nn.Conv2d(aspp_out, mid, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, num_classes, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.conv2(x)
        return x

class MobileNetV2Encoder(nn.Module):
    """
    ВАЖНО: оставляем имена как в torchvision: features[0..18] с под-Seq 'conv',
    но регистрируем их под модулем 'base', чтобы ключи стали encoder.base.<idx>....
    Это совпадает с твоим state_dict: 'encoder.base.0.0', 'encoder.base.1.conv.0.0', ...
    """
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            mb = tvmodels.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        except TypeError:
            mb = tvmodels.mobilenet_v2(pretrained=pretrained)
        self.base = mb.features  # <-- ключевая строка для совпадения имен

    def forward(self, x):
        return self.base(x)  # [B, 1280, Hs, Ws]

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, dropout=0.2):
        super().__init__()
        self.encoder = MobileNetV2Encoder(pretrained=pretrained)
        self.aspp    = ASPP(in_channels=1280, out_channels=192, dropout=dropout)
        self.head    = DeepLabV3Head(aspp_out=192, num_classes=num_classes, dropout=dropout, mid=96)

    def forward(self, x):
        h0, w0 = x.shape[-2:]            # исходный размер
        f = self.encoder(x)              # [B, 1280, Hs, Ws]
        x = self.aspp(f)                 # [B, 192, Hs, Ws]
        logits = self.head(x)            # [B, C, Hs, Ws]
        logits = F.interpolate(logits, size=(h0, w0), mode='bilinear', align_corners=False)
        return logits

def create_deeplabv3(num_classes: int, pretrained: bool = True, dropout: float = 0.2):
    return DeepLabV3(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
