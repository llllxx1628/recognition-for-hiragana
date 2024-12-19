import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for efficient feature extraction.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ChannelAttention(nn.Module):
    """
    Channel Attention module for emphasizing important channels.
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.global_avg_pool(x).view(b, c))
        max_out = self.fc(self.global_max_pool(x).view(b, c))
        combined = avg_out + max_out
        return x * self.sigmoid(combined).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """
    Spatial Attention module for emphasizing important spatial regions.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual Block with depthwise separable convolution.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x, inplace=True)

class MultiScaleConvolution(nn.Module):
    """
    Multi-scale convolution block for extracting features at different receptive fields.
    """

    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvolution, self).__init__()
        self.branch3x3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch5x5 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return (self.branch3x3(x) + self.branch5x5(x)) / 2

class HiraganaRecognitionNet(nn.Module):
    """
    Neural network for recognizing Hiragana characters.
    """

    def __init__(self, num_classes=49):
        super(HiraganaRecognitionNet, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.block1 = MultiScaleConvolution(32, 64)
        self.residual1 = ResidualBlock(64, 64)
        self.attention1 = CBAM(64)

        self.block2 = MultiScaleConvolution(64, 128)
        self.residual2 = ResidualBlock(128, 128, stride=2)
        self.attention2 = CBAM(128)

        self.block3 = MultiScaleConvolution(128, 256)
        self.residual3 = ResidualBlock(256, 256, stride=2)
        self.attention3 = CBAM(256)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.block1(x)
        x = self.residual1(x)
        x = self.attention1(x)

        x = self.block2(x)
        x = self.residual2(x)
        x = self.attention2(x)

        x = self.block3(x)
        x = self.residual3(x)
        x = self.attention3(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  
        return self.classifier(x)

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss to prevent overconfidence in predictions.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        with torch.no_grad():
            true_dist = torch.full_like(logits, self.smoothing / (logits.size(1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=1), dim=1))
