import torch
import math
from torch import nn

class SRGenerator(nn.Module):
    """Super-resolution generator using residual blocks and pixel shuffle upsampling."""
    def __init__(self, upscale_factor):
        super(SRGenerator, self).__init__()
        num_upsamples = int(math.log2(upscale_factor))

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            SRResidualBlock(64),
            SRResidualBlock(64),
            SRResidualBlock(64),
            SRResidualBlock(64),
            SRResidualBlock(64)
        )

        self.conv_bn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        upsampling_layers = [UpscaleBlock(64, 2) for _ in range(num_upsamples)]
        upsampling_layers.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upsample = nn.Sequential(*upsampling_layers)

    def forward(self, x):
        initial_feat = self.initial(x)
        res_out = self.res_blocks(initial_feat)
        combined = self.conv_bn(res_out) + initial_feat
        output = self.upsample(combined)
        return (torch.tanh(output) + 1) / 2


class SRDiscriminator(nn.Module):
    """Patch-based discriminator network for super-resolution GAN."""
    def __init__(self):
        super(SRDiscriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.features(x).view(x.size(0)))


class SRResidualBlock(nn.Module):
    """Residual block with two conv-batchnorm-prelu layers."""
    def __init__(self, channels):
        super(SRResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.layer(x)


class UpscaleBlock(nn.Module):
    """Upsample block using pixel shuffle."""
    def __init__(self, in_channels, scale_factor):
        super(UpscaleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)