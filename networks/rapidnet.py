"""RapidNet-inspired generator blocks adapted for conditional handwriting synthesis.

The original RapidNet MLDC block is an image-recognition feature mixer.  This
adaptation preserves the useful multi-level dilated and large-kernel depthwise
paths while retaining HiGAN+'s conditional normalization, residual upsampling,
and spectral-normalized convolution factories.
"""

import torch
import torch.nn as nn


class RapidMLDCMixer(nn.Module):
    """Efficient multi-level dilated mixer with a large-kernel convolutional FFN."""

    def __init__(self, channels, which_conv, expansion=2):
        super().__init__()
        hidden_channels = channels * expansion

        # RapidNet's conditional positional encoding and MLDC paths. Depthwise
        # spatial kernels keep the high-resolution generator stages economical;
        # pointwise projections provide full cross-channel interaction.
        self.position = which_conv(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.pre = which_conv(channels, channels, kernel_size=1, padding=0)
        self.dilated_2 = which_conv(
            channels, channels, kernel_size=3, padding=2,
            dilation=2, groups=channels,
        )
        self.dilated_3 = which_conv(
            channels, channels, kernel_size=3, padding=3,
            dilation=3, groups=channels,
        )
        self.mix = which_conv(channels, channels, kernel_size=1, padding=0)

        self.ffn_spatial = which_conv(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.ffn_in = which_conv(
            channels, hidden_channels, kernel_size=1, padding=0
        )
        self.ffn_out = which_conv(
            hidden_channels, channels, kernel_size=1, padding=0
        )
        self.activation = nn.GELU()

        # Non-zero residual scales let every new path learn on the first update
        # without abruptly replacing the reliable conditioned residual stream.
        self.mixer_scale = nn.Parameter(torch.full((channels, 1, 1), 0.1))
        self.ffn_scale = nn.Parameter(torch.full((channels, 1, 1), 0.1))

    def forward(self, x):
        positioned = x + self.position(x)
        projected = self.pre(positioned)
        mixed = self.activation(self.dilated_2(projected))
        mixed = mixed + self.activation(self.dilated_3(projected))
        x = x + self.mixer_scale * self.mix(mixed)

        ffn = self.ffn_spatial(x)
        ffn = self.ffn_out(self.activation(self.ffn_in(ffn)))
        return x + self.ffn_scale * ffn


class ConditionedRapidBlock(nn.Module):
    """RapidNet-style MLDC block with HiGAN+ cCBN and anisotropic upsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        which_conv,
        which_bn,
        activation,
        upsample=None,
        expansion=2,
    ):
        super().__init__()
        self.upsample = upsample
        self.activation = activation

        self.bn_in = which_bn(in_channels)
        self.project = which_conv(
            in_channels, out_channels, kernel_size=1, padding=0
        )
        self.bn_mixer = which_bn(out_channels)
        self.mldc = RapidMLDCMixer(
            out_channels, which_conv=which_conv, expansion=expansion
        )

        self.learnable_shortcut = in_channels != out_channels or upsample is not None
        if self.learnable_shortcut:
            self.shortcut = which_conv(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x, y, **kwargs):
        residual = x
        h = self.activation(self.bn_in(x, y))

        if self.upsample is not None:
            h = self.upsample(h)
            residual = self.upsample(residual)

        h = self.project(h)
        h = self.activation(self.bn_mixer(h, y))
        h = self.mldc(h)

        if self.learnable_shortcut:
            residual = self.shortcut(residual)
        return residual + h
