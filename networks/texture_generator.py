"""Lightweight style- and frequency-aware refinement for MAIN's generator.

The base generator remains DEV's proven conditioned GBlock stack.  This module
adds a deliberately low-gain residual that uses global style, the first two
moments of local style tokens, Global Response Normalization, and an explicit
high-pass output.  It targets handwriting texture without changing geometry or
introducing another reconstruction/adversarial loss.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalResponseNorm2d(nn.Module):
    """ConvNeXt-V2 Global Response Normalization for NCHW features."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, features):
        response = torch.linalg.vector_norm(
            features.float(), ord=2, dim=(2, 3), keepdim=True
        ).to(features.dtype)
        normalized = response / (
            response.mean(dim=1, keepdim=True) + self.eps
        )
        return features + self.gamma * (features * normalized) + self.beta


class StyleFrequencyRefinement(nn.Module):
    """Bounded style-distribution modulation and high-frequency synthesis."""

    def __init__(
        self,
        channels,
        style_dim,
        output_channels,
        which_conv,
        style_limit=0.15,
        max_detail_gain=0.15,
        initial_detail_gain=0.03,
    ):
        super().__init__()
        if not 0.0 < initial_detail_gain < max_detail_gain < 1.0:
            raise ValueError(
                'detail gains must satisfy 0 < initial < maximum < 1'
            )
        self.channels = int(channels)
        self.style_dim = int(style_dim)
        self.style_limit = float(style_limit)
        self.max_detail_gain = float(max_detail_gain)

        # Oriented depthwise paths model pen edges and stroke direction cheaply.
        self.horizontal = which_conv(
            channels, channels, kernel_size=(1, 7), padding=(0, 3),
            groups=channels,
        )
        self.vertical = which_conv(
            channels, channels, kernel_size=(7, 1), padding=(3, 0),
            groups=channels,
        )
        self.local = which_conv(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.channel_mix = which_conv(
            channels, channels, kernel_size=1, padding=0
        )
        self.grn = GlobalResponseNorm2d(channels)

        # Global identity plus the mean and standard deviation of local slots
        # describe both the writer-level style and its local distribution.
        self.style_affine = nn.Linear(style_dim * 3, channels * 2)
        # MogaNet/InceptionNeXt-inspired multi-branch gating lets each writer
        # distribution choose horizontal, vertical, or compact local texture.
        self.branch_affine = nn.Linear(style_dim * 3, channels * 3)
        self.to_detail = which_conv(
            channels, output_channels, kernel_size=1, padding=0
        )
        initial_ratio = initial_detail_gain / max_detail_gain
        self.detail_gain_logit = nn.Parameter(torch.tensor(
            math.log(initial_ratio / (1.0 - initial_ratio))
        ))

    def reset_stability_parameters(self):
        # Start as identity modulation.  The non-zero bounded detail gain lets
        # the spatial branch learn immediately while preserving the base G.
        nn.init.zeros_(self.style_affine.weight)
        nn.init.zeros_(self.style_affine.bias)
        nn.init.zeros_(self.branch_affine.weight)
        nn.init.zeros_(self.branch_affine.bias)
        nn.init.zeros_(self.grn.gamma)
        nn.init.zeros_(self.grn.beta)

    def style_descriptor(self, style_tokens):
        if style_tokens.ndim != 3 or style_tokens.size(-1) != self.style_dim:
            raise ValueError('style_tokens must have shape (B, S, style_dim)')
        global_style = style_tokens[:, 0]
        if style_tokens.size(1) > 1:
            local_style = style_tokens[:, 1:]
            local_mean = local_style.mean(dim=1)
            local_variance = (
                local_style - local_mean.unsqueeze(1)
            ).square().mean(dim=1)
            local_std = torch.sqrt(local_variance + 1e-6)
        else:
            local_mean = torch.zeros_like(global_style)
            local_std = torch.zeros_like(global_style)
        return torch.cat([global_style, local_mean, local_std], dim=-1)

    def forward(self, features, style_tokens):
        if features.ndim != 4 or features.size(1) != self.channels:
            raise ValueError('features must have shape (B, channels, H, W)')
        descriptor = self.style_descriptor(style_tokens)
        scale, shift = self.style_affine(descriptor).chunk(2, dim=-1)
        scale = 1.0 + self.style_limit * torch.tanh(scale)
        # Feature-demodulation keeps style strength from changing feature energy
        # and producing the stray lines seen in the old MAIN checkpoints.
        scale = scale / torch.sqrt(scale.square().mean(dim=1, keepdim=True) + 1e-6)
        shift = self.style_limit * torch.tanh(shift)
        branch_weights = torch.softmax(
            self.branch_affine(descriptor).view(-1, 3, self.channels), dim=1
        )

        horizontal = self.horizontal(features)
        vertical = self.vertical(features)
        local = self.local(features)
        detail_features = (
            branch_weights[:, 0].unsqueeze(-1).unsqueeze(-1) * horizontal
            + branch_weights[:, 1].unsqueeze(-1).unsqueeze(-1) * vertical
            + branch_weights[:, 2].unsqueeze(-1).unsqueeze(-1) * local
        )
        detail_features = F.silu(detail_features)
        detail_features = self.channel_mix(detail_features)
        detail_features = self.grn(detail_features)
        detail_features = (
            detail_features * scale.unsqueeze(-1).unsqueeze(-1)
            + shift.unsqueeze(-1).unsqueeze(-1)
        )
        raw_detail = self.to_detail(F.silu(detail_features))

        # A residual high-pass band adds pen-edge texture without changing the
        # low-frequency word shape produced by the base generator.
        padded = F.pad(raw_detail, (1, 1, 1, 1), mode='reflect')
        low_frequency = F.avg_pool2d(padded, kernel_size=3, stride=1)
        high_frequency = raw_detail - low_frequency
        gain = self.max_detail_gain * torch.sigmoid(self.detail_gain_logit)
        return gain * high_frequency

    def detail_gain(self):
        return self.max_detail_gain * torch.sigmoid(self.detail_gain_logit)
