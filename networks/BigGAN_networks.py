# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BigGAN_layers as layers
from .fusion import StyleContentAttentionFusion
from .rapidnet import ConditionedRapidBlock
from networks.utils import init_weights, _len2mask

# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}

    arch[64] = {'in_channels': [ch * item for item in [8, 4, 2, 1]],
                'out_channels': [ch * item for item in [4, 2, 1, 1]],
                'upsample': [(2,1), (2,2), (2,2), (2,2)],
                'resolution': [8, 16, 32, 64],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(2, 7)}}
    return arch


class BlockSpecificStyleProjection(nn.Module):
    """Project the explicit global style token for each GBlock."""

    def __init__(self, style_dim, num_blocks=4, style_chunk_size=32,
                 which_linear=nn.Linear):
        super().__init__()
        self.num_blocks = num_blocks
        self.projections = nn.ModuleList([
            nn.Sequential(
                which_linear(style_dim, style_chunk_size),
                nn.SiLU(),
                which_linear(style_chunk_size, style_chunk_size),
            )
            for _ in range(num_blocks)
        ])

    def forward(self, global_style):
        if global_style.ndim == 3:
            if global_style.size(1) != 1:
                raise ValueError('GBlock conditioning accepts only the global style token')
            global_style = global_style[:, 0]
        elif global_style.ndim != 2:
            raise ValueError('global style must have shape (B, D) or (B, 1, D)')
        return [projection(global_style) for projection in self.projections]


class Generator(nn.Module):
    def __init__(self, G_ch=64, style_dim=32, embed_dim=120,
                 bottom_width=4, bottom_height=4, resolution=128,
                 G_kernel_size=3, G_attn='64', n_class=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 BN_eps=1e-5, SN_eps=1e-12, G_fp16=False,
                 init='ortho', G_param='SN', norm_style='bn', bn_linear='embed', input_nc=3,
                 embed_pad_idx=0, embed_max_norm=1.0, fusion_gate_init=0.25
                 ):
        super(Generator, self).__init__()
        dim_z = style_dim
        self.style_dim = style_dim
        self.name = 'G'
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        self.embed_dim = embed_dim
        # The initial width dimensions
        self.bottom_width = bottom_width
        # The initial height dimension
        self.bottom_height = bottom_height
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_class
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]
        self.bn_linear = bn_linear

        self.z_chunk_size = self.dim_z

        self.text_embedding = nn.Embedding(self.n_classes, self.embed_dim,
                                           padding_idx=embed_pad_idx,
                                           max_norm=embed_max_norm)

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        if self.bn_linear=='SN':
            bn_linear = functools.partial(self.which_linear, bias=False)
        else:
            bn_linear = nn.Linear

        self.which_bn = functools.partial(layers.ccbn,
                                          which_linear=bn_linear,
                                          cross_replica=self.cross_replica,
                                          mybn=self.mybn,
                                          input_size=self.z_chunk_size,
                                          norm_style=self.norm_style,
                                          eps=self.BN_eps)

        self.filter_linear = self.which_linear(self.embed_dim,
                                        self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
        self.style_content_mix = StyleContentAttentionFusion(
            self.embed_dim, self.style_dim, vocab_size=self.n_classes
        )
        if not 0.0 < fusion_gate_init < 1.0:
            raise ValueError('fusion_gate_init must be strictly between 0 and 1')
        # A channel-wise, non-zero gate gives fusion gradients from the first
        # update while retaining the reliable unfused content path.
        gate_logit = torch.logit(torch.tensor(float(fusion_gate_init)))
        self.fusion_gate_logits = nn.Parameter(
            torch.full((self.embed_dim,), gate_logit.item())
        )

        self.bssp = BlockSpecificStyleProjection(style_dim=self.style_dim, num_blocks=len(self.arch['in_channels']), style_chunk_size=self.z_chunk_size, which_linear=self.which_linear)

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[ConditionedRapidBlock(
                in_channels=self.arch['in_channels'][index],
                out_channels=self.arch['out_channels'][index],
                which_conv=self.which_conv,
                which_bn=self.which_bn,
                activation=self.activation,
                upsample=functools.partial(
                    F.interpolate,
                    scale_factor=self.arch['upsample'][index],
                    mode='bilinear',
                    align_corners=False,
                ),
            )]]

            if self.arch['attention'][self.arch['resolution'][index]]:
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], input_nc))

        # Initialize weights. Optionally skip init for testing.
        if self.init != 'none':
            init_weights(self, self.init)
        # General initialization touches fusion Linear weights; restore its
        # identity-like nonzero residual handoffs afterwards.
        self.style_content_mix.reset_stability_parameters()

    def forward(self, z, y, y_lens):
        # Only the explicit global token may bypass character-level fusion.
        # Local tokens must travel through the aligned fusion path.
        ys = self.bssp(z[:, 0])

        char_ids = y
        content = self.text_embedding(y).float().to(y.device)
        fused_content = self.style_content_mix(
            content, z, char_ids=char_ids, y_lens=y_lens
        )
        token_positions = torch.arange(y.size(1), device=y.device).unsqueeze(0)
        valid_tokens = (token_positions < y_lens.unsqueeze(1)).unsqueeze(-1)
        fusion_gate = torch.sigmoid(self.fusion_gate_logits).view(1, 1, -1)
        y_mixed = content + fusion_gate * (fused_content - content)
        y_mixed = y_mixed * valid_tokens.to(y_mixed.dtype)
        h = self.filter_linear(y_mixed) * valid_tokens.to(y_mixed.dtype)

        # Reshape - when y is not a single class value but rather an array of classes, the reshape is needed to create
        # a separate vertical patch for each input.
        h = h.view(h.size(0), h.shape[1] * self.bottom_width, self.bottom_height, -1)
        h = h.permute(0, 3, 2, 1)

        # Loop over blocks
        len_scale = 1
        x_lens = y_lens * self.bottom_width
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                if isinstance(block, layers.Attention):
                    h = block(h, x_lens=x_lens * len_scale)
                else:
                    h = block(h, y=ys[index])
            len_scale *= self.arch['upsample'][index][1]

        # Apply batchnorm-relu-conv-tanh at output
        output = torch.tanh(self.output_layer(h))

        # Mask blanks
        if not self.training:
            out_lens = torch.div(y_lens * output.size(-2), 2, rounding_mode='trunc')
            mask = _len2mask(out_lens.int(), output.size(-1), torch.float32).to(z.device).detach()
            mask = mask.unsqueeze(1).unsqueeze(1)
            output = output * mask + (mask - 1)

        return output

    def fusion_strength(self):
        return torch.sigmoid(self.fusion_gate_logits).mean()

    def _info_attention(self):
        attn_index = -1
        for index in range(len(self.arch['out_channels'])):
            if self.arch['attention'][self.arch['resolution'][index]]:
                attn_index = index

        if attn_index == -1:
            return []

        attn_layer = self.blocks[attn_index][-1]
        out = []
        if hasattr(attn_layer, 'attn1') and hasattr(attn_layer, 'attn2'):
            for l in [attn_layer.attn1, attn_layer.attn2]:
                out.append({'out': getattr(l, '_vis_out', None), 'gamma': l.gamma.item()})
        elif hasattr(attn_layer, 'gamma'):
            out.append({'gamma': attn_layer.gamma.item()})
        return out


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64', input_nc=3):
    arch = {}

    arch[32] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4]],
                'out_channels': [item * ch for item in [1, 2, 4, 4]],
                'downsample': [True] * 3 + [False],
                'resolution': [8, 4, 4, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 5)}}
    arch[33] = {'in_channels': [input_nc] + [ch * item for item in [1, 1, 2, 2, 4, 4]],
                'out_channels': [item * ch for item in [1, 1, 2, 2, 4, 4, 4]],
                'downsample': [False, True, False, True, False, True, False],
                'resolution': [8, 8, 4, 4, 4, 4, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 9)}}
    arch[64] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4]],
               'out_channels': [item * ch for item in [1, 2, 4, 4]],
               'downsample': [True] * 3 + [False],
               'resolution': [32, 16, 8, 8],
               'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 7)}}

    return arch


class WidthContextMixer(nn.Module):
    """Low-resolution width attention for whole-word visual coherence."""

    def __init__(self, channels, num_heads, which_linear):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError('discriminator channels must be divisible by width_heads')
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(channels)
        self.qkv = which_linear(channels, channels * 3, bias=False)
        self.proj = which_linear(channels, channels, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, tokens, valid_mask=None):
        batch, width, channels = tokens.shape
        qkv = self.qkv(self.norm(tokens))
        qkv = qkv.view(
            batch, width, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if valid_mask is not None:
            scores = scores.masked_fill(
                ~valid_mask[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).reshape(batch, width, channels)
        context = self.proj(context)
        if valid_mask is not None:
            context = context * valid_mask.unsqueeze(-1).to(context.dtype)
        return tokens + torch.tanh(self.residual_scale) * context


class Discriminator(nn.Module):
    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                 D_attn='64', num_D_SVs=1, num_D_SV_itrs=1,
                 D_activation=nn.ReLU(inplace=False), SN_eps=1e-12,
                 output_dim=1, D_fp16=False, init='ortho', D_param='SN',
                 input_nc=3, width_context=False, width_heads=4):
        super(Discriminator, self).__init__()
        self.name = 'D'
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Attention?
        self.attention = D_attn
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention, input_nc)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            if self.arch['attention'][self.arch['resolution'][index]]:
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        self.width_context = (
            WidthContextMixer(
                self.arch['out_channels'][-1], width_heads, self.which_linear
            )
            if width_context else None
        )
        # Embedding for projection discrimination
        # self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if self.init != 'none':
            self = init_weights(self, self.init)

    def forward(self, x, x_lens=None, y_lens=None,  **kwargs):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        len_scale = 1
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, x_len=torch.div(x_lens, len_scale, rounding_mode='trunc') if x_lens is not None else None)
            len_scale *= 2 if self.arch['downsample'][index] else 1
        # Preserve vertical evidence while allowing one cheap, low-resolution
        # interaction across the complete valid word width.
        h = self.activation(h)
        width_tokens = torch.sum(h, dim=2).transpose(1, 2)
        valid_mask = None
        if x_lens is not None:
            h_lens = torch.div(
                x_lens * h.size(-1), x.size(-1), rounding_mode='trunc'
            ).long().clamp_(1, h.size(-1))
            valid_mask = _len2mask(
                h_lens.int(), h.size(-1), torch.bool
            ).to(x.device).detach()

        if self.width_context is not None:
            width_tokens = self.width_context(width_tokens, valid_mask)

        if valid_mask is None:
            h = torch.sum(width_tokens, dim=1)
        else:
            h = torch.sum(
                width_tokens * valid_mask.unsqueeze(-1).to(width_tokens.dtype),
                dim=1,
            )
            normalizer = y_lens if y_lens is not None else h_lens
            h = h / torch.clamp(normalizer, min=1).unsqueeze(dim=-1)

        # Get initial class-unconditional output
        out = self.linear(h)

        return out


class StrokePatchBlock(nn.Module):
    """Anisotropic residual block specialized for handwriting strokes."""

    def __init__(self, in_channels, out_channels, which_conv, activation):
        super().__init__()
        self.activation = activation
        self.conv_in = which_conv(in_channels, out_channels)
        self.horizontal = which_conv(
            out_channels, out_channels, kernel_size=(1, 5),
            padding=(0, 2), groups=out_channels,
        )
        self.vertical = which_conv(
            out_channels, out_channels, kernel_size=(5, 1),
            padding=(2, 0), groups=out_channels,
        )
        self.fuse = which_conv(
            out_channels, out_channels, kernel_size=1, padding=0
        )
        self.shortcut = which_conv(
            in_channels, out_channels, kernel_size=1, padding=0
        )
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.conv_in(self.activation(x))
        oriented = self.horizontal(self.activation(h))
        oriented = oriented + self.vertical(self.activation(h))
        h = h + self.fuse(self.activation(oriented))
        return self.downsample(h) + self.downsample(residual)


class PatchDiscriminator(nn.Module):
    """Lightweight spatial critic for stroke shape, joins, and local texture."""

    def __init__(
        self,
        D_ch=32,
        D_max_ch=192,
        D_layers=3,
        num_D_SVs=1,
        num_D_SV_itrs=1,
        SN_eps=1e-12,
        output_dim=1,
        init='ortho',
        D_param='SN',
        input_nc=1,
    ):
        super().__init__()
        self.name = 'P'
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        if D_param == 'SN':
            which_conv = functools.partial(
                layers.SNConv2d,
                kernel_size=3,
                padding=1,
                num_svs=num_D_SVs,
                num_itrs=num_D_SV_itrs,
                eps=SN_eps,
            )
        else:
            which_conv = functools.partial(
                nn.Conv2d, kernel_size=3, padding=1
            )

        self.stem = which_conv(input_nc, D_ch)
        blocks = []
        in_channels = D_ch
        for index in range(D_layers):
            out_channels = min(D_ch * (2 ** (index + 1)), D_max_ch)
            blocks.append(
                StrokePatchBlock(
                    in_channels, out_channels, which_conv, self.activation
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.logits = which_conv(
            in_channels, output_dim, kernel_size=1, padding=0
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, x, **kwargs):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.logits(self.activation(h))
