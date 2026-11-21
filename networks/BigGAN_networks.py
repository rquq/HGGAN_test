import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BigGAN_layers as layers
from .fusion import ContentContext
from networks.utils import init_weights, _len2mask


def G_arch(ch=64):
    """Generator channel and upsample schedule for 32px and 64px outputs."""
    return {
        32: {
            'in_channels': [ch * m for m in (8, 4, 2, 1)],
            'out_channels': [ch * m for m in (4, 2, 1, 1)],
            'upsample': [(2, 1), (2, 2), (2, 2), None],
            'resolution': [8, 16, 32, 32],
        },
        64: {
            'in_channels': [ch * m for m in (8, 4, 2, 1)],
            'out_channels': [ch * m for m in (4, 2, 1, 1)],
            'upsample': [(2, 1), (2, 2), (2, 2), (2, 2)],
            'resolution': [8, 16, 32, 64],
        },
    }


def D_arch(ch=64, input_nc=1):
    """Discriminator channel and downsample schedule for 32px and 64px inputs."""
    return {
        32: {
            'in_channels': [input_nc] + [ch * m for m in (1, 2, 4)],
            'out_channels': [ch * m for m in (1, 2, 4, 4)],
            'downsample': [True, True, True, False],
        },
        64: {
            'in_channels': [input_nc] + [ch * m for m in (1, 2, 4)],
            'out_channels': [ch * m for m in (1, 2, 4, 4)],
            'downsample': [True, True, True, False],
        },
    }


class BlockSpecificStyleProjection(nn.Module):
    """Project the global style vector into block-specific conditioning vectors."""

    def __init__(self, style_dim, num_blocks=4, which_linear=nn.Linear):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                which_linear(style_dim, style_dim),
                nn.SiLU(),
                which_linear(style_dim, style_dim),
            )
            for _ in range(num_blocks)
        ])

    def forward(self, global_style):
        if global_style.ndim == 3:
            if global_style.size(1) != 1:
                raise ValueError('GBlock conditioning expects a single global style token')
            global_style = global_style[:, 0]
        elif global_style.ndim != 2:
            raise ValueError('global style must have shape (B, D) or (B, 1, D)')
        return [proj(global_style) for proj in self.projections]


class Generator(nn.Module):
    def __init__(self, G_ch=64, style_dim=64, embed_dim=120,
                 bottom_width=4, bottom_height=4, resolution=64,
                 n_class=81, num_G_SVs=1, num_G_SV_itrs=1,
                 BN_eps=1e-5, SN_eps=1e-8, init='N02',
                 G_param='SN', bn_linear='SN', input_nc=1,
                 embed_pad_idx=0, embed_max_norm=1.0,
                 fusion_nhead=4, fusion_attn_dim=128, fusion_ffn_dim=360,
                 fusion_residual_init=0.25, wave_residual_init=0.1, **_unused):
        super().__init__()
        self.name = 'G'
        self.style_dim = style_dim
        self.bottom_width = bottom_width
        self.bottom_height = bottom_height
        self.resolution = resolution
        self.init = init
        self.arch = G_arch(G_ch)[resolution]
        self.activation = nn.ReLU(inplace=False)

        self.text_embedding = nn.Embedding(
            n_class, embed_dim, padding_idx=embed_pad_idx, max_norm=embed_max_norm
        )

        if G_param == 'SN':
            which_conv = functools.partial(
                layers.SNConv2d, kernel_size=3, padding=1,
                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs, eps=SN_eps,
            )
            which_depthwise = functools.partial(layers.SNDepthwiseConv2d, eps=SN_eps)
            which_linear = functools.partial(
                layers.SNLinear, num_svs=num_G_SVs, num_itrs=num_G_SV_itrs, eps=SN_eps,
            )
        else:
            which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            which_depthwise = lambda channels, **kw: nn.Conv2d(channels, channels, groups=channels, **kw)
            which_linear = nn.Linear

        bn_proj = functools.partial(which_linear, bias=False) if bn_linear == 'SN' else nn.Linear
        which_bn = functools.partial(layers.ccbn, which_linear=bn_proj, input_size=style_dim, eps=BN_eps)

        self.filter_linear = which_linear(
            embed_dim, self.arch['in_channels'][0] * bottom_width * bottom_height
        )
        self.content_context = ContentContext(
            embed_dim, nhead=fusion_nhead, attn_dim=fusion_attn_dim,
            ffn_dim=fusion_ffn_dim, residual_init=fusion_residual_init,
        )
        self.bssp = BlockSpecificStyleProjection(
            style_dim=style_dim, num_blocks=len(self.arch['in_channels']),
            which_linear=which_linear,
        )

        self.blocks = nn.ModuleList([
            layers.StyleWaveGBlock(
                in_channels=self.arch['in_channels'][i],
                out_channels=self.arch['out_channels'][i],
                style_dim=style_dim,
                which_conv=which_conv,
                which_bn=which_bn,
                which_depthwise=which_depthwise,
                which_linear=which_linear,
                activation=self.activation,
                upsample=(functools.partial(F.interpolate, scale_factor=scale) if scale is not None else None),
                wave_residual_init=wave_residual_init,
            )
            for i, scale in enumerate(self.arch['upsample'])
        ])

        self.output_layer = nn.Sequential(
            layers.bn(self.arch['out_channels'][-1], eps=BN_eps),
            self.activation,
            which_conv(self.arch['out_channels'][-1], input_nc),
        )

        if self.init != 'none':
            init_weights(self, self.init)
        # Re-apply stability inits after general weight initialization.
        self.content_context.reset_stability_parameters()
        for block in self.blocks:
            block.reset_stability_parameters()

    def forward(self, z, y, y_lens):
        if type(z) is not torch.Tensor:
            z = z.as_subclass(torch.Tensor)
        if z.ndim != 3 or z.size(1) != 1 or z.size(2) != self.style_dim:
            raise ValueError('G expects one global style code per image')

        ys = self.bssp(z[:, 0])
        content = self.text_embedding(y).float().to(y.device)
        y_mixed = self.content_context(content, y_lens=y_lens)

        token_positions = torch.arange(y.size(1), device=y.device).unsqueeze(0)
        valid_tokens = (token_positions < y_lens.unsqueeze(1)).unsqueeze(-1)
        valid_mask = valid_tokens.to(y_mixed.dtype)
        h = self.filter_linear(y_mixed * valid_mask) * valid_mask

        h = h.view(h.size(0), h.shape[1] * self.bottom_width, self.bottom_height, -1).permute(0, 3, 2, 1)

        len_scale = 1
        x_lens = y_lens * self.bottom_width
        for index, block in enumerate(self.blocks):
            in_lens = (x_lens * len_scale).clamp(min=1, max=h.size(-1))
            upsample_scale = self.arch['upsample'][index]
            width_scale = upsample_scale[1] if upsample_scale is not None else 1
            out_lens = (in_lens * width_scale).clamp(min=1)
            h = block(h, y=ys[index], x_lens=in_lens, out_x_lens=out_lens)

            valid_width = out_lens.clamp(max=h.size(-1))
            stage_mask = torch.arange(h.size(-1), device=h.device)[None, :] < valid_width[:, None]
            h = h * stage_mask[:, None, None, :].to(h.dtype)
            if upsample_scale is not None:
                len_scale *= width_scale

        output_lens = torch.div(y_lens * h.size(-2), 2, rounding_mode='trunc').clamp(min=1)
        out = self.output_layer[1](self.output_layer[0](h, valid_lens=output_lens))
        output_lens = output_lens.clamp(max=out.size(-1))
        output_mask = _len2mask(output_lens.int(), out.size(-1), torch.float32).to(out.device)[:, None, None, :]
        out = torch.tanh(self.output_layer[2](out * output_mask.to(out.dtype)))

        if not self.training:
            mask = output_mask.detach()
            out = out * mask + (mask - 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, D_ch=64, D_wide=True, resolution=64,
                 num_D_SVs=1, num_D_SV_itrs=1, SN_eps=1e-8,
                 output_dim=1, init='N02', D_param='SN',
                 input_nc=1, pooling_gain=32.0, **_unused):
        super().__init__()
        self.name = 'D'
        self.pooling_gain = float(pooling_gain)
        if not 0.0 < self.pooling_gain < float('inf'):
            raise ValueError('pooling_gain must be finite and positive')
        self.activation = nn.ReLU(inplace=False)
        self.arch = D_arch(D_ch, input_nc)[resolution]

        if D_param == 'SN':
            which_conv = functools.partial(
                layers.SNConv2d, kernel_size=3, padding=1,
                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=SN_eps,
            )
            which_linear = functools.partial(
                layers.SNLinear, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=SN_eps,
            )
        else:
            which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            which_linear = nn.Linear

        self.blocks = nn.ModuleList([
            layers.DBlock(
                in_channels=self.arch['in_channels'][i],
                out_channels=self.arch['out_channels'][i],
                which_conv=which_conv,
                wide=D_wide,
                activation=self.activation,
                preactivation=(i > 0),
                downsample=(nn.AvgPool2d(2) if self.arch['downsample'][i] else None),
            )
            for i in range(len(self.arch['out_channels']))
        ])
        self.linear = which_linear(self.arch['out_channels'][-1], output_dim)

        if init != 'none':
            init_weights(self, init)

    def _pool_features(self, h, h_lens=None):
        width_tokens = h.mean(dim=2).transpose(1, 2)
        if h_lens is None:
            return width_tokens.mean(dim=1)
        h_lens = h_lens.long().clamp(min=1, max=h.size(-1))
        valid_mask = _len2mask(h_lens, h.size(-1), torch.bool).to(h.device)
        return (width_tokens * valid_mask.unsqueeze(-1)).sum(dim=1) / h_lens[:, None]

    def forward(self, x, x_lens=None, y_lens=None, **kwargs):
        h = x
        len_scale = 1
        for index, block in enumerate(self.blocks):
            step_lens = torch.div(x_lens, len_scale, rounding_mode='trunc') if x_lens is not None else None
            h = block(h, x_len=step_lens)
            if self.arch['downsample'][index]:
                len_scale *= 2
        h = self.activation(h)
        h_lens = torch.div(x_lens, len_scale, rounding_mode='floor') if x_lens is not None else None
        return self.linear(self._pool_features(h, h_lens) * self.pooling_gain)


class StrokePatchBlock(nn.Module):
    """Anisotropic residual block specialized for handwriting strokes."""

    def __init__(self, in_channels, out_channels, which_conv, activation, which_depthwise):
        super().__init__()
        self.activation = activation
        self.conv_in = which_conv(in_channels, out_channels)
        self.horizontal = which_depthwise(out_channels, kernel_size=(1, 5), padding=(0, 2))
        self.vertical = which_depthwise(out_channels, kernel_size=(5, 1), padding=(2, 0))
        self.fuse = which_conv(out_channels, out_channels, kernel_size=1, padding=0)
        self.shortcut = which_conv(in_channels, out_channels, kernel_size=1, padding=0)
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.conv_in(self.activation(x))
        oriented = self.horizontal(self.activation(h)) + self.vertical(self.activation(h))
        h = h + self.fuse(self.activation(oriented))
        if h.size(-2) // 2 >= 4 and h.size(-1) // 2 >= 4:
            h = self.downsample(h)
            residual = self.downsample(residual)
        return h + residual


class PatchDiscriminator(nn.Module):
    """Lightweight local critic for stroke sharpness, joins, and texture."""

    def __init__(self, D_ch=32, D_max_ch=192, D_layers=3,
                 num_D_SVs=1, num_D_SV_itrs=1, SN_eps=1e-8,
                 output_dim=1, init='N02', D_param='SN', input_nc=1, **_unused):
        super().__init__()
        self.name = 'P'
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        if D_param == 'SN':
            which_conv = functools.partial(
                layers.SNConv2d, kernel_size=3, padding=1,
                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=SN_eps,
            )
            which_depthwise = functools.partial(layers.SNDepthwiseConv2d, eps=SN_eps)
        else:
            which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            which_depthwise = lambda channels, **kw: nn.Conv2d(channels, channels, groups=channels, **kw)

        self.stem = which_conv(input_nc, D_ch)
        blocks = []
        in_channels = D_ch
        for index in range(D_layers):
            out_channels = min(D_ch * (2 ** (index + 1)), D_max_ch)
            blocks.append(StrokePatchBlock(in_channels, out_channels, which_conv, self.activation, which_depthwise))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.logits = which_conv(in_channels, output_dim, kernel_size=1, padding=0)
        if init != 'none':
            init_weights(self, init)

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.logits(self.activation(h))
