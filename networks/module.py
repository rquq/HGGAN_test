import numpy as np
import torch
from torch import nn
import functools
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM, Identity
from networks.utils import _len2mask, init_weights
from lib.alphabet import get_alphabet_size
import torch.nn.functional as F


def _input_stride_for_height(img_height, base_height=32):
    """Scale encoder stride from image geometry, independent of dataset name."""
    height = int(img_height)
    base_height = int(base_height)
    if height < 1 or base_height < 1:
        raise ValueError('image and base heights must be positive')
    return max(1, height // base_height)


def _mask_width(x, lengths, value=0.0, cache=None):
    """Exclude batch padding, never background pixels inside a valid image."""
    key = (id(lengths), x.size(-1), x.ndim)
    if cache is not None and key in cache:
        invalid = cache[key][1]
    else:
        invalid = torch.arange(x.size(-1), device=x.device)[None, :] >= lengths[:, None]
        invalid = invalid.view(x.size(0), *([1] * (x.ndim - 2)), x.size(-1))
        if cache is not None:
            # Retain lengths with its mask so object IDs cannot be reused.
            cache[key] = (lengths, invalid)
    return x.masked_fill(invalid, value)


def _output_widths(lengths, layer):
    """Propagate exact CNN geometry, including explicit padding offsets."""
    if isinstance(layer, (nn.Sequential, ActFirstResBlock)):
        children = layer if isinstance(layer, nn.Sequential) else (layer.conv_0, layer.conv_1)
        for child in children:
            lengths = _output_widths(lengths, child)
    elif isinstance(layer, Conv2dBlock):
        conv = layer.conv
        delta = (sum(getattr(layer.pad, 'padding', (0, 0))[:2]) + 2 * conv.padding[-1]
                 - conv.dilation[-1] * (conv.kernel_size[-1] - 1))
        stride = conv.stride[-1]
        # Most residual convolutions preserve width. Avoid GPU length/mask
        # recomputation for those layers, and reuse their masks within a forward.
        if stride == 1:
            return lengths if delta == 0 else lengths + delta
        lengths = torch.div(lengths + delta + stride - 1, stride, rounding_mode='floor')
    elif isinstance(layer, (nn.ConstantPad2d, nn.ReflectionPad2d, nn.ReplicationPad2d)):
        lengths = lengths + layer.padding[0] + layer.padding[1]
    elif isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
        def last(value):
            return value[-1] if isinstance(value, (tuple, list)) else value
        kernel = last(layer.kernel_size)
        stride = last(layer.stride) if layer.stride is not None else kernel
        padding = last(layer.padding)
        dilation = last(getattr(layer, 'dilation', 1))
        numerator = lengths + 2 * padding - dilation * (kernel - 1) - 1
        if getattr(layer, 'ceil_mode', False):
            numerator = numerator + stride - 1
        lengths = torch.div(numerator, stride, rounding_mode='floor') + 1
    return lengths


def _valid_backbone_layer(layer, x, lengths, mask_cache=None):
    """Batched equivalent of cropped forwards for the frozen style backbone.

    Mask intermediate activations too: a final mask cannot undo convolutions
    which have already read padded activations. Existing pretraining forwards
    without lengths keep their original BatchNorm behavior.
    """
    if isinstance(layer, nn.Sequential):
        for child in layer:
            x, lengths = _valid_backbone_layer(child, x, lengths, mask_cache)
        return x, lengths
    if isinstance(layer, ActFirstResBlock):
        shortcut = _valid_backbone_layer(layer.conv_s, x, lengths, mask_cache)[0] if layer.learned_shortcut else x
        out, out_lengths = _valid_backbone_layer(layer.conv_0, x, lengths, mask_cache)
        out = layer.dropout(out)
        out, out_lengths = _valid_backbone_layer(layer.conv_1, out, out_lengths, mask_cache)
        # Both branches are already masked; addition preserves their zero tail.
        return shortcut + out, out_lengths
    out_lengths = _output_widths(lengths, layer)
    if isinstance(layer, Conv2dBlock):
        if layer.activation_first and layer.activation is not None:
            x = layer.activation(x)
        # The previous layer already masked its output. These activations map
        # zero to zero, so another input mask would add a redundant GPU kernel.
        x = layer.conv(layer.pad(x))
        if layer.norm is not None:
            x = layer.norm(x)
        if not layer.activation_first and layer.activation is not None:
            x = layer.activation(x)
    else:
        x = layer(x)
        if isinstance(layer, (nn.ConstantPad2d, nn.ReLU)):
            # Explicit padding extends the existing canonical tail; ReLU
            # preserves zero. Neither operation can mix it into valid pixels.
            return x, out_lengths
    fill = layer.value if isinstance(layer, nn.ConstantPad2d) else 0.0
    return _mask_width(x, out_lengths, fill, mask_cache), out_lengths


class _StyleContextBlock(nn.Module):
    """Width context followed by expressive, gated cross-channel mixing."""

    def __init__(self, in_dim, dilation):
        super().__init__()
        hidden_dim = in_dim * 2
        self.depthwise = nn.Conv1d(
            in_dim, in_dim, kernel_size=7, padding=0,
            dilation=dilation, groups=in_dim,
        )
        # Channel normalization at each position is independent of batch padding.
        self.norm = nn.LayerNorm(in_dim)
        self.expand = nn.Conv1d(in_dim, hidden_dim * 2, kernel_size=1)
        self.project = nn.Conv1d(hidden_dim, in_dim, kernel_size=1)
        # Preserve backbone evidence while allowing every context weight to
        # receive gradients from update one (unlike a zero residual gate).
        self.residual_scale = nn.Parameter(torch.full((in_dim,), 0.1))

    def forward(self, x, valid_mask=None):
        invalid = ~valid_mask[:, None, :].bool() if valid_mask is not None else None
        if invalid is not None:
            x = x.masked_fill(invalid, 0.0)
        padding = self.depthwise.dilation[0] * (self.depthwise.kernel_size[0] - 1) // 2
        if valid_mask is None:
            context_input = F.pad(x, (padding, padding), mode='replicate')
        else:
            # Extend each sample's actual feature boundary, not the padded
            # canvas edge. Valid positions are a left-aligned prefix.
            last_valid = valid_mask.long().sum(dim=-1).clamp_min(1) - 1
            positions = torch.arange(-padding, x.size(-1) + padding, device=x.device)
            indices = torch.minimum(positions.clamp_min(0)[None, :], last_valid[:, None])
            context_input = x.gather(2, indices[:, None, :].expand(-1, x.size(1), -1))
        context = self.depthwise(context_input)
        context = self.norm(context.transpose(1, 2)).transpose(1, 2)
        value, gate = self.expand(context).chunk(2, dim=1)
        context = self.project(value * F.silu(gate))
        out = x + self.residual_scale[None, :, None] * context
        return out.masked_fill(invalid, 0.0) if invalid is not None else out


class GatedStyleContext(nn.Module):
    """Gated reference-style context using two sequential residual blocks.

    Two residual blocks, with dilation 1 then 2, cover 19 feature positions:
    the old widest dilated-plus-fusion receptive field without four dense
    parallel branches. Global pooling and multi-scale local-token attention
    remain in StyleEncoder, rather than being duplicated here.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            _StyleContextBlock(in_dim, dilation=1),
            _StyleContextBlock(in_dim, dilation=2),
        ])

    def forward(self, x, valid_mask=None, **kwargs):
        for block in self.blocks:
            x = block(x, valid_mask=valid_mask)
        return x


class StyleBackbone(nn.Module):
    def __init__(self, resolution=16, max_dim=256, in_channel=1, init='N02', dropout=0.0, norm='bn', img_height=64, **kwargs):
        super(StyleBackbone, self).__init__()
        # Derive the front-end stride from input geometry and keep length
        # metadata in lock-step with the CNN at every supported resolution.
        init_stride = _input_stride_for_height(img_height)
        self.reduce_len_scale = 8 * init_stride
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, init_stride, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])
        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.layer_name_mapping = {
            '9': "feat2",
            '13': "feat3",
            '16': "feat4",
        }

        self.cnn_ctc = nn.Sequential(
            nn.ReLU(),
            Conv2dBlock(df, df, 3, 1, 0,
                        norm=norm,
                        activation='relu')
        )
        if init != 'none':
            init_weights(self, init)

    def feature_lengths(self, image_lengths):
        lengths = image_lengths.long()
        feature_lengths = []
        for name, layer in self.cnn_backbone._modules.items():
            lengths = _output_widths(lengths, layer)
            if name in self.layer_name_mapping:
                feature_lengths.append(lengths)
        return _output_widths(lengths, self.cnn_ctc), feature_lengths

    def forward(self, x, ret_feats=False, x_lens=None):
        lengths = x_lens.to(device=x.device, dtype=torch.long) if x_lens is not None else None
        mask_cache = {}
        if lengths is not None:
            x = _mask_width(x, lengths, -1.0, mask_cache)
        feats = []
        for name, layer in self.cnn_backbone._modules.items():
            if lengths is None:
                x = layer(x)
            else:
                x, lengths = _valid_backbone_layer(layer, x, lengths, mask_cache)
            if ret_feats and name in self.layer_name_mapping:
                feats.append(x)

        out = self.cnn_ctc(x) if lengths is None else _valid_backbone_layer(self.cnn_ctc, x, lengths, mask_cache)[0]
        if out.dim() == 4:
            out = out.squeeze(2) if out.size(2) == 1 else out.mean(dim=2)

        return out, feats


def get_1d_sinusoidal_embeddings(length, dim, device):
    pe = torch.zeros(length, dim, device=device)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(np.log(10000.0) / dim))
    pos = torch.arange(0, length, device=device).float().unsqueeze(1)
    pe[:, 0::2] = torch.sin(pos * div_term)
    if dim % 2 == 0:
        pe[:, 1::2] = torch.cos(pos * div_term)
    else:
        pe[:, 1::2] = torch.cos(pos * div_term[:dim//2])
    return pe


def get_2d_sinusoidal_embeddings(height, width, dim, device):
    pe = torch.zeros(height, width, dim, device=device)
    d_h = dim // 2
    d_w = dim - d_h

    # Height embeddings
    div_term_h = torch.exp(torch.arange(0, d_h, 2, device=device).float() * -(np.log(10000.0) / d_h))
    pos_h = torch.arange(0, height, device=device).float().unsqueeze(1)
    pe_h = torch.zeros(height, d_h, device=device)
    pe_h[:, 0::2] = torch.sin(pos_h * div_term_h)
    if d_h % 2 == 0:
        pe_h[:, 1::2] = torch.cos(pos_h * div_term_h)
    else:
        pe_h[:, 1::2] = torch.cos(pos_h * div_term_h[:d_h//2])

    # Width embeddings
    div_term_w = torch.exp(torch.arange(0, d_w, 2, device=device).float() * -(np.log(10000.0) / d_w))
    pos_w = torch.arange(0, width, device=device).float().unsqueeze(1)
    pe_w = torch.zeros(width, d_w, device=device)
    pe_w[:, 0::2] = torch.sin(pos_w * div_term_w)
    if d_w % 2 == 0:
        pe_w[:, 1::2] = torch.cos(pos_w * div_term_w)
    else:
        pe_w[:, 1::2] = torch.cos(pos_w * div_term_w[:d_w//2])

    pe[:, :, :d_h] = pe_h.unsqueeze(1).expand(-1, width, -1)
    pe[:, :, d_h:] = pe_w.unsqueeze(0).expand(height, -1, -1)
    return pe


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def _gradient_reverse(x, scale):
    return _GradientReverse.apply(x, scale)




class StyleEncoder(nn.Module):
    """Encode a global latent and optional spatial reference evidence."""

    def __init__(self, style_dim=32, in_dim=256, init='N02',
                 num_style_tokens=1, backbone_channels=(64, 128, 256),
                 n_class=None, content_grl=1.0, feature_scales=None,
                 memory_dim=64, memory_height=4, memory_max_width=32,
                 memory_feature_index=1, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError(f"Unknown StyleEncoder option(s): {', '.join(sorted(kwargs))}")
        if num_style_tokens != 1:
            raise ValueError('Spatial-memory encoding uses exactly one global latent token')
        if min(style_dim, in_dim, memory_dim, memory_height, memory_max_width) < 1:
            raise ValueError('Style and memory dimensions must be positive')
        if not 0 <= memory_feature_index < len(backbone_channels):
            raise ValueError('memory_feature_index is outside the backbone maps')
        available_scales = tuple(2 ** i for i in range(len(backbone_channels)))
        feature_scales = tuple(feature_scales) if feature_scales is not None else available_scales
        if (not feature_scales or len(set(feature_scales)) != len(feature_scales)
                or any(scale not in available_scales for scale in feature_scales)):
            raise ValueError(f'feature_scales must be distinct members of {available_scales}')
        self.style_dim = int(style_dim)
        self._in_dim = int(in_dim)
        self.num_style_tokens = 1
        self.content_grl = content_grl
        # These maps remain available to the unchanged contextual objective.
        self.feature_indices = tuple(available_scales.index(scale) for scale in feature_scales)
        self.memory_dim = int(memory_dim)
        self.memory_height = int(memory_height)
        self.memory_max_width = int(memory_max_width)
        self.memory_feature_index = int(memory_feature_index)

        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim), nn.LeakyReLU(),
        )
        self.global_context_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.mu = nn.Linear(in_dim, style_dim)
        self.logvar = nn.Linear(in_dim, style_dim)
        self.sequence_model = GatedStyleContext(in_dim)
        self.memory_projection = nn.Conv2d(
            backbone_channels[self.memory_feature_index], self.memory_dim,
            kernel_size=1, bias=False,
        )
        self.memory_norm = nn.LayerNorm(self.memory_dim, elementwise_affine=False)
        self.content_probe = nn.Linear(style_dim, n_class or get_alphabet_size('all'))
        if init != 'none':
            init_weights(self, init)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -10.)

    @staticmethod
    def global_token(style_tokens):
        return style_tokens[:, 0]

    def predict_content(self, style_tokens, reverse=True):
        style_for_probe = style_tokens[:, :1]
        if reverse:
            style_for_probe = _gradient_reverse(style_for_probe, self.content_grl)
        return self.content_probe(style_for_probe)

    def _encode_global_style(self, feat, feat_len):
        feat_mask = (
            torch.arange(feat.size(-1), device=feat.device)[None, :] < feat_len[:, None]
        ) if feat_len is not None else None
        contextual = self.sequence_model(feat, valid_mask=feat_mask)
        global_context = (
            contextual.mean(dim=-1) if feat_len is None
            else contextual.sum(dim=-1) / feat_len.unsqueeze(1).to(feat.dtype)
        )
        global_style = self.linear_style(self.global_context_norm(global_context)).unsqueeze(1)
        return self.mu(global_style), self.logvar(global_style).clamp(-14.0, 4.0)

    def _build_reference_memory(self, feature, lengths):
        """Pool valid regions before padding so batch width cannot alter evidence."""
        if lengths is None:
            valid_widths = [feature.size(-1)] * feature.size(0)
        else:
            valid_widths = lengths.clamp(1, feature.size(-1)).detach().cpu().tolist()
        pooled_widths = [min(width, self.memory_max_width) for width in valid_widths]
        batch_width = max(pooled_widths)
        pooled = torch.cat([
            F.pad(
                F.adaptive_avg_pool2d(
                    row[:, :, :, :valid_width],
                    (self.memory_height, pooled_width),
                ),
                (0, batch_width - pooled_width),
            )
            for row, valid_width, pooled_width in zip(
                feature.split(1), valid_widths, pooled_widths,
            )
        ], dim=0)
        visual = self.memory_norm(self.memory_projection(pooled).flatten(2).transpose(1, 2))
        widths = torch.tensor(pooled_widths, device=feature.device)
        valid = (
            torch.arange(batch_width, device=feature.device)[None, None, :]
            < widths[:, None, None]
        ).expand(-1, self.memory_height, -1).reshape(feature.size(0), -1)
        return visual.masked_fill(~valid.unsqueeze(-1), 0.0), valid

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False,
                vae_mode=False, ret_backbone_feat=False, backbone_features=None,
                global_only=False, return_memory=False):
        if global_only and return_memory:
            raise ValueError('global_only and return_memory are mutually exclusive')
        feat, all_feats = (
            cnn_backbone(img, ret_feats=True, x_lens=img_len)
            if backbone_features is None else backbone_features
        )
        if max((*self.feature_indices, self.memory_feature_index)) >= len(all_feats):
            raise RuntimeError('The backbone did not return the configured feature maps')
        backbone = getattr(cnn_backbone, 'module', cnn_backbone)
        if img_len is not None:
            feat_len, map_lengths = backbone.feature_lengths(img_len)
            feat_len = feat_len.clamp(1, feat.size(-1))
        else:
            feat_len, map_lengths = None, None
        global_mu, global_logvar = self._encode_global_style(feat, feat_len)
        if vae_mode:
            sample = global_mu + torch.randn_like(global_mu) * torch.exp(0.5 * global_logvar)
            style = (sample, global_mu, global_logvar)
        else:
            style = global_mu

        if ret_feats or ret_backbone_feat:
            features = []
            for index in self.feature_indices:
                feature = all_feats[index]
                if map_lengths is not None:
                    valid = torch.arange(feature.size(-1), device=feature.device)[None, :] < map_lengths[index][:, None]
                    feature = feature.masked_fill(~valid[:, None, None, :], 0.0)
                features.append(feature)
            result = (style, features, feat) if ret_backbone_feat else (style, features)
        else:
            result = style
        # Loss-only calls do not allocate spatial memory.
        if return_memory:
            index = self.memory_feature_index
            memory, valid = self._build_reference_memory(
                all_feats[index], map_lengths[index] if map_lengths is not None else None,
            )
            return result, memory, valid
        return result


class MaskedAttentiveStatsPool(nn.Module):
    """Pool writer features while ignoring right-padding.

    The mean carries broad writer style while the standard deviation preserves
    stroke and spacing variation that uniform temporal averaging discards.
    Attention focuses the pool on informative glyph positions.
    """
    def __init__(self, in_dim, attention_dim=128):
        super().__init__()
        attention_dim = max(1, int(attention_dim))
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, attention_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, 1, kernel_size=1),
        )
        self.projection = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.LayerNorm(in_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat, valid_mask):
        if valid_mask.dtype is not torch.bool:
            valid_mask = valid_mask.bool()
        scores = self.attention(feat).squeeze(1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=feat.dtype)
        mean = torch.sum(feat * weights.unsqueeze(1), dim=-1)
        centered = feat - mean.unsqueeze(-1)
        var = torch.sum(centered.square() * weights.unsqueeze(1), dim=-1)
        std = torch.sqrt(var.clamp_min(1e-5))
        return self.projection(torch.cat((mean, std), dim=1))


class WriterIdentifier(nn.Module):
    def __init__(self, n_writer=372, in_dim=256, init='N02',
                 pool='mean', attention_dim=128):
        super(WriterIdentifier, self).__init__()
        self.reduce_len_scale = 32
        self.pool = str(pool).lower()
        if self.pool not in ('mean', 'attentive_stats'):
            raise ValueError("Writer pool must be 'mean' or 'attentive_stats'")

        ######################################
        # Construct WriterIdentifier
        ######################################

        self.linear_wid = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, n_writer),
        )
        self.attentive_pool = (
            MaskedAttentiveStatsPool(in_dim, attention_dim)
            if self.pool == 'attentive_stats' else None
        )

        if init != 'none':
            init_weights(self, init)

    def _pool_features(self, feat, img_len, cnn_backbone):
        feat_len = torch.div(
            img_len, cnn_backbone.reduce_len_scale, rounding_mode='trunc'
        ).clamp(min=1, max=feat.size(-1))
        valid_mask = _len2mask(feat_len, feat.size(-1), dtype=None)
        if self.attentive_pool is not None:
            return self.attentive_pool(feat, valid_mask)
        mask = valid_mask.unsqueeze(1).to(dtype=feat.dtype)
        return (feat * mask).sum(dim=-1) / feat_len.unsqueeze(1).to(feat.dtype)

    def forward(self, img, img_len, cnn_backbone, ret_feats=False):
        feat, all_feats = cnn_backbone(img, ret_feats)
        wid_feat = self._pool_features(feat, img_len, cnn_backbone)
        wid_logits = self.linear_wid(wid_feat)
        if ret_feats:
            return wid_logits, all_feats
        return wid_logits

    def forward_from_feat(self, feat, img_len, cnn_backbone):
        """Classify an already-computed backbone feature map.

        StyleEncoder has just run the frozen backbone on the style-transfer
        image. Reusing that feature avoids a second identical CNN pass while
        retaining gradients from the writer loss back to the generated image.
        """
        wid_feat = self._pool_features(feat, img_len, cnn_backbone)
        return self.linear_wid(wid_feat)

    def return_feat(self, img, img_len, cnn_backbone):
        """Return intermediate writer features (before classification head)."""
        feat, _ = cnn_backbone(img, ret_feats=False)
        wid_feat = self._pool_features(feat, img_len, cnn_backbone)
        # Pass through all linear layers except the last classification one.
        for layer in self.linear_wid[:-1]:
            wid_feat = layer(wid_feat)
        return wid_feat


class Recognizer(nn.Module):
    # resolution: 32  max_dim: 512  in_channel: 1  norm: 'none'  init: 'N02'  dropout: 0.  n_class: 72  rnn_depth: 0
    def __init__(self, n_class, resolution=16, max_dim=256, in_channel=1, norm='none',
                 init='none', rnn_depth=1, dropout=0.0, bidirectional=True, img_height=64, **kwargs):
        super(Recognizer, self).__init__()
        # Match the CNN's horizontal downsampling using the same geometry rule
        # as StyleBackbone; no dataset or benchmark identity is consulted.
        init_stride = _input_stride_for_height(img_height)
        self.len_scale = 8 * init_stride
        self.use_rnn = rnn_depth > 0
        self.bidirectional = bidirectional

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, init_stride, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])

        ######################################
        # Construct Classifier
        ######################################
        cnn_c = [nn.ReLU(),
                 Conv2dBlock(df, df, 3, 1, 0,
                             norm=norm,
                             activation='relu')]

        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.cnn_ctc = nn.Sequential(*cnn_c)
        if self.use_rnn:
            if bidirectional:
                self.rnn_ctc = DeepBLSTM(df, df, rnn_depth, bidirectional=True)
            else:
                self.rnn_ctc = DeepLSTM(df, df, rnn_depth)
        self.ctc_cls = nn.Linear(df, n_class)

        if init != 'none':
            init_weights(self, init)

    def forward(self, x, x_len=None, return_log_probs=None):
        cnn_feat = self.cnn_backbone(x)
        cnn_feat2 = self.cnn_ctc(cnn_feat)
        ctc_feat = cnn_feat2.squeeze(-2).transpose(1, 2)
        if self.use_rnn:
            if x_len is None:
                x_len = torch.full((x.size(0),), x.size(-1), dtype=torch.long, device=x.device)
            if self.bidirectional:
                ctc_len = torch.clamp(torch.div(x_len, self.len_scale, rounding_mode='trunc'), min=1)
            else:
                ctc_len = None
            ctc_feat = self.rnn_ctc(
                ctc_feat,
                ctc_len.cpu() if isinstance(ctc_len, torch.Tensor) else ctc_len,
            )
        logits = self.ctc_cls(ctc_feat)
        if return_log_probs is None:
            return_log_probs = self.training
        if return_log_probs:
            return logits.transpose(0, 1).log_softmax(2)
        return logits

    def frozen_bn(self):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fix_bn)
