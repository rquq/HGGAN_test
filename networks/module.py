import numpy as np
import torch
from torch import nn
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepLSTM
from networks.utils import _len2mask, init_weights
import torch.nn.functional as F

def _input_stride_for_height(img_height, base_height=32):
    """Scale encoder stride from image geometry, independent of dataset name."""
    height = int(img_height)
    base_height = int(base_height)
    if height < 1 or base_height < 1:
        raise ValueError('image and base heights must be positive')
    return max(1, height // base_height)



def _build_ocr_style_cnn(resolution, max_dim, in_channel, norm, dropout, init_stride):
    """Build the shared CNN feature extractor and CTC head for StyleBackbone and Recognizer."""
    nf = resolution
    cnn_f = [
        nn.ConstantPad2d(2, -1),
        Conv2dBlock(in_channel, nf, 5, init_stride, 0, norm='none', activation='none'),
    ]
    for _ in range(2):
        nf_out = min(int(nf * 2), max_dim)
        cnn_f += [
            ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2),
            nn.ZeroPad2d((1, 1, 0, 0)),
            ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2),
            nn.ZeroPad2d(1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        nf = min(nf_out, max_dim)

    df = nf
    for i in range(2):
        df_out = min(int(df * 2), max_dim)
        cnn_f += [
            ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout),
            ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout),
            nn.MaxPool2d(kernel_size=3, stride=2) if i < 1 else nn.ZeroPad2d((1, 1, 0, 0)),
        ]
        df = min(df_out, max_dim)

    cnn_c = [
        nn.ReLU(),
        Conv2dBlock(df, df, 3, 1, 0, norm=norm, activation='relu'),
    ]
    return nn.Sequential(*cnn_f), nn.Sequential(*cnn_c), df



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



class StyleBackbone(nn.Module):
    def __init__(self, resolution=16, max_dim=256, in_channel=1, init='N02', dropout=0.0, norm='bn', img_height=64, **kwargs):
        super(StyleBackbone, self).__init__()
        # Derive the front-end stride from input geometry and keep length
        # metadata in lock-step with the CNN at every supported resolution.
        init_stride = _input_stride_for_height(img_height)
        self.reduce_len_scale = 8 * init_stride
        self.cnn_backbone, self.cnn_ctc, _ = _build_ocr_style_cnn(
            resolution, max_dim, in_channel, norm, dropout, init_stride
        )
        self.layer_name_mapping = {
            '9': "feat2",
            '13': "feat3",
            '16': "feat4",
        }
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



class StyleEncoder(nn.Module):
    def __init__(self, style_dim=32, in_dim=256, init='N02'):
        super(StyleEncoder, self).__init__()
        self.style_dim = style_dim

        ######################################
        # Construct StyleEncoder
        ######################################
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(in_dim, style_dim)
        self.logvar = nn.Linear(in_dim, style_dim)
        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        style = self.linear_style(style)
        mu = self.mu(style)

        if vae_mode:
            logvar = self.logvar(style)
            style = self.reparameterize(mu, logvar)
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu



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

        self.cnn_backbone, self.cnn_ctc, df = _build_ocr_style_cnn(
            resolution, max_dim, in_channel, norm, dropout, init_stride
        )
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
