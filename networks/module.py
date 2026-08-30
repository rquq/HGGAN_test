import numpy as np
import torch
from torch import nn
import functools
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM, Identity
from networks.utils import _len2mask, init_weights
import torch.nn.functional as F


class HeavyCNNAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # 1. Global Multi-scale dilated convolutions for global context (slant, spacing, aspect ratio)
        self.conv1 = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv_dilated1 = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=2, dilation=2)
        self.conv_dilated2 = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=4, dilation=4)
        self.conv_dilated3 = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=8, dilation=8)

        # 2. Local detail branch (depthwise and small convolutions to capture fine-grained glyph strokes and curves)
        self.local_conv1 = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1)
        self.local_conv2 = nn.Conv1d(in_dim, in_dim, kernel_size=5, padding=2, groups=in_dim)
        self.local_fuse = nn.Conv1d(in_dim * 2, in_dim, kernel_size=1)

        # 3. Global bottleneck fusion
        self.fuse = nn.Sequential(
            nn.Conv1d(in_dim * 4, in_dim, kernel_size=1),
            nn.GroupNorm(8, in_dim),
            nn.SiLU(),
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1)
        )

        # 4. Gating layers to dynamically fuse local and global features based on allographic complexity
        self.gate_global = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate_local = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 5. Channel Squeeze-and-Excitation for focused style extraction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_dim, in_dim // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_dim // 4, in_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        # Global context mapping
        x1 = F.silu(self.conv1(x))
        x2 = F.silu(self.conv_dilated1(x))
        x3 = F.silu(self.conv_dilated2(x))
        x4 = F.silu(self.conv_dilated3(x))

        fused_global = torch.cat([x1, x2, x3, x4], dim=1)
        out_global = self.fuse(fused_global)

        # Local context mapping
        l1 = F.silu(self.local_conv1(x))
        l2 = F.silu(self.local_conv2(x))
        out_local = self.local_fuse(torch.cat([l1, l2], dim=1))

        # Dynamic Gated Fusion of Global and Local contexts
        g_g = self.gate_global(out_global)
        g_l = self.gate_local(out_local)
        out_fused = out_global * g_g + out_local * g_l

        # Squeeze-and-Excitation gating
        scale = self.se(out_fused)
        out = out_fused * scale

        return x + self.gamma * out


class StyleBackbone(nn.Module):
    def __init__(self, resolution=16, max_dim=256, in_channel=1, init='N02', dropout=0.0, norm='bn', img_height=64, **kwargs):
        super(StyleBackbone, self).__init__()
        self.reduce_len_scale = 16
        nf = resolution
        init_stride = 1 if int(img_height) <= 32 else 2
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

    def forward(self, x, ret_feats=False):
        feats = []
        for name, layer in self.cnn_backbone._modules.items():
            x = layer(x)
            if ret_feats and name in self.layer_name_mapping:
                feats.append(x)

        out = self.cnn_ctc(x).squeeze(-2)

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
    def __init__(self, style_dim=32, in_dim=256, init='N02', num_style_tokens=8,
                 backbone_channels=(64, 128, 256), n_class=80, content_grl=1.0,
                 local_query_residual=0.5,
                 local_attention_residual_init=0.25,
                 local_query_anchor_strength=0.5, **kwargs):
        super(StyleEncoder, self).__init__()
        self.style_dim = style_dim
        self._in_dim = in_dim
        self.num_style_tokens = num_style_tokens
        self.content_grl = content_grl
        self.local_query_residual = float(local_query_residual)
        self.local_attention_residual_init = float(local_attention_residual_init)
        self.local_query_anchor_strength = float(
            local_query_anchor_strength
        )
        if num_style_tokens < 1:
            raise ValueError('num_style_tokens must be at least 1')
        if self.local_query_residual < 0:
            raise ValueError('local_query_residual must be non-negative')
        if not 0.0 < self.local_attention_residual_init < 1.0:
            raise ValueError(
                'local_attention_residual_init must be strictly between 0 and 1'
            )
        if not 0.0 <= self.local_query_anchor_strength <= 1.0:
            raise ValueError(
                'local_query_anchor_strength must be in [0, 1]'
            )

        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )
        self.mu = nn.Linear(in_dim, style_dim)
        self.logvar = nn.Linear(in_dim, style_dim)
        self.sequence_model = HeavyCNNAttention(in_dim)

        # Build every trainable projection before the optimizer is created. The old
        # forward-time replacement silently left new parameters unoptimised.
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(channels, in_dim, kernel_size=1)
            for channels in backbone_channels
        ])
        for layer in self.proj_layers:
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Token zero is an explicit global style summary. The remaining compact
        # query set captures local stroke details without a 32x32 content-rich code.
        query_count = num_style_tokens - 1
        style_query_init = torch.empty(1, query_count, in_dim)
        if query_count:
            # Orthogonal rows start as distinct local stroke slots while matching
            # the old N(0, 0.02) per-component scale in expectation.
            nn.init.orthogonal_(style_query_init[0])
            style_query_init.mul_(0.02 * (in_dim ** 0.5))
        self.style_queries = nn.Parameter(style_query_init)
        # Keep spatial-attention queries separated throughout long training.
        # The trainable component still adapts, while the fixed copy prevents
        # the partial query collapse measured in the epoch-50 checkpoint.
        self.register_buffer(
            'style_query_anchors', style_query_init.detach().clone()
        )

        # A fixed orthogonal code gives every local slot a permanent identity.
        # Writer evidence is still learned; the code only prevents all slots from
        # converging to the same direction after attention and projection.
        slot_anchors = torch.empty(1, query_count, style_dim)
        if query_count:
            nn.init.orthogonal_(slot_anchors[0])
            slot_anchors.mul_(style_dim ** 0.5)
        self.register_buffer('local_slot_anchors', slot_anchors)

        attention_gate_logit = torch.logit(torch.tensor(
            self.local_attention_residual_init
        )).item()
        self.local_attention_gate_logits = nn.Parameter(
            torch.full((in_dim,), attention_gate_logit)
        )
        self.style_cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=4, batch_first=True
        )
        # Affine-free norms add no checkpoint state. They prevent the frozen
        # backbone's very different feature scales from dominating attention.
        self.style_key_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.style_query_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.local_output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.content_probe = nn.Linear(style_dim, n_class)

        if init != 'none':
            init_weights(self, init)
        nn.init.constant_(self.logvar.weight, 0.)
        nn.init.constant_(self.logvar.bias, -10.)

    @staticmethod
    def _width_mask(img_len, source_width, target_width):
        if img_len is None:
            return None, None
        scaled_len = torch.ceil(
            img_len.to(dtype=torch.float32) * (float(target_width) / float(source_width))
        ).long().clamp_(min=1, max=target_width)
        positions = torch.arange(target_width, device=img_len.device).unsqueeze(0)
        return positions < scaled_len.unsqueeze(1), scaled_len

    @staticmethod
    def global_token(style_tokens):
        return style_tokens[:, 0]

    def predict_content(self, style_tokens, reverse=True):
        style_for_probe = style_tokens
        if reverse:
            style_for_probe = _gradient_reverse(style_for_probe, self.content_grl)
        return self.content_probe(style_for_probe)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):
        feat, all_feats = cnn_backbone(img, ret_feats=True)
        if len(self.proj_layers) != len(all_feats):
            raise RuntimeError(
                f'StyleEncoder expected {len(self.proj_layers)} backbone feature maps, '
                f'but received {len(all_feats)}. Set EncModel.backbone_channels explicitly.'
            )
        for index, (proj_layer, feature) in enumerate(zip(self.proj_layers, all_feats)):
            if proj_layer.in_channels != feature.size(1):
                raise RuntimeError(
                    f'Backbone feature {index} has {feature.size(1)} channels, but the '
                    f'configured projection expects {proj_layer.in_channels}.'
                )

        feat_mask, feat_len = self._width_mask(img_len, img.size(-1), feat.size(-1))
        feat_mask_f = feat_mask.unsqueeze(1).to(feat.dtype) if feat_mask is not None else 1.0
        feat_m = self.sequence_model(feat * feat_mask_f) * feat_mask_f
        if feat_mask is None:
            global_context = feat_m.mean(dim=-1)
        else:
            global_context = feat_m.sum(dim=-1) / feat_len.unsqueeze(1).to(feat.dtype)

        feat_m_trans = feat_m.transpose(1, 2)
        pe_1d = get_1d_sinusoidal_embeddings(
            feat_m_trans.size(1), self._in_dim, feat_m_trans.device
        )
        feat_m_trans = feat_m_trans + pe_1d.unsqueeze(0)
        if feat_mask is not None:
            feat_m_trans = feat_m_trans * feat_mask.unsqueeze(-1).to(feat_m_trans.dtype)

        spatial_tokens = [feat_m_trans]
        padding_masks = [~feat_mask] if feat_mask is not None else []
        masked_all_feats = []
        for proj_layer, feature in zip(self.proj_layers, all_feats):
            width_mask, _ = self._width_mask(img_len, img.size(-1), feature.size(-1))
            width_mask_f = (
                width_mask[:, None, None, :].to(feature.dtype)
                if width_mask is not None else 1.0
            )
            feature_masked = feature * width_mask_f
            masked_all_feats.append(feature_masked)
            feature_projected = proj_layer(feature_masked)
            feature_pooled = F.adaptive_avg_pool2d(
                feature_projected, (4, feature.size(-1))
            )
            height, width = feature_pooled.shape[-2:]
            pe_2d = get_2d_sinusoidal_embeddings(
                height, width, self._in_dim, feature_pooled.device
            )
            feature_pooled = feature_pooled + pe_2d.permute(2, 0, 1).unsqueeze(0)
            if width_mask is not None:
                feature_pooled = feature_pooled * width_mask[:, None, None, :].to(
                    feature_pooled.dtype
                )
            spatial_tokens.append(feature_pooled.flatten(2).transpose(1, 2))
            if width_mask is not None:
                padding_masks.append(
                    ~width_mask[:, None, :].expand(-1, height, -1).reshape(feature.size(0), -1)
                )

        style_keys = self.style_key_norm(torch.cat(spatial_tokens, dim=1))
        key_padding_mask = torch.cat(padding_masks, dim=1) if padding_masks else None

        batch_size = img.size(0)
        style_queries = (
            self.style_queries
            + self.local_query_anchor_strength * self.style_query_anchors
        ).expand(batch_size, -1, -1)
        if style_queries.size(1):
            pe_queries = get_1d_sinusoidal_embeddings(
                style_queries.size(1), self._in_dim, style_queries.device
            )
            style_queries = self.style_query_norm(
                style_queries + pe_queries.unsqueeze(0)
            )
            local_attended, _ = self.style_cross_attn(
                query=style_queries,
                key=style_keys,
                value=style_keys,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            attention_strength = torch.sigmoid(
                self.local_attention_gate_logits
            ).view(1, 1, -1)
            local_style = self.linear_style(self.local_output_norm(
                style_queries + attention_strength * local_attended
            ))
        else:
            local_style = style_queries

        global_style = self.linear_style(global_context).unsqueeze(1)
        style = torch.cat([global_style, local_style], dim=1)
        global_mu = self.mu(global_style)
        if local_style.size(1):
            local_data_mu = self.mu(local_style)
            # Match the fixed code to each token's learned RMS. This makes the
            # anti-collapse residual scale-aware without backpropagating through
            # the scale estimate or overpowering writer-specific evidence.
            local_data_rms = local_data_mu.detach().square().mean(
                dim=-1, keepdim=True
            ).sqrt().clamp_min(0.05)
            local_identity = self.local_slot_anchors.expand(
                batch_size, -1, -1
            ).to(dtype=local_data_mu.dtype)
            local_mu = (
                local_data_mu
                + self.local_query_residual * local_data_rms * local_identity
            )
        else:
            local_mu = self.mu(local_style)
        style_tokens_mu = torch.cat([global_mu, local_mu], dim=1)

        if vae_mode:
            logvar = torch.clamp(self.logvar(style), min=-14.0, max=4.0)
            std = torch.exp(0.5 * logvar)
            style_tokens_sampled = torch.randn_like(std) * std + style_tokens_mu
            style_tokens = (style_tokens_sampled, style_tokens_mu, logvar)
        else:
            style_tokens = style_tokens_mu

        if ret_feats:
            return style_tokens, masked_all_feats
        return style_tokens


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
        self.len_scale = 16
        self.use_rnn = rnn_depth > 0
        self.bidirectional = bidirectional

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        init_stride = 1 if int(img_height) <= 32 else 2
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
