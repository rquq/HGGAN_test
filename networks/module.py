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
    def __init__(self, resolution=16, max_dim=256, in_channel=1, init='N02', dropout=0.0, norm='bn'):
        super(StyleBackbone, self).__init__()
        self.reduce_len_scale = 16
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 2, 0,
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
    import numpy as np
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
    import numpy as np
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


class StyleEncoder(nn.Module):
    def __init__(self, style_dim=32, in_dim=256, init='N02', **kwargs):
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
        
        # ADD: Heavy CNN to capture global word geometry (slant, spacing, ratio)
        self.sequence_model = HeavyCNNAttention(in_dim)
        
        # Projection layers for multi-scale CNN backbone features — pre-built with default
        # channels [64, 128, 256] matching StyleBackbone, dynamically adaptable if needed.
        self._in_dim = in_dim
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(64, in_dim, kernel_size=1),
            nn.Conv2d(128, in_dim, kernel_size=1),
            nn.Conv2d(256, in_dim, kernel_size=1),
        ])
        for layer in self.proj_layers:
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        # Learned style query tokens (style_dim tokens, each of size in_dim)
        self.num_style_tokens = style_dim  # one query per style-dim slot
        self.style_queries = nn.Parameter(torch.randn(1, self.num_style_tokens, in_dim) * 0.02)
        self.style_cross_attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=4, batch_first=True)
        
        if init != 'none':
            init_weights(self, init)

        # Initialize log-variance weights to 0.0 and bias to -10.0 to start training almost deterministically
        nn.init.constant_(self.logvar.weight, 0.)
        nn.init.constant_(self.logvar.bias, -10.)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        proj_keys = [k for k in state_dict.keys() if k.startswith(prefix + 'proj_layers.')]
        if len(proj_keys) > 0:
            prefix_len = len(prefix)
            layer_indices = sorted(list(set(
                int(k[prefix_len:].split('.')[1])
                for k in proj_keys
                if k[prefix_len:].startswith('proj_layers.') and k[prefix_len:].split('.')[1].isdigit()
            )))
            if self.proj_layers is None or len(self.proj_layers) != len(layer_indices):
                layers = []
                for idx in layer_indices:
                    w_key = f"{prefix}proj_layers.{idx}.weight"
                    if w_key in state_dict:
                        in_ch = state_dict[w_key].shape[1]
                        layers.append(nn.Conv2d(in_ch, self._in_dim, kernel_size=1))
                if len(layers) > 0:
                    self.proj_layers = nn.ModuleList(layers)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _build_proj_layers(self, all_feats):
        """Build projection Conv2d layers matching the actual backbone feature channels."""
        if self.proj_layers is not None and len(self.proj_layers) == len(all_feats):
            channels_match = all(
                layer.weight.shape[1] == f.size(1) 
                for layer, f in zip(self.proj_layers, all_feats)
            )
            if channels_match:
                self.proj_layers = self.proj_layers.to(all_feats[0].device)
                return

        layers = []
        for f in all_feats:
            layers.append(nn.Conv2d(f.size(1), self._in_dim, kernel_size=1))
        self.proj_layers = nn.ModuleList(layers).to(all_feats[0].device)
        # Apply same init as rest of the module
        for layer in self.proj_layers:
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):
        # Always request intermediate features to capture local details
        feat, all_feats = cnn_backbone(img, ret_feats=True)
        
        # Build/adapt projection layers if needed
        if self.proj_layers is None or len(self.proj_layers) != len(all_feats):
            self._build_proj_layers(all_feats)
        else:
            self.proj_layers = self.proj_layers.to(all_feats[0].device)
        
        # 1. Global context from main feature using Heavy CNN
        feat_m = self.sequence_model(feat) # feat is (B, C, W), feat_m is (B, C, W)
        
        # Transpose and add 1D sinusoidal positional embedding
        feat_m_trans = feat_m.transpose(1, 2) # (B, W, in_dim)
        pe_1d = get_1d_sinusoidal_embeddings(feat_m_trans.size(1), self._in_dim, feat_m_trans.device)
        feat_m_trans = feat_m_trans + pe_1d.unsqueeze(0)
        
        # 2. Project and pool each intermediate feature map dynamically
        spatial_tokens = [feat_m_trans] # Start with global sequence (B, W, in_dim)
        for proj_layer, f in zip(self.proj_layers, all_feats):
            f_proj = proj_layer(f) # (B, in_dim, H_f, W_f)
            # Pool height to 4 to condense vertical dimension while retaining full horizontal resolution
            f_pooled = F.adaptive_avg_pool2d(f_proj, (4, f.size(-1))) # (B, in_dim, 4, W_f)
            # Add 2D sinusoidal positional embedding
            H_p, W_p = f_pooled.size(2), f_pooled.size(3)
            pe_2d = get_2d_sinusoidal_embeddings(H_p, W_p, self._in_dim, f_pooled.device)
            f_pooled = f_pooled + pe_2d.permute(2, 0, 1).unsqueeze(0)
            
            f_flat = f_pooled.flatten(2).transpose(1, 2) # (B, 4 * W_f, in_dim)
            spatial_tokens.append(f_flat)
            
        # Concatenate all spatial/global tokens
        style_keys = torch.cat(spatial_tokens, dim=1) # (B, total_tokens, in_dim)
        
        # 3. Query style sequence dynamically using learned style query tokens
        B = img.size(0)
        style_queries = self.style_queries.expand(B, -1, -1) # (B, num_style_tokens, in_dim)
        # Add 1D sinusoidal positional embedding to queries to distinguish query slots chronologically
        pe_queries = get_1d_sinusoidal_embeddings(self.num_style_tokens, self._in_dim, style_queries.device)
        style_queries = style_queries + pe_queries.unsqueeze(0)
        
        # Cross-attention: queries look at the spatial style keys
        style_seq, _ = self.style_cross_attn(query=style_queries, key=style_keys, value=style_keys)
        
        style = self.linear_style(style_seq)
        style_tokens_mu = self.mu(style) # (B, 32, style_dim)

        if vae_mode:
            logvar = self.logvar(style)
            # Clamp logvar to prevent exponential exploding values and training instability
            logvar = torch.clamp(logvar, min=-14.0, max=4.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            style_tokens_sampled = eps * std + style_tokens_mu


        if vae_mode:
            style_tokens = (style_tokens_sampled, style_tokens_mu, logvar)
        else:
            style_tokens = style_tokens_mu

        if ret_feats:
            return style_tokens, all_feats
        else:
            return style_tokens


class WriterIdentifier(nn.Module):
    def __init__(self, n_writer=372, in_dim=256, init='N02'):
        super(WriterIdentifier, self).__init__()
        self.reduce_len_scale = 32

        ######################################
        # Construct WriterIdentifier
        ######################################

        self.linear_wid = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, n_writer),
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone, ret_feats=False):
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = torch.div(img_len, cnn_backbone.reduce_len_scale, rounding_mode='trunc')
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        wid_logits = self.linear_wid(wid_feat)
        if ret_feats:
            return wid_logits, all_feats
        else:
            return wid_logits

    def return_feat(self, img, img_len, cnn_backbone):
        """Return intermediate writer features (before classification head)."""
        feat, _ = cnn_backbone(img, ret_feats=False)
        img_len = torch.div(img_len, cnn_backbone.reduce_len_scale, rounding_mode='trunc')
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        # Pass through all linear layers except the last classification one
        for layer in self.linear_wid[:-1]:
            wid_feat = layer(wid_feat)
        return wid_feat


class Recognizer(nn.Module):
    # resolution: 32  max_dim: 512  in_channel: 1  norm: 'none'  init: 'N02'  dropout: 0.  n_class: 72  rnn_depth: 0
    def __init__(self, n_class, resolution=16, max_dim=256, in_channel=1, norm='none',
                 init='none', rnn_depth=1, dropout=0.0, bidirectional=True):
        super(Recognizer, self).__init__()
        self.len_scale = 16
        self.use_rnn = rnn_depth > 0
        self.bidirectional = bidirectional

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 2, 0,
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

    def forward(self, x, x_len=None):
        cnn_feat = self.cnn_backbone(x)
        cnn_feat2 = self.cnn_ctc(cnn_feat)
        ctc_feat = cnn_feat2.squeeze(-2).transpose(1, 2)
        if self.use_rnn:
            if self.bidirectional:
                ctc_len = torch.div(x_len, self.len_scale, rounding_mode='trunc')
            else:
                ctc_len = None
            ctc_feat = self.rnn_ctc(ctc_feat, ctc_len.cpu())
        logits = self.ctc_cls(ctc_feat)
        if self.training:
            logits = logits.transpose(0, 1).log_softmax(2)
            logits.requires_grad_(True)
        return logits

    def frozen_bn(self):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fix_bn)
