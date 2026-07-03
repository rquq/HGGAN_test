import numpy as np
import torch
from torch import nn
import functools
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM, Identity
from networks.mamba import MambaAttention
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
        
        # ADD: Heavy CNN to capture global word geometry (slant, spacing, ratio)
        self.sequence_model = HeavyCNNAttention(in_dim)
        
        # Local-Global Multi-scale feature projection (channel sum = 2 * (64 + 128 + 256) = 896 due to mean + max concatenation)
        self.local_proj = nn.Sequential(
            nn.Conv1d(896, in_dim, kernel_size=1),
            nn.GroupNorm(8, in_dim),
            nn.SiLU()
        )
        
        # Projection layer to fuse concatenated average and max pooled sequential style features
        self.style_proj = nn.Linear(in_dim * 2, in_dim)
        
        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):
        # Always request intermediate features to capture local details
        feat, all_feats = cnn_backbone(img, ret_feats=True)
        
        # 1. Global context from main feature using Heavy CNN
        feat_m = self.sequence_model(feat) # feat is (B, C, W), feat_m is (B, C, W)
        
        # 2. Extract and Fuse multi-scale local details (layer 9, 13, 16)
        local_features = []
        for f in all_feats:
            # f is (B, C_f, H_f, W_f)
            # Retain both general contextual trends (mean pooling) and key structural stroke bounds (max pooling)
            f_mean = torch.mean(f, dim=2) # (B, C_f, W_f)
            f_max = torch.max(f, dim=2)[0] # (B, C_f, W_f)
            
            # Concatenate pooled features to preserve spatial curvature allographs
            f_pooled = torch.cat([f_mean, f_max], dim=1) # (B, C_f * 2, W_f)
            
            # Interpolate width to match main feature width
            f_resized = F.interpolate(f_pooled, size=feat_m.size(-1), mode='linear', align_corners=False)
            local_features.append(f_resized)
            
        local_fused = torch.cat(local_features, dim=1) # (B, 896, W)
        local_mapped = self.local_proj(local_fused) # (B, in_dim, W)
        
        # Stabilized Local-Global Fusion
        feat_final = feat_m + local_mapped
        
        # Output 32 style tokens using both average and max pooling to optimize local/global representation
        style_avg = torch.nn.functional.adaptive_avg_pool1d(feat_final, 32) # (B, C, 32)
        style_max = torch.nn.functional.adaptive_max_pool1d(feat_final, 32) # (B, C, 32)
        
        style_seq = torch.cat([style_avg, style_max], dim=1) # (B, C * 2, 32)
        style_seq = style_seq.transpose(1, 2) # (B, 32, C * 2)
        style_seq = self.style_proj(style_seq) # (B, 32, C)
        
        style = self.linear_style(style_seq)
        style_tokens_mu = self.mu(style) # (B, 32, style_dim)

        if vae_mode:
            logvar = self.logvar(style)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            style_tokens_sampled = eps * std + style_tokens_mu
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

    def return_feat(self, img, img_len):
        feat = self.cnn_backbone(img)
        img_len = img_len // self.reduce_len_scale
        out_w = self.cnn_wid(feat).squeeze(-2)
        img_len_mask = _len2mask(img_len, out_w.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (out_w * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        for j in range(2):
            wid_feat = self.linear_wid[j](wid_feat)
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
