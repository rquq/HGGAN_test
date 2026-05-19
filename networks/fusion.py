import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import MambaBlock, RMSNorm

class StyleContentCrossAttention(nn.Module):
    """
    Direct Query-Key-Value Cross-Attention using PyTorch Scaled Dot Product Attention (SDPA).
    Allows content character tokens to directly query and align with style sequence features.
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, content_seq, style_seq):
        """
        Args:
            content_seq: (B, L, D) sequence of content embeddings
            style_seq: (B, S_len, D) sequence of style embeddings
        """
        B, L, D = content_seq.shape
        S = style_seq.shape[1]
        
        # Project and reshape for Multi-Head: (B, nh, SeqLen, head_dim)
        q = self.q_proj(content_seq).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(style_seq).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(style_seq).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        # Native SDPA check for PyTorch 2.0+ (FlashAttention/Memory-Efficient Attention under the hood)
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q * scale, k.transpose(-2, -1)) # (B, nh, L, S)
        attn_weights = torch.softmax(scores, dim=-1)
        if self.training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        attn_out = torch.matmul(attn_weights, v) # (B, nh, L, head_dim)
        
        # Reshape back to (B, L, D) and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)
        
        x = self.norm1(content_seq + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

class StyleContentMamba(nn.Module):
    """
    Optimized 1D Prefix-Context Mamba Fusion with Dynamic Cross-Attention.
    Treats the style vector as a prompt/prefix token for the content sequence,
    and then applies cross-attention for high-fidelity allograph alignment.
    """
    def __init__(self, d_model, style_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        # 1. Feature Projections
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.content_proj = nn.Linear(d_model, d_model)
        
        # 2. Optimized 1D Sequence Engine
        self.mamba = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, bidirectional=True)
        
        # 3. Dynamic Cross-Attention for Allograph Learning
        self.cross_attn = StyleContentCrossAttention(d_model, nhead=4)
        
        # 4. Normalization and Stability
        self.norm = RMSNorm(d_model)
        
        # 5. Refined Spatial Modulation
        self.style_mod = nn.Sequential(
            nn.Linear(style_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )

    def forward(self, content_seq, style_seq):
        """
        Args:
            content_seq: (B, L, D) sequence of content embeddings
            style_seq: (B, 32, style_dim) sequence of style tokens
        """
        B, L, D = content_seq.shape
        S_len = style_seq.shape[1]
        
        # --- STAGE 1: Sequence Preparation ---
        s_feat = self.style_proj(style_seq) # (B, S_len, D)
        c_feat = self.content_proj(content_seq) # (B, L, D)
        
        # Dual-Prompting: Style acts as prefix (forward scan) AND suffix (backward scan)
        combined = torch.cat([s_feat, c_feat, s_feat], dim=1) # (B, S_len+L+S_len, D)
        
        # --- STAGE 2: 1D Dual-Context Fusion ---
        fused = self.mamba(combined)
        
        # --- STAGE 3: Extract Refined Content ---
        content_refined = fused[:, S_len:-S_len, :] # Discard the prefix and suffix
        content_fused = self.norm(content_refined + content_seq)
        
        # --- STAGE 4: Allograph Refinement via Dynamic Cross-Attention ---
        content_final = self.cross_attn(content_fused, s_feat)
        
        # --- STAGE 5: Style Modulation ---
        style_vec = style_seq.sum(dim=1) / style_seq.size(1) # (B, style_dim)
        mod_params = self.style_mod(style_vec).unsqueeze(1) # (B, 1, D*2)
        scale, shift = mod_params.chunk(2, dim=-1)
        
        return content_final * (1 + scale) + shift

class MixMamba(nn.Module):
    def __init__(self, d_model, style_dim):
        super().__init__()
        self.fusion = StyleContentMamba(d_model, style_dim)
        
    def forward(self, content_seq, style_seq):
        return self.fusion(content_seq, style_seq)
