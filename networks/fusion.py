import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import MambaBlock, RMSNorm

class StyleContentMamba(nn.Module):
    """
    Optimized 1D Prefix-Context Mamba Fusion for Single-Stream Generator.
    Treats the style vector as a prompt/prefix token for the content sequence.
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
        
        # 3. Normalization and Stability
        self.norm = RMSNorm(d_model)
        
        # 4. Refined Spatial Modulation
        self.style_mod = nn.Sequential(
            nn.Linear(style_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )

    def forward(self, content_seq, style_vec):
        """
        Args:
            content_seq: (B, L, D) sequence of content embeddings
            style_vec: (B, style_dim) single style vector
        """
        B, L, D = content_seq.shape
        
        # --- STAGE 1: Sequence Preparation ---
        s_feat = self.style_proj(style_vec).unsqueeze(1) # (B, 1, D)
        c_feat = self.content_proj(content_seq) # (B, L, D)
        
        # Dual-Prompting: Style acts as prefix (forward scan) AND suffix (backward scan)
        combined = torch.cat([s_feat, c_feat, s_feat], dim=1) # (B, 1+L+1, D)
        
        # --- STAGE 2: 1D Dual-Context Fusion ---
        fused = self.mamba(combined)
        
        # --- STAGE 3: Extract Refined Content ---
        content_refined = fused[:, 1:-1, :] # Discard the prefix and suffix
        content_final = self.norm(content_refined + content_seq)
        
        # --- STAGE 4: Style Modulation ---
        mod_params = self.style_mod(style_vec).unsqueeze(1) # (B, 1, D*2)
        scale, shift = mod_params.chunk(2, dim=-1)
        
        return content_final * (1 + scale) + shift

class MixMamba(nn.Module):
    def __init__(self, d_model, style_dim):
        super().__init__()
        self.fusion = StyleContentMamba(d_model, style_dim)
        
    def forward(self, content_seq, style_vec):
        return self.fusion(content_seq, style_vec)
