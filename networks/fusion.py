import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import Mamba2DBlock, RMSNorm

class StyleContentMamba(nn.Module):
    """
    Optimized 2D-Interaction Mamba Fusion for Single-Stream Generator.
    Uses JIT-accelerated scans and a 2D interaction grid between Style and Content.
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
        
        # 2. Optimized 2D Interaction Engine (Single-Stage)
        self.mamba_2d = Mamba2DBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        
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
        
        # --- STAGE 1: Create 2D Interaction Grid ---
        # Even with one style vector, we treat it as a sequence of length 1.
        # Grid shape: (B, 1, L, D)
        s_feat = self.style_proj(style_vec).unsqueeze(1).unsqueeze(2) # (B, 1, 1, D)
        c_feat = self.content_proj(content_seq).unsqueeze(1) # (B, 1, L, D)
        
        grid = s_feat + c_feat # (B, 1, L, D)
        
        # --- STAGE 2: 2D Cross-Scan Fusion ---
        grid_flat = grid.view(B, 1 * L, D)
        fused_grid = self.mamba_2d(grid_flat, 1, L)
        
        # --- STAGE 3: Extract Refined Content ---
        content_refined = fused_grid.view(B, 1, L, D).mean(dim=1)
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
