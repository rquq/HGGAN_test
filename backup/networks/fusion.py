import torch
import torch.nn as nn
from networks.mamba import MambaBlock, RMSNorm

class StyleContentMamba(nn.Module):
    """
    Mamba-based fusion block that mixes style and content.
    Style is prioritized by being used as a prefix context for the Mamba sequence processing.
    """
    def __init__(self, d_model, style_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        # Project style to match content dimension
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, d_model),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d_model, d_model)
        )
        
        # Mamba block for bidirectional sequence processing
        self.mamba = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, bidirectional=True)
        self.norm = RMSNorm(d_model)
        
        # Optional: Style-based scaling for the output (AdaIN-like)
        self.style_scale = nn.Linear(style_dim, d_model)
        self.style_shift = nn.Linear(style_dim, d_model)

    def forward(self, content_seq, style_vec):
        """
        Args:
            content_seq: (B, L, D) sequence of content embeddings
            style_vec: (B, S_D) style vector
        Returns:
            fused_seq: (B, L, D) fused sequence
        """
        B, L, D = content_seq.shape
        
        # 1. Project style and treat as the first "token" in the sequence
        style_token = self.style_proj(style_vec).unsqueeze(1) # (B, 1, D)
        
        # 2. Concatenate: Style at the beginning serves as a prior for the content
        combined = torch.cat([style_token, content_seq], dim=1) # (B, L+1, D)
        
        # 3. Process with Mamba
        fused = self.mamba(combined) # (B, L+1, D)
        
        # 4. Extract content tokens
        content_fused = fused[:, 1:]
        
        # 5. Apply style-conditioned modulation (AdaIN-like)
        scale = self.style_scale(style_vec).unsqueeze(1)
        shift = self.style_shift(style_vec).unsqueeze(1)
        
        out = self.norm(content_fused) * (1 + scale) + shift
        
        return out

class MixMamba(nn.Module):
    """
    Simplified mixture model using a single robust Mamba fusion stage.
    """
    def __init__(self, d_model, style_dim):
        super().__init__()
        self.fusion = StyleContentMamba(d_model, style_dim)
        
    def forward(self, content_seq, style_vec):
        return self.fusion(content_seq, style_vec)
