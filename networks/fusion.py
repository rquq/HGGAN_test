import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ensure_dim3


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for tensor stability."""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class StyleContentCrossAttention(nn.Module):
    """
    Unified Allograph Cross-Attention & Character-Conditioned Modulation.
    Performs single-pass Multi-Head Cross Attention between content characters and style tokens,
    guided by a character-conditioned learned routing prior for explicit allograph selection.
    Outputs both the cross-attended features and character-level AdaIN scale/shift parameters.
    """
    def __init__(self, d_model, nhead=4, dropout=0.1, routing_dim=16, vocab_size=256):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.vocab_size = vocab_size
        self.warned_out_of_vocab = False
        
        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Character-to-style token routing layers (Allograph selection prior)
        self.char_routing_emb = nn.Embedding(vocab_size, routing_dim)
        self.context_routing_proj = nn.Linear(d_model, routing_dim)
        self.style_routing_proj = nn.Linear(d_model, routing_dim)
        
        # Character-level AdaIN scale and shift modulation projection
        self.mod_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, content_seq, style_seq, char_ids=None, mask=None):
        """
        Args:
            content_seq: (B, L, D) sequence of content embeddings
            style_seq: (B, S, D) sequence of style tokens
            char_ids: (B, L) raw character indices for allograph routing
            mask: (B, L) sequence mask for content tokens
        """
        style_seq = ensure_dim3(style_seq)
        B, L, D = content_seq.shape
        S = style_seq.shape[1]
        
        # 1. Multi-Head Q-K-V Projections
        q = self.q_proj(content_seq).view(B, L, self.nhead, self.head_dim).transpose(1, 2) # (B, nh, L, head_dim)
        k = self.k_proj(style_seq).view(B, S, self.nhead, self.head_dim).transpose(1, 2)   # (B, nh, S, head_dim)
        v = self.v_proj(style_seq).view(B, S, self.nhead, self.head_dim).transpose(1, 2)   # (B, nh, S, head_dim)
        
        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q * scale, k.transpose(-2, -1)) # (B, nh, L, S)
        
        # 2. Add Character-Conditioned Allograph Routing Prior
        if char_ids is not None:
            if not self.warned_out_of_vocab and (char_ids >= self.vocab_size).any():
                self.warned_out_of_vocab = True
                print(f"[Warning] Character IDs exceed vocab_size ({self.vocab_size}). Clamping to [0, {self.vocab_size - 1}].")
            char_ids_clipped = torch.clamp(char_ids, 0, self.vocab_size - 1)
            
            char_q_static = self.char_routing_emb(char_ids_clipped)
            char_q_context = self.context_routing_proj(content_seq)
            char_q = char_q_static + char_q_context # (B, L, routing_dim)
            
            style_routing = self.style_routing_proj(style_seq) # (B, S, routing_dim)
            routing_prior = torch.matmul(char_q, style_routing.transpose(-2, -1)) # (B, L, S)
            scores = scores + routing_prior.unsqueeze(1)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Attended style sequence: (B, nh, L, head_dim) -> (B, L, D)
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D)
        
        # 3. Dynamic Allographic AdaIN Modulation (Per-character scale & shift)
        mod_params = self.mod_proj(attn_out)
        scale_char, shift_char = mod_params.chunk(2, dim=-1)
        
        # Combine residual attention with character modulation
        attn_projected = self.out_proj(attn_out)
        fused = content_seq * (1 + scale_char) + shift_char + attn_projected
        
        # 4. FFN & Layer Normalization
        x = self.norm1(content_seq + fused)
        out = self.norm2(x + self.ffn(x))
        
        if mask is not None:
            out = out * mask.to(out.dtype).unsqueeze(-1)
            
        return out


class StyleContentFusion(nn.Module):
    """
    Lightweight, High-Performance Content + Style Fusion Engine with Allograph Learning.
    
    Architecture Overview:
    - Stage 1: Feature projections for content and style sequence.
    - Stage 2: Local stroke & ligature 1D Conv (k=5) for character-to-character transition context.
    - Stage 3: Unified Allograph Cross-Attention & Character-Conditioned Modulation.
    - Stage 4: Global Style Modulation Residual (pools style tokens to predict global slant/density bias).
    """
    def __init__(self, d_model, style_dim, d_state=16, d_conv=4, expand=2, vocab_size=256):
        super().__init__()
        self.d_model = d_model
        
        # 1. Feature Projections
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.content_proj = nn.Linear(d_model, d_model)
        
        # 2. Local Stroke & Ligature 1D Conv (Depthwise-Separable, k=5)
        self.local_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1)
        )
        self.local_gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. Dynamic Allograph Cross-Attention & Modulation
        self.cross_attn = StyleContentCrossAttention(d_model, nhead=4, vocab_size=vocab_size)
        self.norm = RMSNorm(d_model)
        
        # 4. Global Style Modulation Residual Bias
        self.global_style_mod = nn.Sequential(
            nn.Linear(style_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )

    def forward(self, content_seq, style_seq, char_ids=None, y_lens=None):
        style_seq = ensure_dim3(style_seq)
        B, L, D = content_seq.shape
        
        # Sequence mask for padding tokens
        mask = None
        if y_lens is not None:
            range_tensor = torch.arange(L, device=content_seq.device).unsqueeze(0)
            mask = (range_tensor < y_lens.unsqueeze(1))
            
        # --- STAGE 1: Feature Projection ---
        s_feat = self.style_proj(style_seq)
        c_feat = self.content_proj(content_seq)
        if mask is not None:
            c_feat = c_feat * mask.to(c_feat.dtype).unsqueeze(-1)
            
        # --- STAGE 2: Local Stroke & Ligature Enhancement (1D Conv) ---
        c_trans = c_feat.transpose(1, 2)
        local_feat = self.local_cnn(c_trans)
        gate_val = self.local_gate(c_trans)
        content_local = c_feat + (local_feat * gate_val).transpose(1, 2)
        content_local = self.norm(content_local + content_seq)
        if mask is not None:
            content_local = content_local * mask.to(content_local.dtype).unsqueeze(-1)
            
        # --- STAGE 3: Unified Allograph Cross-Attention & Character Modulation ---
        content_fused = self.cross_attn(content_local, s_feat, char_ids=char_ids, mask=mask)
        if mask is not None:
            content_fused = content_fused * mask.to(content_fused.dtype).unsqueeze(-1)
            
        # --- STAGE 4: Global Style Residual Modulation ---
        style_vec = torch.mean(style_seq, dim=1) # Global mean pool across S style tokens
        mod_params = self.global_style_mod(style_vec).unsqueeze(1) # (B, 1, D*2)
        scale, shift = mod_params.chunk(2, dim=-1)
        
        out = content_fused * (1 + scale) + shift
        if mask is not None:
            out = out * mask.to(out.dtype).unsqueeze(-1)
            
        return out


class MixFusion(nn.Module):
    def __init__(self, d_model, style_dim, vocab_size=256):
        super().__init__()
        self.fusion = StyleContentFusion(d_model, style_dim, vocab_size=vocab_size)
        
    def forward(self, content_seq, style_seq, char_ids=None, y_lens=None):
        return self.fusion(content_seq, style_seq, char_ids=char_ids, y_lens=y_lens)

