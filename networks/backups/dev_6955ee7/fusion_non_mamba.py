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


class StyleCrossSequenceBlock(nn.Module):
    """
    Pure PyTorch Sequence-Wide Context Engine (Non-Mamba Replacement).
    Uses a 1D Bidirectional GRU to scan character sequences across full sequence length L,
    cross-gated with style sequence features to achieve global sequence context.
    """
    def __init__(self, d_model, style_dim):
        super().__init__()
        self.d_model = d_model
        self.rnn = nn.GRU(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.style_gate_proj = nn.Sequential(
            nn.Linear(style_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x, style_seq, mask=None):
        style_seq = ensure_dim3(style_seq)
        if mask is not None:
            f_mask = mask.to(x.dtype).unsqueeze(-1)
            x_in = x * f_mask
        else:
            x_in = x

        rnn_out, _ = self.rnn(x_in)
        style_vec = style_seq.sum(dim=1) / style_seq.size(1)
        gate = self.style_gate_proj(style_vec).unsqueeze(1)
        out = self.norm(rnn_out * gate)

        if mask is not None:
            out = out * f_mask
        return out


class StyleContentCrossAttention(nn.Module):
    """
    Direct Query-Key-Value Cross-Attention using pure PyTorch.
    Allows content character tokens to directly query and align with style sequence features,
    guided by a learned character-conditioned allograph routing prior.
    """
    def __init__(self, d_model, nhead=4, dropout=0.1, routing_dim=16, vocab_size=256):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.vocab_size = vocab_size
        self.warned_out_of_vocab = False
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Learned character-to-style token routing layers
        self.char_routing_emb = nn.Embedding(vocab_size, routing_dim)
        self.context_routing_proj = nn.Linear(d_model, routing_dim)
        self.style_routing_proj = nn.Linear(d_model, routing_dim)

    def forward(self, content_seq, style_seq, char_ids=None, mask=None):
        """
        Args:
            content_seq: (B, L, D) sequence of content embeddings
            style_seq: (B, S, D) sequence of style embeddings
            char_ids: (B, L) raw character labels
            mask: (B, L) sequence mask for content tokens
        """
        style_seq = ensure_dim3(style_seq)
        B, L, D = content_seq.shape
        S = style_seq.shape[1]
        
        # Project and reshape for Multi-Head: (B, nh, SeqLen, head_dim)
        q = self.q_proj(content_seq).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(style_seq).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(style_seq).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        # Pure, version-friendly scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q * scale, k.transpose(-2, -1)) # (B, nh, L, S)
        
        # Add character-conditioned learned routing prior
        if char_ids is not None:
            if not self.warned_out_of_vocab and (char_ids >= self.vocab_size).any():
                self.warned_out_of_vocab = True
                print(f"[Warning] Found character IDs exceeding vocab_size ({self.vocab_size}) in StyleContentCrossAttention. "
                      f"Max ID found: {char_ids.max().item()}. Clamping to range [0, {self.vocab_size - 1}].")
            char_ids_clipped = torch.clamp(char_ids, 0, self.vocab_size - 1)
            char_q_static = self.char_routing_emb(char_ids_clipped)
            char_q_context = self.context_routing_proj(content_seq)
            char_q = char_q_static + char_q_context
            
            style_routing = self.style_routing_proj(style_seq)
            routing_prior = torch.matmul(char_q, style_routing.transpose(-2, -1))
            scores = scores + routing_prior.unsqueeze(1)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v) # (B, nh, L, head_dim)
        
        # Reshape back to (B, L, D) and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)
        
        x = self.norm1(content_seq + attn_out)
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        if mask is not None:
            out = out * mask.to(out.dtype).unsqueeze(-1)
        return out


class AllographicModulation(nn.Module):
    """
    Dynamic character-conditioned allograph modulation (AdaIN style).
    Allows each content character token to dynamically pool style tokens that best match
    its spatial/glyph properties, predicting character-specific scale and shift.
    """
    def __init__(self, d_model, routing_dim=16, vocab_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.warned_out_of_vocab = False
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.mod_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )
        # Learned character-to-style token routing layers
        self.char_routing_emb = nn.Embedding(vocab_size, routing_dim)
        self.context_routing_proj = nn.Linear(d_model, routing_dim)
        self.style_routing_proj = nn.Linear(d_model, routing_dim)
        
    def forward(self, content_seq, style_seq, char_ids=None, mask=None):
        """
        Args:
            content_seq: (B, L, D) refined content sequence
            style_seq: (B, S, D) style sequence features
            char_ids: (B, L) raw character labels to guide vertical styling
            mask: (B, L) sequence mask for content tokens
        """
        style_seq = ensure_dim3(style_seq)
        B, L, D = content_seq.shape
        S = style_seq.shape[1]
        
        # Compute dynamic character-conditioned query-key alignment weights
        q = self.q_proj(content_seq) # (B, L, D)
        k = self.k_proj(style_seq)   # (B, S, D)
        v = self.v_proj(style_seq)   # (B, S, D)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5) # (B, L, S)
        
        # Add character-conditioned learned routing prior
        if char_ids is not None:
            if not self.warned_out_of_vocab and (char_ids >= self.vocab_size).any():
                self.warned_out_of_vocab = True
                print(f"[Warning] Found character IDs exceeding vocab_size ({self.vocab_size}) in AllographicModulation. "
                      f"Max ID found: {char_ids.max().item()}. Clamping to range [0, {self.vocab_size - 1}].")
            char_ids_clipped = torch.clamp(char_ids, 0, self.vocab_size - 1)
            char_q_static = self.char_routing_emb(char_ids_clipped) # (B, L, routing_dim)
            char_q_context = self.context_routing_proj(content_seq) # (B, L, routing_dim)
            char_q = char_q_static + char_q_context
            
            style_routing = self.style_routing_proj(style_seq) # (B, S, routing_dim)
            routing_prior = torch.matmul(char_q, style_routing.transpose(-2, -1)) # (B, L, S)
            scores = scores + routing_prior
            
        attn_weights = torch.softmax(scores, dim=-1) # (B, L, S)
        
        # Character-specific style features: (B, L, S) * (B, S, D) -> (B, L, D)
        style_char = torch.matmul(attn_weights, v) 
        
        # Dynamic AdaIN scale & shift parameters per character
        mod_params = self.mod_proj(style_char) # (B, L, D*2)
        scale, shift = mod_params.chunk(2, dim=-1)
        
        out = content_seq * (1 + scale) + shift
        if mask is not None:
            out = out * mask.to(out.dtype).unsqueeze(-1)
        return out


class StyleContentFusion(nn.Module):
    """
    High-Performance Content + Style Fusion Engine with Allograph Learning & Sequence Context.
    
    Architecture Pipeline (Mirrors Mamba Fusion 6-stage architecture):
    - Stage 1: Feature projections for content and style sequence.
    - Stage 2: 1D Bidirectional Sequence Context Engine (Sequence-wide scan).
    - Stage 3: Local Stroke Boundary 1D CNN Gate (smooths scan noise & preserves boundaries).
    - Stage 4: Dynamic Cross-Attention for Allograph Learning.
    - Stage 5: Character-Conditioned Allographic Modulation (Dedicated AdaIN).
    - Stage 6: Global Style Modulation Residual (pools style tokens to predict global slant/density bias).
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
        
        # 2. Sequential 1D Sequence-Wide Context Engine (Non-Mamba scan replacement)
        self.seq_scan = StyleCrossSequenceBlock(d_model, style_dim=style_dim)
        
        # 3. Local Stroke Boundary 1D CNN Gate
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
        
        # 4. Dynamic Cross-Attention for Allograph Learning
        self.cross_attn = StyleContentCrossAttention(d_model, nhead=4, vocab_size=vocab_size)
        
        # 5. Normalization and Stability
        self.norm = RMSNorm(d_model)
        
        # 6. Character-Conditioned Allographic Modulation
        self.allograph_mod = AllographicModulation(d_model, vocab_size=vocab_size)
        
        # 7. Global Style Modulation Residual Bias
        self.global_style_mod = nn.Sequential(
            nn.Linear(style_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )

    def forward(self, content_seq, style_seq, char_ids=None, y_lens=None):
        """
        Args:
            content_seq: (B, L, D) sequence of content embeddings
            style_seq: (B, S, style_dim) sequence of style tokens
            char_ids: (B, L) raw character labels
            y_lens: (B,) sequence lengths of content characters
        """
        style_seq = ensure_dim3(style_seq)
        B, L, D = content_seq.shape
        
        # Construct sequence mask
        mask = None
        if y_lens is not None:
            range_tensor = torch.arange(L, device=content_seq.device).unsqueeze(0)
            mask = (range_tensor < y_lens.unsqueeze(1)) # (B, L)
            
        # --- STAGE 1: Sequence Preparation ---
        s_feat = self.style_proj(style_seq) # (B, S_len, D)
        c_feat = self.content_proj(content_seq) # (B, L, D)
        if mask is not None:
            c_feat = c_feat * mask.to(c_feat.dtype).unsqueeze(-1)
            
        # --- STAGE 2: 1D Bidirectional Sequence Context Scan ---
        fused = self.seq_scan(c_feat, style_seq, mask=mask)
        content_fused = self.norm(fused + content_seq)
        if mask is not None:
            content_fused = content_fused * mask.to(content_fused.dtype).unsqueeze(-1)
            
        # --- STAGE 3: Local Stroke Boundary Smoothing (CNN Gate) ---
        c_trans = content_fused.transpose(1, 2)
        local_feat = self.local_cnn(c_trans)
        gate_val = self.local_gate(c_trans)
        content_local = content_fused + (local_feat * gate_val).transpose(1, 2)
        if mask is not None:
            content_local = content_local * mask.to(content_local.dtype).unsqueeze(-1)
            
        # --- STAGE 4: Allograph Refinement via Dynamic Cross-Attention ---
        content_final = self.cross_attn(content_local, s_feat, char_ids=char_ids, mask=mask)
        if mask is not None:
            content_final = content_final * mask.to(content_final.dtype).unsqueeze(-1)
            
        # --- STAGE 5: Allographic Modulation (Dynamic Character-Conditioned AdaIN) ---
        content_modulated = self.allograph_mod(content_final, s_feat, char_ids=char_ids, mask=mask)
        if mask is not None:
            content_modulated = content_modulated * mask.to(content_modulated.dtype).unsqueeze(-1)
            
        # --- STAGE 6: Global Style Modulation Residual ---
        style_vec = style_seq.sum(dim=1) / style_seq.size(1) # (B, style_dim)
        mod_params = self.global_style_mod(style_vec).unsqueeze(1) # (B, 1, D*2)
        scale, shift = mod_params.chunk(2, dim=-1)
        
        out = content_modulated * (1 + scale) + shift
        if mask is not None:
            out = out * mask.to(out.dtype).unsqueeze(-1)
        return out


class MixFusion(nn.Module):
    def __init__(self, d_model, style_dim, vocab_size=256):
        super().__init__()
        self.fusion = StyleContentFusion(d_model, style_dim, vocab_size=vocab_size)
        
    def forward(self, content_seq, style_seq, char_ids=None, y_lens=None):
        return self.fusion(content_seq, style_seq, char_ids=char_ids, y_lens=y_lens)
