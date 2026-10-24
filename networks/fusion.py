import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import get_2d_sinusoidal_embeddings


def _mask_tokens(sequence, mask):
    if mask is None:
        return sequence
    return sequence.masked_fill(~mask.unsqueeze(-1), 0.0)


class ContentSelfAttention(nn.Module):
    """Mix target positions without reference-style conditioning."""

    def __init__(self, d_model, nhead=4, attn_dim=128,
                 max_seq_len=32):
        super().__init__()
        if nhead < 1 or attn_dim < 1 or attn_dim % nhead:
            raise ValueError('attn_dim must be positive and divisible by nhead')
        if max_seq_len < 1:
            raise ValueError('max_seq_len must be positive')
        self.nhead = nhead
        self.head_dim = attn_dim // nhead
        self.max_seq_len = max_seq_len
        self.content_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, attn_dim * 3, bias=False)
        self.attn_out = nn.Linear(attn_dim, d_model, bias=False)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(nhead, max_seq_len * 2 - 1)
        )

    def reset_stability_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.attn_out.weight)
        nn.init.zeros_(self.relative_position_bias)

    def forward(self, content_seq, mask=None):
        batch_size, length, _ = content_seq.shape
        conditioned = self.content_norm(content_seq)
        qkv = self.qkv(conditioned).view(
            batch_size, length, 3, self.nhead, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        positions = torch.arange(length, device=content_seq.device)
        relative = (positions[:, None] - positions[None, :]).clamp(
            -self.max_seq_len + 1, self.max_seq_len - 1
        ) + self.max_seq_len - 1
        bias = self.relative_position_bias[:, relative].unsqueeze(0).to(query.dtype)
        if mask is not None:
            bias = bias.expand(batch_size, -1, -1, -1).masked_fill(
                ~mask[:, None, None, :], float('-inf')
            )
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch_size, length, -1)
        return _mask_tokens(self.attn_out(attended), mask)


class SpatialStyleModulation(nn.Module):
    """Use retrieved reference evidence to modulate early generator features.

    The reference branch predicts a per-position, per-channel residual gain,
    rather than adding its visual features directly to the generator stream.
    The gain and its learned scale are unconstrained; this is not an artifact
    filter or a guarantee that downstream convolutions preserve glyph shapes.
    """

    def __init__(self, d_model, memory_dim=64, memory_height=4,
                 nhead=4, attn_dim=128, residual_init=0.25):
        super().__init__()
        if nhead < 1 or attn_dim < 1 or attn_dim % nhead:
            raise ValueError('attn_dim must be positive and divisible by nhead')
        if min(d_model, memory_dim, memory_height) < 1 or residual_init <= 0:
            raise ValueError('Feature/memory dimensions and residual_init must be positive')
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.memory_height = memory_height
        self.residual_init = float(residual_init)
        self.nhead = nhead
        self.head_dim = attn_dim // nhead
        self.feature_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(memory_dim, elementwise_affine=False)
        self.q_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, attn_dim, bias=False)
        self.gain_proj = nn.Linear(attn_dim, d_model, bias=False)
        self.reference_scale = nn.Parameter(torch.full((d_model,), self.residual_init))
        self.reset_stability_parameters()

    def reset_stability_parameters(self):
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.gain_proj):
            nn.init.xavier_uniform_(layer.weight)
        with torch.no_grad():
            self.reference_scale.fill_(self.residual_init)

    def forward(self, feature, reference_memory=None, memory_valid_mask=None,
                x_lens=None):
        if feature.ndim != 4 or feature.size(1) != self.d_model:
            raise ValueError('Spatial retrieval requires BCHW features with matching channels')
        if reference_memory is None:
            if memory_valid_mask is not None:
                raise ValueError('A memory mask requires reference memory')
            return feature
        if (reference_memory.ndim != 3
                or reference_memory.size(0) != feature.size(0)
                or reference_memory.size(2) != self.memory_dim
                or reference_memory.size(1) < self.memory_height
                or reference_memory.size(1) % self.memory_height):
            raise ValueError('Reference memory must be B x (height * width) x memory_dim')
        if (memory_valid_mask is None or memory_valid_mask.dtype != torch.bool
                or memory_valid_mask.shape != reference_memory.shape[:2]
                or memory_valid_mask.device != feature.device
                or reference_memory.device != feature.device):
            raise ValueError('Reference memory requires a matching boolean validity mask')

        # Random-style rows bypass retrieval entirely: never run an all-masked
        # softmax, and never invent reference features from Gaussian latents.
        active = memory_valid_mask.any(dim=1).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return feature
        selected = feature.index_select(0, active)
        memory = reference_memory.index_select(0, active)
        valid_memory = memory_valid_mask.index_select(0, active)
        memory = memory.masked_fill(~valid_memory.unsqueeze(-1), 0.0)
        batch_size, channels, height, width = selected.shape
        sequence = selected.flatten(2).transpose(1, 2)
        query_mask = None
        if x_lens is not None:
            lengths = x_lens.to(feature.device).index_select(0, active).long().clamp(1, width)
            query_mask = (
                torch.arange(width, device=feature.device)[None, None, :]
                < lengths[:, None, None]
            ).expand(-1, height, -1).reshape(batch_size, -1)
        sequence = _mask_tokens(sequence, query_mask)

        # Position influences selection only; values remain visual evidence.
        position = get_2d_sinusoidal_embeddings(
            self.memory_height, memory.size(1) // self.memory_height,
            self.memory_dim, memory.device,
        ).reshape(1, memory.size(1), self.memory_dim).to(memory.dtype)
        keys = self.memory_norm(memory) + position
        query = self.q_proj(self.feature_norm(sequence)).view(
            batch_size, sequence.size(1), self.nhead, self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(keys).view(
            batch_size, memory.size(1), self.nhead, self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(memory).view(
            batch_size, memory.size(1), self.nhead, self.head_dim,
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=valid_memory[:, None, None, :],
            dropout_p=0.0, is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, sequence.size(1), -1)
        refined = _mask_tokens(
            sequence * (1.0 + self.reference_scale * self.gain_proj(attended)),
            query_mask,
        )
        refined = refined.transpose(1, 2).reshape(batch_size, channels, height, width)
        return feature.index_copy(0, active, refined)


class LocalGatedFeedForward(nn.Module):
    """Content transformation with neighbouring-position context."""

    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.expand = nn.Linear(d_model, hidden_dim * 2)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim
        )
        self.project = nn.Linear(hidden_dim, d_model)

    def reset_stability_parameters(self):
        nn.init.xavier_uniform_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='linear')
        nn.init.zeros_(self.depthwise.bias)
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, content_seq, mask=None):
        value, gate = self.expand(self.norm(content_seq)).chunk(2, dim=-1)
        # Clear affine biases before width mixing so extra batch padding cannot
        # leak into the final valid characters.
        value = _mask_tokens(value, mask)
        value = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
        transformed = self.project(value * F.silu(gate))
        return _mask_tokens(transformed, mask)


class ContentContext(nn.Module):
    """Target self-attention and local mixing before constructing the G seed.

    Each pre-normalized sublayer has one direct learnable residual scale.
    Nonzero initialization enables immediate learning, without sigmoid
    ceilings, nested output gates, forced-uniform attention or RMS caps.
    The same topology serves every supported image height.
    """

    def __init__(self, d_model, nhead=4, attn_dim=128,
                 ffn_dim=None, max_seq_len=32, residual_init=0.25):
        super().__init__()
        if residual_init <= 0:
            raise ValueError('residual_init must be positive')
        hidden_dim = ffn_dim if ffn_dim is not None else d_model * 3
        if hidden_dim < 1:
            raise ValueError('ffn_dim must be positive')
        self.residual_init = float(residual_init)
        self.content_attention = ContentSelfAttention(
            d_model, nhead=nhead, attn_dim=attn_dim,
            max_seq_len=max_seq_len,
        )
        self.local_mixer = LocalGatedFeedForward(d_model, hidden_dim)
        self.context_scale = nn.Parameter(torch.full((d_model,), self.residual_init))
        self.local_scale = nn.Parameter(torch.full((d_model,), self.residual_init))
        self.reset_stability_parameters()

    @property
    def residual_scales(self):
        return torch.stack((self.context_scale, self.local_scale))

    def reset_stability_parameters(self):
        self.content_attention.reset_stability_parameters()
        self.local_mixer.reset_stability_parameters()
        with torch.no_grad():
            for scale in (self.context_scale, self.local_scale):
                scale.fill_(self.residual_init)

    def forward(self, content_seq, y_lens=None):
        if content_seq.size(1) < 1:
            raise ValueError('fusion requires at least one target position')
        mask = None
        if y_lens is not None:
            lengths = y_lens.to(content_seq.device).long().clamp(1, content_seq.size(1))
            positions = torch.arange(content_seq.size(1), device=content_seq.device)
            mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        content_seq = _mask_tokens(content_seq, mask)
        content_seq = _mask_tokens(
            content_seq + self.context_scale * self.content_attention(
                content_seq, mask=mask
            ), mask,
        )
        return _mask_tokens(
            content_seq + self.local_scale * self.local_mixer(content_seq, mask=mask), mask
        )
