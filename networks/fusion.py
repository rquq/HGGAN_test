import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ensure_dim3


def _logit(probability):
    return math.log(probability / (1.0 - probability))


class StyleConditionedSelfAttention(nn.Module):
    """Word-level content context, conditioned only by the global style token."""

    def __init__(self, d_model, style_dim, nhead=4, attn_dim=128,
                 ffn_dim=None, max_seq_len=32, residual_init=0.25,
                 conditioning_limit=0.5):
        super().__init__()
        if attn_dim % nhead:
            raise ValueError('attn_dim must be divisible by nhead')
        if not 0.0 < residual_init < 1.0:
            raise ValueError('residual_init must be strictly between 0 and 1')

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = attn_dim // nhead
        self.max_seq_len = max_seq_len
        self.residual_init = residual_init
        self.conditioning_limit = float(conditioning_limit)
        ffn_dim = ffn_dim or d_model * 2

        self.attn_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn_residual_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, attn_dim * 3, bias=False)
        self.attn_out = nn.Linear(attn_dim, d_model, bias=False)
        self.ffn_in = nn.Linear(d_model, ffn_dim * 2)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

        # Global style controls both normalized branches and their residual strength.
        # It never provides attention keys/values; reference modulation owns local style routing.
        self.style_mod = nn.Linear(style_dim, d_model * 4 + 2)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(nhead, max_seq_len * 2 - 1)
        )
        self.reset_stability_parameters()

    def reset_stability_parameters(self):
        nn.init.normal_(self.style_mod.weight, 0.0, 0.01)
        nn.init.zeros_(self.style_mod.bias)
        with torch.no_grad():
            self.style_mod.bias[-2:].fill_(_logit(self.residual_init))
        nn.init.zeros_(self.relative_position_bias)

    @staticmethod
    def _condition(x, shift, scale):
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _attention_bias(self, length, batch_size, mask, dtype, device):
        positions = torch.arange(length, device=device)
        relative = positions[:, None] - positions[None, :]
        relative = relative.clamp(
            -self.max_seq_len + 1, self.max_seq_len - 1
        ) + self.max_seq_len - 1
        bias = self.relative_position_bias[:, relative].unsqueeze(0).to(dtype=dtype)
        if mask is None:
            return bias
        bias = bias.expand(batch_size, -1, -1, -1).clone()
        return bias.masked_fill(
            ~mask[:, None, None, :], torch.finfo(dtype).min
        )

    def forward(self, content_seq, global_style, mask=None):
        batch_size, length, _ = content_seq.shape
        modulation = self.style_mod(global_style)
        (
            shift_attn,
            scale_attn,
            shift_ffn,
            scale_ffn,
            gate_attn,
            gate_ffn,
        ) = torch.split(
            modulation,
            [self.d_model, self.d_model, self.d_model, self.d_model, 1, 1],
            dim=-1,
        )
        shift_attn = self.conditioning_limit * torch.tanh(shift_attn)
        scale_attn = self.conditioning_limit * torch.tanh(scale_attn)
        shift_ffn = self.conditioning_limit * torch.tanh(shift_ffn)
        scale_ffn = self.conditioning_limit * torch.tanh(scale_ffn)

        attn_input = self._condition(
            self.attn_norm(content_seq), shift_attn, scale_attn
        )
        qkv = self.qkv(attn_input).view(
            batch_size, length, 3, self.nhead, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention_bias = self._attention_bias(
            length, batch_size, mask, query.dtype, query.device
        )
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_bias,
            dropout_p=0.0, is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, length, -1)
        content_seq = (
            content_seq
            + torch.sigmoid(gate_attn).unsqueeze(1) * self.attn_out(attended)
        )
        if mask is not None:
            content_seq = content_seq * mask.unsqueeze(-1).to(content_seq.dtype)

        ffn_input = self._condition(
            self.ffn_norm(content_seq), shift_ffn, scale_ffn
        )
        value_branch, gate_branch = self.ffn_in(ffn_input).chunk(2, dim=-1)
        ffn_output = self.ffn_out(value_branch * F.silu(gate_branch))
        ffn_output = self.ffn_residual_norm(ffn_output)
        content_seq = content_seq + torch.sigmoid(gate_ffn).unsqueeze(1) * ffn_output
        if mask is not None:
            content_seq = content_seq * mask.unsqueeze(-1).to(content_seq.dtype)
        return content_seq


class ReferenceStyleModulation(nn.Module):
    """Retrieve reference detail from contextual target features, not raw IDs.

    Pre-normalized attention and one learned residual scale replace forced
    uniform routing, vocabulary-specific gates and stacked amplitude caps.
    """

    def __init__(self, d_model, modulation_residual_init=0.15):
        super().__init__()
        if modulation_residual_init <= 0:
            raise ValueError('modulation_residual_init must be positive')
        self.modulation_residual_init = float(modulation_residual_init)
        self.content_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.style_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.reference_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mod_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )
        self.modulation_scale = nn.Parameter(torch.full(
            (d_model,), self.modulation_residual_init
        ))
        self.reset_stability_parameters()

    def reset_stability_parameters(self):
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.normal_(self.mod_proj[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.mod_proj[-1].bias)
        with torch.no_grad():
            self.modulation_scale.fill_(self.modulation_residual_init)

    def forward(self, content_seq, local_style_seq, mask=None):
        local_style_seq = ensure_dim3(local_style_seq)
        if local_style_seq.size(1) < 1:
            raise ValueError('reference modulation requires at least one local token')
        content = self.content_norm(content_seq)
        style = self.style_norm(local_style_seq)
        retrieved = F.scaled_dot_product_attention(
            self.q_proj(content).unsqueeze(1),
            self.k_proj(style).unsqueeze(1),
            self.v_proj(style).unsqueeze(1), dropout_p=0.0,
        ).squeeze(1)
        scale, shift = self.mod_proj(self.reference_norm(retrieved)).chunk(2, dim=-1)
        output = content_seq + self.modulation_scale * (content * scale + shift)
        if mask is not None:
            output = output.masked_fill(~mask.unsqueeze(-1), 0.0)
        return output


class StyleContentAttentionFusion(nn.Module):
    """Coarse-to-fine content/style fusion with one unambiguous job per stage."""

    def __init__(self, d_model, style_dim, nhead=4, attn_dim=128,
                 ffn_dim=None, max_seq_len=32,
                 local_projection_residual_init=0.1,
                 reference_residual_init=0.15):
        super().__init__()
        if not 0.0 < local_projection_residual_init < 1.0:
            raise ValueError(
                'local_projection_residual_init must be strictly between 0 and 1'
            )
        self.d_model = d_model
        self.local_projection_residual_init = float(
            local_projection_residual_init
        )
        self.local_style_proj = nn.Sequential(
            nn.Linear(style_dim, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.local_projection_gate_logits = nn.Parameter(torch.full(
            (d_model,), _logit(self.local_projection_residual_init)
        ))
        self.local_style_input_norm = nn.LayerNorm(style_dim, elementwise_affine=False)
        self.local_style_output_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.content_context = StyleConditionedSelfAttention(
            d_model=d_model, style_dim=style_dim, nhead=nhead,
            attn_dim=attn_dim, ffn_dim=ffn_dim, max_seq_len=max_seq_len,
        )
        self.local_depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=5, padding=2, groups=d_model
        )
        self.local_pointwise = nn.Linear(d_model, d_model)
        self.local_selector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.local_residual_gate_logits = nn.Parameter(
            torch.full((d_model,), _logit(0.25))
        )
        # Retain the existing namespace; this feature-only module has a new
        # parameter schema and is not an exact resume of the former GAN.
        self.allograph_mod = ReferenceStyleModulation(
            d_model, modulation_residual_init=reference_residual_init,
        )
        self.reset_stability_parameters()

    def reset_stability_parameters(self):
        self.content_context.reset_stability_parameters()
        self.allograph_mod.reset_stability_parameters()
        # The tall bias-free base projection preserves angles in style space;
        # the learned MLP is only a small residual refinement.
        nn.init.orthogonal_(self.local_style_proj[0].weight)
        nn.init.normal_(self.local_style_proj[2].weight, 0.0, 0.01)
        with torch.no_grad():
            self.local_projection_gate_logits.fill_(
                _logit(self.local_projection_residual_init)
            )
            self.local_residual_gate_logits.fill_(_logit(0.25))

    def forward(self, content_seq, style_seq, y_lens=None):
        style_seq = ensure_dim3(style_seq)
        if style_seq.size(1) < 2:
            raise ValueError(
                'fusion requires token 0 as global style and at least one local style token'
            )

        _, max_length, _ = content_seq.shape
        mask = None
        if y_lens is not None:
            valid_lengths = y_lens.to(content_seq.device).long().clamp(1, max_length)
            positions = torch.arange(max_length, device=content_seq.device).unsqueeze(0)
            mask = positions < valid_lengths.unsqueeze(1)
            content_seq = content_seq * mask.unsqueeze(-1).to(content_seq.dtype)

        global_style = style_seq[:, 0]
        # Keep local-token identity through the 32→d_model handoff. The old
        # two-layer projection mapped distinct tokens back to one nearly
        # identical vector while these existing weights can form a good residual.
        local_style_input = self.local_style_input_norm(style_seq[:, 1:])
        local_style_hidden = self.local_style_proj[0](local_style_input)
        projection_strength = torch.sigmoid(
            self.local_projection_gate_logits
        ).view(1, 1, -1)
        local_style = self.local_style_output_norm(
            local_style_hidden
            + projection_strength * self.local_style_proj[2](
                F.silu(local_style_hidden)
            )
        )

        # Stage 1: global style changes word-level content relationships, not glyph routing.
        content_context = self.content_context(content_seq, global_style, mask=mask)

        # Stage 2: local character continuity with a small, learnable residual handoff.
        context_channels = content_context.transpose(1, 2)
        local_delta = self.local_depthwise(context_channels).transpose(1, 2)
        local_delta = self.local_pointwise(F.silu(local_delta))
        local_delta = local_delta * self.local_selector(content_context)
        local_strength = torch.sigmoid(self.local_residual_gate_logits).view(1, 1, -1)
        content_local = content_context + local_strength * local_delta
        if mask is not None:
            content_local = content_local * mask.unsqueeze(-1).to(content_local.dtype)

        # Stage 3: contextual target features retrieve local reference detail.
        return self.allograph_mod(
            content_local, local_style, mask=mask
        )
