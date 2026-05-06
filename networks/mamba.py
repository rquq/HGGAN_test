import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import _len2mask

# --- JIT Optimized Scan Functions ---
# --- Optimized JIT Scan Functions ---
@torch.jit.script
def selective_scan_fwd(x_conv, deltaA, deltaB_u, C_proj):
    B, L, D_inner, D_state = deltaA.shape
    h = torch.zeros((B, D_inner, D_state), device=x_conv.device, dtype=x_conv.dtype)
    y = torch.empty((B, L, D_inner), device=x_conv.device, dtype=x_conv.dtype)
    
    # Pre-calculate projected C for the whole sequence to minimize operations in loop
    # B, L, D_inner, D_state * B, L, 1, D_state -> B, L, D_inner, D_state
    for i in range(L):
        h = deltaA[:, i] * h + deltaB_u[:, i]
        # Vectorized dot product over the state dimension
        y[:, i] = torch.sum(h * C_proj[:, i].unsqueeze(1), dim=-1)
    return y

@torch.jit.script
def selective_scan_bwd(x_conv, deltaA, deltaB_u, C_proj):
    B, L, D_inner, D_state = deltaA.shape
    h = torch.zeros((B, D_inner, D_state), device=x_conv.device, dtype=x_conv.dtype)
    y = torch.empty((B, L, D_inner), device=x_conv.device, dtype=x_conv.dtype)
    
    for i in range(L - 1, -1, -1):
        h = deltaA[:, i] * h + deltaB_u[:, i]
        y[:, i] = torch.sum(h * C_proj[:, i].unsqueeze(1), dim=-1)
    return y

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, bidirectional=True, use_sn=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.bidirectional = bidirectional
        self.d_inner = int(self.expand * self.d_model)

        def make_linear(in_dim, out_dim, bias=True):
            lin = nn.Linear(in_dim, out_dim, bias=bias)
            if use_sn: return nn.utils.spectral_norm(lin)
            return lin

        self.in_proj = make_linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = make_linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = make_linear(1, self.d_inner, bias=True)
        A = torch.arange(1, self.d_state + 1).repeat(self.d_inner, 1).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = make_linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(self.d_inner)
        if self.bidirectional:
            self.x_proj_b = make_linear(self.d_inner, self.d_state * 2 + 1, bias=False)
            self.dt_proj_b = make_linear(1, self.d_inner, bias=True)
            self.A_log_b = nn.Parameter(torch.log(A))
            self.D_b = nn.Parameter(torch.ones(self.d_inner))
        self._custom_init()

    def _custom_init(self):
        dt_min, dt_max = 0.001, 0.1
        dt_init_floor = 1e-4
        for dt_proj in ([self.dt_proj, self.dt_proj_b] if self.bidirectional else [self.dt_proj]):
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad(): dt_proj.bias.copy_(inv_dt)
            dt_proj.weight.data.fill_(0.1)

    def _prepare_scan(self, x_conv, dt_proj, x_proj, A_log):
        proj = x_proj(x_conv)
        delta, B_proj, C_proj = torch.split(proj, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(dt_proj(delta))
        A = -torch.exp(A_log.float())
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = delta.unsqueeze(-1) * B_proj.unsqueeze(2) * x_conv.unsqueeze(-1)
        return deltaA, deltaB_u, C_proj

    def _scan_step(self, x_conv, dt_proj, x_proj, A_log, D, reverse=False):
        dA, dB, C = self._prepare_scan(x_conv, dt_proj, x_proj, A_log)
        y = selective_scan_bwd(x_conv, dA, dB, C) if reverse else selective_scan_fwd(x_conv, dA, dB, C)
        return y + x_conv * D

    def forward(self, x):
        L = x.shape[1]
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        y = self._scan_step(x_conv, self.dt_proj, self.x_proj, self.A_log, self.D)
        if self.bidirectional:
            y_b = self._scan_step(x_conv, self.dt_proj_b, self.x_proj_b, self.A_log_b, self.D_b, reverse=True)
            y = (y + y_b) * 0.5
        return self.out_proj(self.norm(y) * F.silu(z))

class Mamba2DBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_sn=False):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, bidirectional=False, use_sn=use_sn)
        self.mamba_v = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, bidirectional=False, use_sn=use_sn)
        
        # Gated fusion to learn how to weigh horizontal vs vertical features per pixel
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(self.mamba.d_inner, d_model)

    def forward(self, x, H, W):
        B, L, C = x.shape
        xz = self.mamba.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv_base = self.mamba.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv_base = F.silu(x_conv_base)

        # Horizontal
        y_h_fwd = self.mamba._scan_step(x_conv_base, self.mamba.dt_proj, self.mamba.x_proj, self.mamba.A_log, self.mamba.D, reverse=False)
        y_h_bwd = self.mamba._scan_step(x_conv_base, self.mamba.dt_proj, self.mamba.x_proj, self.mamba.A_log, self.mamba.D, reverse=True)
        y_h = (y_h_fwd + y_h_bwd) * 0.5

        # Vertical
        x_conv_v = x_conv_base.view(B, H, W, -1).transpose(1, 2).reshape(B, L, -1)
        y_v_fwd = self.mamba_v._scan_step(x_conv_v, self.mamba_v.dt_proj, self.mamba_v.x_proj, self.mamba_v.A_log, self.mamba_v.D, reverse=False)
        y_v_bwd = self.mamba_v._scan_step(x_conv_v, self.mamba_v.dt_proj, self.mamba_v.x_proj, self.mamba_v.A_log, self.mamba_v.D, reverse=True)
        y_v = (y_v_fwd + y_v_bwd).view(B, W, H, -1).transpose(1, 2).reshape(B, L, -1) * 0.5

        # Gated interaction between Horizontal and Vertical scans
        # This allows the model to prioritize the word flow (H) or stroke geometry (V) dynamically
        gates = self.gate(x)
        gate_h, gate_v = gates.chunk(2, dim=-1)
        y_fused = (y_h * gate_h + y_v * gate_v)

        out = self.mamba.norm(y_fused) * F.silu(z)
        return self.proj(out)

class MambaAttention(nn.Module):
    def __init__(self, in_dim, which_conv=None):
        super(MambaAttention, self).__init__()
        use_sn = (which_conv is not None and 'SN' in which_conv.__name__)
        # UPGRADE: Using the 2D Vision Mamba block for spatial modeling
        self.mamba = Mamba2DBlock(in_dim, d_state=16, expand=1, use_sn=use_sn)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, x_len=None, **kwargs):
        dims = x.dim()
        if dims == 3:
            x = x.unsqueeze(-1)
        B, C, W, H = x.size()
        # Flatten for 2D Mamba interaction
        x_flat = x.permute(0, 2, 3, 1).reshape(B, W * H, C)
        out_flat = self.mamba(x_flat, W, H)
        # Reshape back to original dimensions
        out = out_flat.view(B, W, H, C).permute(0, 3, 1, 2)
        out = self.gamma * out + x
        if dims == 3:
            out = out.squeeze(-1)
        return out