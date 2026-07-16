import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Numerically Stable Associative Scan for T4 Optimization ---
def associative_scan_mamba(dA, dB):
    """
    Stable associative scan for h_i = a_i * h_{i-1} + b_i
    dA: (B, L, D, N) - a_i values (in 0-1)
    dB: (B, L, D, N) - b_i values
    Returns: h (B, L, D, N)
    """
    B, L, D, N = dA.shape
    
    # We use a simple but effective recursive doubling (all-prefix-sums)
    # for associative operator (a2, b2) o (a1, b1) = (a2*a1, a2*b1 + b2)
    
    # Recursive doubling (Log L steps)
    res_a = dA
    res_b = dB
    
    step = 1
    while step < L:
        # Shifted versions
        a_left = res_a[:, :-step]
        b_left = res_b[:, :-step]
        
        a_right = res_a[:, step:]
        b_right = res_b[:, step:]
        
        # Combine: (a_r, b_r) o (a_l, b_l) = (a_r * a_l, a_r * b_l + b_r)
        new_a = a_right * a_left
        new_b = a_right * b_left + b_right
        
        # Non-inplace update to preserve gradient tracking
        res_a = torch.cat([res_a[:, :step], new_a], dim=1)
        res_b = torch.cat([res_b[:, :step], new_b], dim=1)
        
        step *= 2
        
    return res_b

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
        
        self.x_proj = make_linear(self.d_inner, 1 + self.d_state * 2, bias=False)
        self.dt_proj = make_linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, self.d_state + 1).repeat(self.d_inner, 1).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = make_linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(self.d_inner)
        
        if self.bidirectional:
            self.x_proj_b = make_linear(self.d_inner, 1 + self.d_state * 2, bias=False)
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

    def _core_scan(self, x_conv, dt_proj, x_proj, A_log, D, mask=None, reverse=False):
        if reverse:
            x_conv = torch.flip(x_conv, [1])
            if mask is not None:
                mask = torch.flip(mask, [1])
            
        proj = x_proj(x_conv)
        delta, B, C = torch.split(proj, [1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(dt_proj(delta)) 
        A = -torch.exp(A_log.float()) 
        
        dA = torch.exp(delta.unsqueeze(-1) * A) 
        dB = (delta * x_conv).unsqueeze(-1) * B.unsqueeze(2)
        
        if mask is not None:
            m = (mask > 0).unsqueeze(-1).unsqueeze(-1) # (B, L, 1, 1)
            dA = torch.where(m, dA, torch.ones_like(dA))
            dB = torch.where(m, dB, torch.zeros_like(dB))
        
        h = associative_scan_mamba(dA, dB)
        y = torch.sum(h * C.unsqueeze(2), dim=-1)
        
        y = y + x_conv * D
        
        if reverse:
            y = torch.flip(y, [1])
        return y

    def forward(self, x, mask=None):
        L = x.shape[1]
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        if mask is not None:
            f_mask = mask.to(x.dtype).unsqueeze(-1)
            x_in = x_in * f_mask
            z = z * f_mask
        
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        if mask is not None:
            f_mask = mask.to(x.dtype).unsqueeze(-1)
            x_conv = x_conv * f_mask
        
        y = self._core_scan(x_conv, self.dt_proj, self.x_proj, self.A_log, self.D, mask=mask)
        
        if self.bidirectional:
            y_b = self._core_scan(x_conv, self.dt_proj_b, self.x_proj_b, self.A_log_b, self.D_b, mask=mask, reverse=True)
            y = (y + y_b) * 0.5
            
        out = self.out_proj(self.norm(y) * F.silu(z))
        if mask is not None:
            out = out * mask.to(x.dtype).unsqueeze(-1)
        return out




class StyleCrossMambaBlock(nn.Module):
    def __init__(self, d_model, style_dim, d_state=16, d_conv=4, expand=2, bidirectional=True, use_sn=False):
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
        
        self.x_proj = make_linear(self.d_inner, 1 + self.d_state * 2, bias=False)
        self.dt_proj = make_linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, self.d_state + 1).repeat(self.d_inner, 1).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = make_linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(self.d_inner)
        
        if self.bidirectional:
            self.x_proj_b = make_linear(self.d_inner, 1 + self.d_state * 2, bias=False)
            self.dt_proj_b = make_linear(1, self.d_inner, bias=True)
            self.A_log_b = nn.Parameter(torch.log(A))
            self.D_b = nn.Parameter(torch.ones(self.d_inner))
            
        # Style cross-attention projections at gating stage
        self.style_proj = make_linear(style_dim, self.d_inner, bias=False)
        self.q_proj = make_linear(self.d_inner, self.d_inner, bias=False)
        self.k_proj = make_linear(self.d_inner, self.d_inner, bias=False)
        self.v_proj = make_linear(self.d_inner, self.d_inner, bias=False)
        self.style_gate_proj = make_linear(self.d_inner, self.d_inner, bias=True)
        
        self._custom_init()

    def _custom_init(self):
        dt_min, dt_max = 0.001, 0.1
        dt_init_floor = 1e-4
        for dt_proj in ([self.dt_proj, self.dt_proj_b] if self.bidirectional else [self.dt_proj]):
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad(): dt_proj.bias.copy_(inv_dt)
            dt_proj.weight.data.fill_(0.1)

    def _core_scan(self, x_conv, dt_proj, x_proj, A_log, D, mask=None, reverse=False):
        if reverse:
            x_conv = torch.flip(x_conv, [1])
            if mask is not None:
                mask = torch.flip(mask, [1])
            
        proj = x_proj(x_conv)
        delta, B, C = torch.split(proj, [1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(dt_proj(delta)) 
        A = -torch.exp(A_log.float()) 
        
        dA = torch.exp(delta.unsqueeze(-1) * A) 
        dB = (delta * x_conv).unsqueeze(-1) * B.unsqueeze(2)
        
        if mask is not None:
            m = (mask > 0).unsqueeze(-1).unsqueeze(-1) # (B, L, 1, 1)
            dA = torch.where(m, dA, torch.ones_like(dA))
            dB = torch.where(m, dB, torch.zeros_like(dB))
        
        h = associative_scan_mamba(dA, dB)
        y = torch.sum(h * C.unsqueeze(2), dim=-1)
        
        y = y + x_conv * D
        
        if reverse:
            y = torch.flip(y, [1])
        return y

    def forward(self, x, style_seq, mask=None):
        L = x.shape[1]
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        if mask is not None:
            f_mask = mask.to(x.dtype).unsqueeze(-1)
            x_in = x_in * f_mask
            z = z * f_mask
        
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        if mask is not None:
            f_mask = mask.to(x.dtype).unsqueeze(-1)
            x_conv = x_conv * f_mask
        
        y = self._core_scan(x_conv, self.dt_proj, self.x_proj, self.A_log, self.D, mask=mask)
        
        if self.bidirectional:
            y_b = self._core_scan(x_conv, self.dt_proj_b, self.x_proj_b, self.A_log_b, self.D_b, mask=mask, reverse=True)
            y = (y + y_b) * 0.5
            
        # Draw keys/values from the style sequence
        s_feat = self.style_proj(style_seq) # (B, S, d_inner)
        q = self.q_proj(y) # (B, L, d_inner)
        k = self.k_proj(s_feat) # (B, S, d_inner)
        v = self.v_proj(s_feat) # (B, S, d_inner)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_inner ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        style_context = torch.matmul(attn_weights, v)
        
        style_gate = torch.sigmoid(self.style_gate_proj(style_context))
        y_modulated = y * (1 - style_gate) + style_context * style_gate
        
        out = self.out_proj(self.norm(y_modulated) * F.silu(z))
        if mask is not None:
            out = out * mask.to(x.dtype).unsqueeze(-1)
        return out