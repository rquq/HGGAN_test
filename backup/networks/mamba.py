import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    """
    Highly stabilized Mamba implementation with Spectral Norm support.
    """
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
            if use_sn:
                return nn.utils.spectral_norm(lin)
            return lin

        self.in_proj = make_linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _scan(self, x_conv, dt_proj, x_proj, A_log, D, reverse=False):
        B, L, _ = x_conv.shape
        proj = x_proj(x_conv)
        delta, B_proj, C_proj = torch.split(proj, [1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(dt_proj(delta))
        A = -torch.exp(A_log)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = delta.unsqueeze(-1) * B_proj.unsqueeze(2) * x_conv.unsqueeze(-1)
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x_conv.device)
        ys = []
        iterator = range(L - 1, -1, -1) if reverse else range(L)
        
        for i in iterator:
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y_i = (h * C_proj[:, i].unsqueeze(1)).sum(dim=-1)
            ys.append(y_i)
        
        if reverse:
            ys = ys[::-1]
            
        y = torch.stack(ys, dim=1)
        y = y + x_conv * D
        return y

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_in)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        y = self._scan(x_conv, self.dt_proj, self.x_proj, self.A_log, self.D)
        if self.bidirectional:
            y_b = self._scan(x_conv, self.dt_proj_b, self.x_proj_b, self.A_log_b, self.D_b, reverse=True)
            y = (y + y_b) * 0.5
            
        y = self.norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out

class MambaAttention(nn.Module):
    def __init__(self, in_dim, which_conv=None):
        super(MambaAttention, self).__init__()
        self.chanel_in = in_dim
        # Check if spectral norm is likely needed (based on which_conv)
        use_sn = (which_conv is not None and 'SN' in which_conv.__name__)
        self.mamba = MambaBlock(in_dim, d_state=8, expand=1, bidirectional=True, use_sn=use_sn)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, x_len=None, **kwargs):
        B, C, W, H = x.size()
        x_flat = x.view(B, C, W * H).transpose(1, 2)
        out_flat = self.mamba(x_flat)
        out = out_flat.transpose(1, 2).view(B, C, W, H)
        out = self.gamma * out + x
        return out
