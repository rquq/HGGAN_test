import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P


def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            v = F.normalize(gram_schmidt(torch.matmul(u, W), vs), eps=eps)
            vs.append(v)
            u = F.normalize(gram_schmidt(torch.matmul(v, W.t()), us), eps=eps)
            us.append(u)
            if update:
                u_[i][:] = u
        svs.append(torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t())))
    return svs, us, vs


class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer(f'u{i}', torch.randn(1, num_outputs))
            self.register_buffer(f'sv{i}', torch.ones(1))

    @property
    def u(self):
        return [getattr(self, f'u{i}') for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self, f'sv{i}') for i in range(self.num_svs)]

    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_itrs):
            svs, _, _ = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNDepthwiseConv2d(nn.Conv2d):
    """Spectral norm per depthwise channel filter (block-diagonal operator)."""

    def __init__(self, channels, kernel_size, padding=0, bias=True, eps=1e-12):
        super().__init__(channels, channels, kernel_size, padding=padding,
                         groups=channels, bias=bias)
        self.eps = eps

    def W_(self):
        norm = self.weight.flatten(1).norm(dim=1).amax().clamp_min(self.eps)
        return self.weight / norm

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


def _mask_valid_width(x, valid_lens, mask_cache=None):
    """Zero out right-padded columns beyond each sample's valid width."""
    if valid_lens is None:
        return x
    key = (id(valid_lens), x.size(-1), x.device)
    if mask_cache is not None and key in mask_cache:
        invalid = mask_cache[key][1]
    else:
        lengths = valid_lens.to(device=x.device, dtype=torch.long)
        invalid = torch.arange(x.size(-1), device=x.device)[None, :] >= lengths[:, None]
        if mask_cache is not None:
            # Keep valid_lens alive so its tensor id cannot be reused inside this cache.
            mask_cache[key] = (valid_lens, invalid)
    return x.masked_fill(invalid[:, None, None, :], 0.0)


def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    scale = torch.rsqrt(var + eps)
    if gain is not None:
        scale = scale * gain
    shift = mean * scale
    if bias is not None:
        shift = shift - bias
    return x * scale - shift


def _masked_batch_stats(x, valid_lens):
    """Compute batch mean/var strictly over non-padded word columns."""
    lengths = valid_lens.to(device=x.device, dtype=torch.long).clamp(min=0, max=x.size(-1))
    mask = (torch.arange(x.size(-1), device=x.device)[None, :] < lengths[:, None]).to(x.dtype)
    mask = mask[:, None, None, :]
    count = (lengths.sum() * x.size(-2)).clamp_min(1).to(x.dtype)
    mean = (x * mask).sum(dim=(0, 2, 3)) / count
    var = ((x - mean.view(1, -1, 1, 1)) * mask).square().sum(dim=(0, 2, 3)) / count
    return mean, var, count


def _update_running_stats(running_mean, running_var, mean, var, count, momentum):
    unbiased_var = var * (count / (count - 1).clamp_min(1))
    with torch.no_grad():
        running_mean.lerp_(mean.detach(), momentum)
        running_var.lerp_(unbiased_var.detach(), momentum)


class ccbn(nn.Module):
    """Style-conditional BatchNorm with valid-width statistic masking."""

    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y, valid_lens=None):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        if self.training and valid_lens is not None:
            mean, var, count = _masked_batch_stats(x, valid_lens)
            _update_running_stats(self.stored_mean, self.stored_var, mean, var, count, self.momentum)
            return fused_bn(x, mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1), gain, bias, self.eps)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                           self.training, self.momentum, self.eps)
        return out * gain + bias

    def extra_repr(self):
        return f'out: {self.output_size}, in: {self.input_size}'


class bn(nn.Module):
    """Unconditional BatchNorm with valid-width statistic masking."""

    def __init__(self, output_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.output_size = output_size
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y=None, valid_lens=None):
        if self.training and valid_lens is not None:
            mean, var, count = _masked_batch_stats(x, valid_lens)
            _update_running_stats(self.stored_mean, self.stored_var, mean, var, count, self.momentum)
            return fused_bn(
                x, mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1),
                self.gain.view(1, -1, 1, 1), self.bias.view(1, -1, 1, 1), self.eps,
            )
        return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                            self.bias, self.training, self.momentum, self.eps)


def _make_haar_2d_kernel():
    """Orthonormal 2x2 Haar DWT filter bank: (LL, LH, HL, HH)."""
    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
    hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
    return torch.stack((ll, lh, hl, hh), dim=0).unsqueeze(1)


def haar_dwt2d(x, haar_kernel):
    """2D orthonormal Haar DWT with right/bottom zero-padding for odd dimensions."""
    b, c, h, w = x.shape
    pad_h, pad_w = h & 1, w & 1
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
    weight = haar_kernel.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
    y = F.conv2d(x, weight, stride=2, groups=c)
    return y.view(b, c, 4, y.size(-2), y.size(-1))


def haar_iwt2d(subbands, haar_kernel, out_hw):
    """Inverse 2D orthonormal Haar DWT cropped to exact (H, W)."""
    b, c, _, hw, ww = subbands.shape
    weight = haar_kernel.to(device=subbands.device, dtype=subbands.dtype).repeat(c, 1, 1, 1)
    x = F.conv_transpose2d(subbands.reshape(b, 4 * c, hw, ww), weight, stride=2, groups=c)
    return x[:, :, :out_hw[0], :out_hw[1]]


class StyleWaveletStrokeMixer(nn.Module):
    """Single-level style-modulated Haar wavelet stroke mixer."""

    def __init__(self, channels, style_dim,
                 which_conv=SNConv2d, which_depthwise=SNDepthwiseConv2d,
                 which_linear=SNLinear, residual_init=0.1):
        super().__init__()
        self.channels = int(channels)
        self.style_dim = int(style_dim)
        self.residual_init = float(residual_init)

        self.register_buffer('haar_kernel', _make_haar_2d_kernel())
        self.wave_conv = which_depthwise(4 * channels, kernel_size=3, padding=1)
        self.style_band_gate = which_linear(style_dim, 4 * channels)
        self.out_proj = which_conv(channels, channels, kernel_size=1, padding=0)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), self.residual_init))

    def reset_stability_parameters(self):
        with torch.no_grad():
            self.residual_scale.fill_(self.residual_init)
            if self.style_band_gate.bias is not None:
                self.style_band_gate.bias.zero_()

    def forward(self, x, y, valid_lens=None, mask_cache=None):
        x_masked = _mask_valid_width(x, valid_lens, mask_cache)
        b, c, h, w = x_masked.shape
        if h < 2 or w < 2:
            return x_masked

        subbands = haar_dwt2d(x_masked, self.haar_kernel)
        next_lens = torch.div(valid_lens + 1, 2, rounding_mode='floor') if valid_lens is not None else None
        hw, ww = subbands.shape[-2:]
        flat = _mask_valid_width(subbands.reshape(b, 4 * c, hw, ww), next_lens)
        conv_out = _mask_valid_width(self.wave_conv(flat), next_lens)

        # Bound sub-band gain to (0.5, 1.5) over (LL, LH, HL, HH).
        band_scale = (1.0 + 0.5 * torch.tanh(self.style_band_gate(y))).view(b, 4 * c, 1, 1)
        mod_out = (conv_out * band_scale).view(b, c, 4, hw, ww)
        wave_feat = _mask_valid_width(haar_iwt2d(mod_out, self.haar_kernel, (h, w)), valid_lens, mask_cache)
        mixed = self.out_proj(wave_feat)
        return _mask_valid_width(x_masked + self.residual_scale * mixed, valid_lens, mask_cache)


class StyleWaveGBlock(nn.Module):
    """WaveStroke Generator block: BigGAN CBN residual + 1-level StyleWaveletStrokeMixer."""

    def __init__(self, in_channels, out_channels, style_dim,
                 which_conv=SNConv2d, which_bn=ccbn,
                 which_depthwise=SNDepthwiseConv2d, which_linear=SNLinear,
                 activation=nn.ReLU(inplace=False), upsample=None,
                 wave_residual_init=0.1):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.bn1 = which_bn(in_channels)
        self.conv1 = which_conv(in_channels, out_channels)
        self.bn2 = which_bn(out_channels)
        self.wave_mixer = StyleWaveletStrokeMixer(
            channels=out_channels,
            style_dim=style_dim,
            which_conv=which_conv,
            which_depthwise=which_depthwise,
            which_linear=which_linear,
            residual_init=wave_residual_init,
        )
        self.conv2 = which_conv(out_channels, out_channels)
        self.learnable_sc = in_channels != out_channels or upsample is not None
        if self.learnable_sc:
            self.conv_sc = which_conv(in_channels, out_channels, kernel_size=1, padding=0)

    def reset_stability_parameters(self):
        self.wave_mixer.reset_stability_parameters()

    def forward(self, x, y, x_lens=None, out_x_lens=None):
        mask_cache = {}
        h = _mask_valid_width(self.activation(self.bn1(x, y, valid_lens=x_lens)), x_lens, mask_cache)
        x = _mask_valid_width(x, x_lens, mask_cache)
        if self.upsample is not None:
            input_width = x.size(-1)
            h = self.upsample(h)
            x = self.upsample(x)
            if out_x_lens is None and x_lens is not None:
                out_x_lens = torch.div(x_lens * x.size(-1), input_width, rounding_mode='floor')
        valid_lens = out_x_lens if out_x_lens is not None else x_lens
        h = _mask_valid_width(h, valid_lens, mask_cache)
        x = _mask_valid_width(x, valid_lens, mask_cache)

        h = self.activation(self.bn2(self.conv1(h), y, valid_lens=valid_lens))
        h = self.wave_mixer(h, y, valid_lens=valid_lens, mask_cache=mask_cache)
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return _mask_valid_width(h + x, valid_lens, mask_cache)


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                 preactivation=False, activation=None, downsample=None):
        super().__init__()
        hidden_channels = out_channels if wide else in_channels
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
        self.conv1 = which_conv(in_channels, hidden_channels)
        self.conv2 = which_conv(hidden_channels, out_channels)
        self.learnable_sc = (in_channels != out_channels) or (downsample is not None)
        if self.learnable_sc:
            self.conv_sc = which_conv(in_channels, out_channels, kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x, x_len=None, **kwargs):
        mask_cache = {}
        x = _mask_valid_width(x, x_len, mask_cache)
        h = F.relu(x) if self.preactivation else x
        h = _mask_valid_width(self.conv1(h), x_len, mask_cache)
        h = _mask_valid_width(self.conv2(self.activation(h)), x_len, mask_cache)
        if self.downsample:
            h = self.downsample(h)
        out_lens = torch.div(x_len, 2, rounding_mode='floor') if self.downsample and x_len is not None else x_len
        return _mask_valid_width(h + self.shortcut(x), out_lens, mask_cache)
