''' Layers
    This file contains various layers for the BigGAN models.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P


# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
    def forward(self, input):
        return input


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
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
    """Kernel-matrix SN for independent one-filter depthwise groups.

    The grouped kernel matrix is block diagonal: its largest singular value
    is the largest filter L2 norm, not the norm of a dense stack of filters.
    Like ordinary convolution SN, this is not the full spatial operator norm.
    """

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


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features,  bias=True,
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
        # Comparing positions already handles negative/oversized lengths exactly
        # like clamping to [0, width], without another tensor operation.
        invalid = torch.arange(x.size(-1), device=x.device)[None, :] >= lengths[:, None]
        if mask_cache is not None:
            # Keep the source tensor alive so its Python ID cannot be reused.
            mask_cache[key] = (valid_lens, invalid)
    return x.masked_fill(invalid[:, None, None, :], 0.0)


# Self Attention module from self-attention gan
class SelfAttention(nn.Module):
    """ Self attention Layer"""
    INF_VALUE = 1e8
    def __init__(self, in_dim, which_conv=SNConv2d):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Sequential(which_conv(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.key_conv = nn.Sequential(which_conv(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.value_conv = which_conv(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_len=None, **kwargs):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # Accept the generator's historical plural spelling as well as D's
        # x_len. Padding must not participate in attention's softmax denominator.
        if x_len is None:
            x_len = kwargs.get('x_lens')
        m_batchsize, C, height, width = x.size()
        spatial_mask = None
        if x_len is not None:
            lengths = x_len.to(device=x.device, dtype=torch.long).clamp(0, width)
            width_mask = torch.arange(width, device=x.device)[None, :] < lengths[:, None]
            spatial_mask = width_mask[:, None, :].expand(-1, height, -1).reshape(m_batchsize, -1)
            x = x.masked_fill(~width_mask[:, None, None, :], 0.0)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        if spatial_mask is not None:
            energy = energy.masked_fill(
                ~spatial_mask[:, None, :], torch.finfo(energy.dtype).min
            )
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        if spatial_mask is not None:
            out = out.masked_fill(~width_mask[:, None, None, :], 0.0)
        return out


Attention = SelfAttention

# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift


def _masked_batch_stats(x, valid_lens):
    """Batch statistics over real word columns, excluding right padding."""
    lengths = valid_lens.to(device=x.device, dtype=torch.long).clamp(
        min=0, max=x.size(-1)
    )
    mask = (
        torch.arange(x.size(-1), device=x.device)[None, :]
        < lengths[:, None]
    ).to(dtype=x.dtype)[:, None, None, :]
    count = (lengths.sum() * x.size(-2)).clamp_min(1).to(dtype=x.dtype)
    mean = (x * mask).sum(dim=(0, 2, 3)) / count
    centered = (x - mean.view(1, -1, 1, 1)) * mask
    var = centered.square().sum(dim=(0, 2, 3)) / count
    return mean, var, count


def _update_running_stats(running_mean, running_var, mean, var, count, momentum):
    # Match BatchNorm's biased variance for normalization and unbiased variance
    # for the running estimate.
    unbiased_var = var * (count / (count - 1).clamp_min(1))
    with torch.no_grad():
        running_mean.lerp_(mean.detach(), momentum)
        running_var.lerp_(unbiased_var.detach(), momentum)


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = (m2 - m ** 2)
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats
class myBN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super(myBN, self).__init__()
        # momentum for updating running stats
        self.momentum = momentum
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Register buffers
        self.register_buffer('stored_mean', torch.zeros(num_channels))
        self.register_buffer('stored_var', torch.ones(num_channels))
        self.register_buffer('accumulation_counter', torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
            # If accumulating standing stats, increment them
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            # If not accumulating standing stats, take running averages
            else:
                self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
                self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
            return out
        # If not in training mode, use the stored statistics
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            # If using standing stats, divide them by the accumulation counter
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
    # If number of channels specified in norm_style:
    if 'ch' in norm_style:
        ch = int(norm_style.split('_')[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    # If number of groups specified in norm style
    elif 'grp' in norm_style:
        groups = int(norm_style.split('_')[-1])
    # If neither, default to groups = 16
    else:
        groups = 16
    return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False, norm_style='bn', ):
        super(ccbn, self).__init__()
        if cross_replica:
            raise ValueError('cross_replica=True is unsupported; use local normalization')
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # Norm style?
        self.norm_style = norm_style

        if self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ['bn', 'in']:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y, valid_lens=None):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        # If using my batchnorm
        if self.mybn:
            return self.bn(x, gain=gain, bias=bias)
        else:
            if self.norm_style == 'bn':
                if self.training and valid_lens is not None:
                    mean, var, count = _masked_batch_stats(x, valid_lens)
                    _update_running_stats(
                        self.stored_mean, self.stored_var, mean, var,
                        count, 0.1,
                    )
                    return fused_bn(
                        x, mean.view(1, -1, 1, 1),
                        var.view(1, -1, 1, 1), gain, bias, self.eps,
                    )
                out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                                   self.training, 0.1, self.eps)
            elif self.norm_style == 'in':
                out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                                      self.training, 0.1, self.eps)
            elif self.norm_style == 'gn':
                out = groupnorm(x, self.norm_style)
            elif self.norm_style == 'nonorm':
                out = x
            return out * gain + bias

    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s += ' cross_replica={cross_replica}'
        return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
    def __init__(self, output_size, eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False):
        super(bn, self).__init__()
        if cross_replica:
            raise ValueError('cross_replica=True is unsupported; use local normalization')
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn

        if mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        # Register buffers if not using myBN
        else:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y=None, valid_lens=None):
        if self.mybn:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            return self.bn(x, gain=gain, bias=bias)
        else:
            if self.training and valid_lens is not None:
                mean, var, count = _masked_batch_stats(x, valid_lens)
                _update_running_stats(
                    self.stored_mean, self.stored_var, mean, var,
                    count, self.momentum,
                )
                return fused_bn(
                    x, mean.view(1, -1, 1, 1),
                    var.view(1, -1, 1, 1),
                    self.gain.view(1, -1, 1, 1),
                    self.bias.view(1, -1, 1, 1), self.eps,
                )
            return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                                self.bias, self.training, self.momentum, self.eps)


# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must
# be preselected)
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv1=nn.Conv2d, which_conv2=nn.Conv2d, which_bn=bn, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv1, self.which_conv2, self.which_bn = which_conv1, which_conv2, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv1(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv2(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv1(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    _mask_valid_width = staticmethod(_mask_valid_width)

    def forward(self, x, y, x_lens=None, out_x_lens=None, **kwargs):
        mask_cache = {}
        h = self.activation(self.bn1(x, y, valid_lens=x_lens))
        h = _mask_valid_width(h, x_lens, mask_cache)
        x = _mask_valid_width(x, x_lens, mask_cache)
        if self.upsample:
            input_width = x.size(-1)
            h = self.upsample(h)
            x = self.upsample(x)
            if out_x_lens is None and x_lens is not None:
                out_x_lens = torch.div(
                    x_lens * x.size(-1), input_width, rounding_mode='floor'
                )
        valid_lens = out_x_lens if out_x_lens is not None else x_lens
        h = _mask_valid_width(h, valid_lens, mask_cache)
        x = _mask_valid_width(x, valid_lens, mask_cache)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y, valid_lens=valid_lens))
        h = _mask_valid_width(h, valid_lens, mask_cache)
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return _mask_valid_width(h + x, valid_lens, mask_cache)


def _make_haar_2d_kernel():
    """Orthonormal 2x2 Haar DWT filter bank: (LL, LH, HL, HH)."""
    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
    hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
    return torch.stack((ll, lh, hl, hh), dim=0).unsqueeze(1)  # (4, 1, 2, 2)


def haar_dwt2d(x, haar_kernel):
    """2D orthonormal Haar DWT supporting odd spatial dimensions via right/bottom zero pad."""
    b, c, h, w = x.shape
    pad_h = h & 1
    pad_w = w & 1
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
    weight = haar_kernel.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
    y = F.conv2d(x, weight, stride=2, groups=c)
    return y.view(b, c, 4, y.size(-2), y.size(-1))


def haar_iwt2d(subbands, haar_kernel, out_hw):
    """Inverse 2D orthonormal Haar DWT restoring exact target (H, W)."""
    b, c, _, hw, ww = subbands.shape
    weight = haar_kernel.to(device=subbands.device, dtype=subbands.dtype).repeat(c, 1, 1, 1)
    x = F.conv_transpose2d(subbands.reshape(b, 4 * c, hw, ww), weight, stride=2, groups=c)
    return x[:, :, :out_hw[0], :out_hw[1]]


class StyleWaveletStrokeMixer(nn.Module):
    """Style-modulated multi-level Haar wavelet + gated CNN stroke mixer.

    Combines multi-scale Haar wavelet convolutions (WTConv, ECCV 2024),
    gated CNN multiplicative fusion (MambaOut, CVPR 2025), and per-stage
    style subband modulation over (LL, LH, HL, HH) so writer style directly
    steers horizontal, vertical, diagonal, and structural stroke energy while
    preserving exact valid-width padding invariance.
    """

    def __init__(self, channels, style_dim, wt_levels=2,
                 which_conv=SNConv2d, which_depthwise=SNDepthwiseConv2d,
                 which_linear=SNLinear, residual_init=0.1):
        super().__init__()
        if wt_levels < 1:
            raise ValueError('wt_levels must be at least 1')
        self.channels = int(channels)
        self.style_dim = int(style_dim)
        self.wt_levels = int(wt_levels)
        self.residual_init = float(residual_init)

        self.register_buffer('haar_kernel', _make_haar_2d_kernel())
        self.wave_in = which_conv(channels, channels, kernel_size=1, padding=0)
        self.gate_proj = which_conv(channels, channels, kernel_size=1, padding=0)
        self.wave_convs = nn.ModuleList([
            which_depthwise(4 * channels, kernel_size=3, padding=1)
            for _ in range(self.wt_levels)
        ])
        self.style_band_gates = nn.ModuleList([
            which_linear(style_dim, 4 * channels)
            for _ in range(self.wt_levels)
        ])
        self.out_proj = which_conv(channels, channels, kernel_size=1, padding=0)
        self.residual_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), self.residual_init)
        )

    def reset_stability_parameters(self):
        with torch.no_grad():
            self.residual_scale.fill_(self.residual_init)
            for gate in self.style_band_gates:
                if gate.bias is not None:
                    gate.bias.zero_()

    def forward(self, x, y, valid_lens=None, mask_cache=None):
        x_masked = _mask_valid_width(x, valid_lens, mask_cache)
        gate = F.silu(self.gate_proj(x_masked))
        curr_ll = _mask_valid_width(self.wave_in(x_masked), valid_lens, mask_cache)
        curr_lens = valid_lens

        ll_outputs = []
        high_outputs = []
        shapes = []
        lens_stack = []

        for level in range(self.wt_levels):
            if curr_ll.size(-2) < 2 or curr_ll.size(-1) < 2:
                break
            shapes.append((curr_ll.size(-2), curr_ll.size(-1)))
            lens_stack.append(curr_lens)
            subbands = haar_dwt2d(curr_ll, self.haar_kernel)
            next_lens = (
                torch.div(curr_lens + 1, 2, rounding_mode='floor')
                if curr_lens is not None else None
            )
            b, c, _, hw, ww = subbands.shape
            flat = _mask_valid_width(subbands.reshape(b, 4 * c, hw, ww), next_lens)
            conv_out = _mask_valid_width(self.wave_convs[level](flat), next_lens)
            # Bounded per-subband style gain in (0.5, 1.5) over (LL, LH, HL, HH).
            band_scale = (
                1.0 + 0.5 * torch.tanh(self.style_band_gates[level](y))
            ).view(b, 4 * c, 1, 1)
            mod_out = (conv_out * band_scale).view(b, c, 4, hw, ww)
            ll_outputs.append(mod_out[:, :, 0])
            high_outputs.append(mod_out[:, :, 1:])
            curr_ll = _mask_valid_width(subbands[:, :, 0], next_lens)
            curr_lens = next_lens

        reconstructed = None
        for level in reversed(range(len(ll_outputs))):
            ll = (
                ll_outputs[level]
                if reconstructed is None else (ll_outputs[level] + reconstructed)
            )
            combined = torch.cat((ll.unsqueeze(2), high_outputs[level]), dim=2)
            reconstructed = haar_iwt2d(combined, self.haar_kernel, shapes[level])
            reconstructed = _mask_valid_width(reconstructed, lens_stack[level])

        wave_feat = curr_ll if reconstructed is None else reconstructed
        mixed = self.out_proj(
            _mask_valid_width(wave_feat * gate, valid_lens, mask_cache)
        )
        return _mask_valid_width(x_masked + self.residual_scale * mixed, valid_lens, mask_cache)


class StyleWaveGBlock(GBlock):
    """WaveStroke Generator block: BigGAN CBN residual + StyleWaveletStrokeMixer."""

    def __init__(self, in_channels, out_channels, style_dim,
                 which_conv1=nn.Conv2d, which_conv2=nn.Conv2d, which_bn=bn,
                 which_depthwise=SNDepthwiseConv2d, which_linear=SNLinear,
                 activation=None, upsample=None, wt_levels=2,
                 wave_residual_init=0.1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            which_conv1=which_conv1,
            which_conv2=which_conv2,
            which_bn=which_bn,
            activation=activation,
            upsample=upsample,
        )
        self.wave_mixer = StyleWaveletStrokeMixer(
            channels=out_channels,
            style_dim=style_dim,
            wt_levels=wt_levels,
            which_conv=which_conv2,
            which_depthwise=which_depthwise,
            which_linear=which_linear,
            residual_init=wave_residual_init,
        )

    def reset_stability_parameters(self):
        self.wave_mixer.reset_stability_parameters()

    def forward(self, x, y, x_lens=None, out_x_lens=None, **kwargs):
        mask_cache = {}
        h = self.activation(self.bn1(x, y, valid_lens=x_lens))
        h = _mask_valid_width(h, x_lens, mask_cache)
        x = _mask_valid_width(x, x_lens, mask_cache)
        if self.upsample:
            input_width = x.size(-1)
            h = self.upsample(h)
            x = self.upsample(x)
            if out_x_lens is None and x_lens is not None:
                out_x_lens = torch.div(
                    x_lens * x.size(-1), input_width, rounding_mode='floor'
                )
        valid_lens = out_x_lens if out_x_lens is not None else x_lens
        h = _mask_valid_width(h, valid_lens, mask_cache)
        x = _mask_valid_width(x, valid_lens, mask_cache)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y, valid_lens=valid_lens))
        h = self.wave_mixer(h, y, valid_lens=valid_lens, mask_cache=mask_cache)
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return _mask_valid_width(h + x, valid_lens, mask_cache)


# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                 preactivation=False, activation=None, downsample=None, ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

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
        if self.preactivation:
            # Must use an out-of-place ReLU activation to preserve shortcut connection state
            h = F.relu(x)
        else:
            h = x
        h = _mask_valid_width(self.conv1(h), x_len, mask_cache)
        h = _mask_valid_width(self.conv2(self.activation(h)), x_len, mask_cache)
        if self.downsample:
            h = self.downsample(h)
        out_lengths = torch.div(x_len, 2, rounding_mode='floor') if self.downsample and x_len is not None else x_len
        return _mask_valid_width(h + self.shortcut(x), out_lengths, mask_cache)
