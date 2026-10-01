import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _len2mask(length, max_len=None, dtype=torch.float32):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if length.numel() == 0:
        return torch.empty((0, 0), device=length.device, dtype=dtype or torch.float32)
    max_len = max_len or int(length.max().item())
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg




def recn_l1_loss(img1, img2, img_lens):
    mask = _len2mask(img_lens, img1.size(-1)).to(img1.device)
    diff_img = (img1 - img2) * mask.view(mask.size(0), 1, 1, mask.size(1))
    loss = diff_img.abs().sum() / (diff_img.size(1) * diff_img.size(2) * torch.clamp(img_lens.sum(), min=1))
    return loss




def KLloss(mu, logvar):
    # Mean, rather than sum, makes this weight independent of token count and
    # style dimension and prevents KL from abruptly dominating the generator.
    loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    return loss.flatten(1).mean(dim=1).mean()


class SpectralDistributionLoss(nn.Module):
    """Match stroke-frequency statistics without aligning different words.

    Only valid-width square patches are used. Their spatial means are removed,
    and FFT magnitudes are averaged across patches; phase/position is excluded
    so small alignment differences do not dominate the reconstruction signal.
    This magnitude-only term is not a substitute for spatial or contextual
    supervision.
    """

    def __init__(self, patch_size=None, width_samples=4, patch_fraction=0.5):
        super().__init__()
        self.patch_size = None if patch_size is None else int(patch_size)
        self.width_samples = int(width_samples)
        self.patch_fraction = float(patch_fraction)
        if (self.patch_size is not None and self.patch_size < 2) or self.width_samples < 1:
            raise ValueError('patch_size must be None or >= 2 and width_samples >= 1')
        if not 0.0 < self.patch_fraction <= 1.0:
            raise ValueError('patch_fraction must be in (0, 1]')

    def _spectrum(self, image, patch_size):
        height, width = image.shape[-2:]
        n_rows = min(math.ceil(height / patch_size), height - patch_size + 1)
        n_cols = min(self.width_samples, width - patch_size + 1)
        row_starts = torch.linspace(0, height - patch_size, n_rows).round().int().tolist()
        col_starts = torch.linspace(0, width - patch_size, n_cols).round().int().tolist()
        patches = torch.cat([
            image[..., row:row + patch_size, col:col + patch_size]
            for row in row_starts for col in col_starts
        ], dim=0).float()
        patches = patches - patches.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.rfft2(patches, norm='ortho').abs().log1p()
        return spectrum.mean(dim=(0, 1))

    def forward(self, target, generated, target_lengths, generated_lengths):
        if target.ndim != 4 or generated.ndim != 4 or target.shape[:2] != generated.shape[:2]:
            raise ValueError('target and generated must have matching batch/channel dimensions')
        if target_lengths.numel() != target.size(0) or generated_lengths.numel() != generated.size(0):
            raise ValueError('valid-width lengths must match the batch size')

        target_widths = target_lengths.detach().clamp(1, target.size(-1)).tolist()
        generated_widths = generated_lengths.detach().clamp(1, generated.size(-1)).tolist()
        losses = []
        for index, (target_width, generated_width) in enumerate(zip(target_widths, generated_widths)):
            target_width, generated_width = int(target_width), int(generated_width)
            # A fixed pixel window changes its relative scale between 32px and
            # 64px inputs. By default, use half of the smaller image height;
            # an explicit patch_size remains available for diagnostics.
            patch_size = min(
                self.patch_size or max(2, round(min(target.size(-2), generated.size(-2)) * self.patch_fraction)),
                target.size(-2), generated.size(-2),
                target_width, generated_width,
            )
            if patch_size < 2:
                continue
            target_image = target[index:index + 1, :, :, :target_width]
            generated_image = generated[index:index + 1, :, :, :generated_width]
            losses.append(F.l1_loss(
                self._spectrum(target_image, patch_size),
                self._spectrum(generated_image, patch_size),
            ))
        return torch.stack(losses).mean() if losses else generated.sum() * 0.0


##############################################################################
# Contextual loss
##############################################################################
class CXLoss(nn.Module):
    def __init__(self, sigma=0.5, b=1.0, similarity="cosine", max_tokens=256):
        super(CXLoss, self).__init__()
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
        ):
            raise ValueError('max_tokens must be a positive int or None')
        self.similarity = similarity
        self.sigma = sigma
        self.b = b
        self.max_tokens = max_tokens

    def _pool_valid_crop(self, feature):
        """Downsample a single already-cropped feature map when over budget."""
        height, width = feature.shape[-2:]
        if self.max_tokens is None or height * width <= self.max_tokens:
            return feature
        scale = math.sqrt(self.max_tokens / float(height * width))
        out_height = max(1, int(height * scale))
        out_width = max(1, int(width * scale))
        while out_height * out_width > self.max_tokens:
            if out_height >= out_width and out_height > 1:
                out_height -= 1
            elif out_width > 1:
                out_width -= 1
            else:
                break
        return F.adaptive_avg_pool2d(feature, (out_height, out_width))

    def center_by_T(self, featureI, featureT):
        # Center each sample independently; a batch-wide mean couples writers.
        meanT = featureT.mean(dim=(2, 3), keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = torch.clamp(features.norm(p=2, dim=1, keepdim=True), min=1e-8)
        features = features.div(norms)
        return features

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        # [0] means get the value, torch min will return the index as well
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def _forward_impl(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        N, C, H_T, W_T = featureT.shape
        _, _, H_I, W_I = featureI.shape

        featI_flat = featureI.reshape(N, C, H_I * W_I)
        featT_flat = featureT.reshape(N, C, H_T * W_T)

        # batched matrix multiplication: (N, P_T, C) x (N, C, P_I) -> (N, P_T, P_I)
        dist = torch.bmm(featT_flat.transpose(1, 2), featI_flat).clamp(-1.0, 1.0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist, axis=1)

        CX = self.calc_CX(relative_dist, axis=1)

        # Take max over spatial dimensions of Inference feature map (dim=2, which is P_I)
        CX_max = CX.max(dim=2)[0]
        CX_mean = torch.mean(CX_max, dim=1)
        CX_loss = torch.mean(-torch.log(CX_mean + 1e-5))
        return CX_loss

    def forward(self, featureT, featureI, target_lengths=None, input_lengths=None):
        if target_lengths is None and input_lengths is None and self.max_tokens is None:
            return self._forward_impl(featureT, featureI)
        if target_lengths is None and input_lengths is None:
            return torch.stack([
                self._forward_impl(
                    self._pool_valid_crop(featureT[index:index + 1]),
                    self._pool_valid_crop(featureI[index:index + 1]),
                )
                for index in range(featureT.size(0))
            ]).mean()
        if target_lengths is None or input_lengths is None:
            raise ValueError('target_lengths and input_lengths must be supplied together')
        if len(target_lengths) != featureT.size(0) or len(input_lengths) != featureI.size(0):
            raise ValueError('contextual-loss lengths must match their batch sizes')

        losses = []
        target_widths = target_lengths.detach().clamp(
            1, featureT.size(-1)
        ).tolist()
        input_widths = input_lengths.detach().clamp(
            1, featureI.size(-1)
        ).tolist()
        for index, (target_width, input_width) in enumerate(zip(
            target_widths, input_widths
        )):
            target_width = int(target_width)
            input_width = int(input_width)
            target_crop = featureT[index:index + 1, :, :, :target_width]
            input_crop = featureI[index:index + 1, :, :, :input_width]
            # Crop padding away before pooling so invalid columns never leak
            # into the valid feature representation.
            losses.append(self._forward_impl(
                self._pool_valid_crop(target_crop),
                self._pool_valid_crop(input_crop),
            ))
        return torch.stack(losses).mean()



##############################################################################
# Gram style loss
##############################################################################
class GramStyleLoss(nn.Module):
    def __init__(self):
        super(GramStyleLoss, self).__init__()
        self.gram = GramMatrix()

    def __call__(self, input_feat, target_feat, feat_len=None):
        input_gram = self.gram(input_feat, feat_len)
        target_gram = self.gram(target_feat, feat_len)
        # A raw Gram MSE scales with feature energy and channel count, so its
        # magnitude can vary by orders of magnitude between backbone stages.
        # Normalize by the detached energy of both Gram matrices to make the
        # loss dimensionless while preserving gradients through input_gram.
        error = (input_gram - target_gram).square().mean(dim=(1, 2))
        energy = 0.5 * (
            input_gram.square().mean(dim=(1, 2))
            + target_gram.square().mean(dim=(1, 2))
        )
        return (error / energy.detach().clamp_min(1e-6)).mean()


class GramMatrix(nn.Module):
    def forward(self, input, feat_len=None):
        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=False)
            if input.device.type == 'cuda'
            else (
                torch.amp.autocast(input.device.type, enabled=False)
                if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast')
                else torch.autograd.set_grad_enabled(torch.is_grad_enabled())
            )
        )
        with autocast_ctx:
            input = input.float()
            a, b, c, d = input.size()

            if feat_len is not None:
                # mask for varying lengths
                mask = _len2mask(feat_len, d).view(a, 1, 1, d)
                input = input * mask
                denom = (c * torch.clamp(feat_len, min=1)).view(a, 1, 1) * b
            else:
                denom = float(b * c * d)

            features = input.view(a, b, c * d)
            G = torch.bmm(features, features.transpose(1, 2))

            return G / denom


def _global_style(style):
    return style if style.dim() == 2 else style[:, 0]




def supervised_contrastive_style_loss(styles, writer_ids, temperature=0.1):
    """Pull same-writer global style tokens together and repel other writers."""
    features = F.normalize(_global_style(styles), dim=-1)
    writer_ids = writer_ids.view(-1)
    count = features.size(0)
    if count < 2:
        return features.sum() * 0.0

    self_mask = torch.eye(count, dtype=torch.bool, device=features.device)
    positive_mask = writer_ids[:, None].eq(writer_ids[None, :]) & ~self_mask
    valid_anchors = positive_mask.any(dim=1)
    if not valid_anchors.any():
        return features.sum() * 0.0

    logits = torch.matmul(features, features.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    mean_positive_log_prob = (
        (log_prob * positive_mask.to(log_prob.dtype)).sum(dim=1)
        / positive_mask.sum(dim=1).clamp_min(1)
    )
    return -mean_positive_log_prob[valid_anchors].mean()
