import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils import _len2mask


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


def global_style_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, num_style_tokens: int
) -> torch.Tensor:
    """Apply the existing mean L1 to global style, retaining its all-token scale."""
    if (
        isinstance(num_style_tokens, bool)
        or not isinstance(num_style_tokens, int)
        or num_style_tokens <= 0
    ):
        raise ValueError('num_style_tokens must be a positive integer')
    if (
        pred.ndim != 3
        or target.ndim != 3
        or pred.shape != target.shape
        or pred.size(0) < 1
        or pred.size(1) not in (1, num_style_tokens)
        or pred.size(2) < 1
    ):
        raise ValueError(
            'pred and target must have matching shape (B, 1 or N, D)'
        )
    return F.l1_loss(pred[:, :1], target[:, :1]) / num_style_tokens


def global_content_adversarial_loss(
    logits: torch.Tensor, labels: torch.Tensor, num_style_tokens: int
) -> torch.Tensor:
    """Apply the existing bag-of-character BCE to global logits at all-token scale."""
    if (
        isinstance(num_style_tokens, bool)
        or not isinstance(num_style_tokens, int)
        or num_style_tokens <= 0
    ):
        raise ValueError('num_style_tokens must be a positive integer')
    if (
        logits.ndim != 3
        or logits.size(0) < 1
        or logits.size(1) != 1
        or logits.size(2) < 1
    ):
        raise ValueError('logits must have shape (B, 1, C)')
    if labels.ndim != 2 or labels.size(0) != logits.size(0):
        raise ValueError('labels must have shape (B, L) matching logits batch size')

    content_targets = torch.zeros_like(logits)
    content_indices = labels[:, None, :].expand(-1, 1, -1)
    content_indices = content_indices.clamp(0, logits.size(2) - 1)
    content_targets.scatter_(2, content_indices, 1.0)
    content_targets[..., 0] = 0.0
    positive_fraction = content_targets.mean().detach().clamp(1e-4, 0.5)
    positive_weight = ((1.0 - positive_fraction) / positive_fraction).clamp(max=10.0)
    return F.binary_cross_entropy_with_logits(
        logits, content_targets, pos_weight=positive_weight
    ) / num_style_tokens
