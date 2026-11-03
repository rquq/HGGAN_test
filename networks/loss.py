import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils import _len2mask


class SlicedWassersteinLoss(nn.Module):
    """Equal-word, multiscale matching of frozen local feature distributions.

    Adapted from Heitz et al., CVPR 2021. Unlike paired feature reconstruction,
    sorting discards spatial correspondence. Only real training features define
    the detached scale; generated magnitude discrepancies remain penalized.
    """

    def __init__(self, channels, projections=32, samples_per_word=64, seed=123456):
        super().__init__()
        if not channels or min(*channels, projections, samples_per_word) < 1:
            raise ValueError('Channels, projections and samples must be positive')
        self.samples_per_word = int(samples_per_word)
        self.num_scales = len(channels)
        self.register_buffer('projection_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('projection_seed', torch.tensor(seed, dtype=torch.long))
        rng = torch.Generator(device='cpu').manual_seed(seed)
        for index, channel in enumerate(channels):
            directions = F.normalize(torch.randn(channel, projections, generator=rng), dim=0)
            self.register_buffer(f'directions_{index}', directions)

    def _quantiles(self, features, lengths, directions):
        if features.ndim != 4 or features.size(1) != directions.size(0):
            raise ValueError('Expected BCHW features matching the projection channels')
        lengths = torch.as_tensor(lengths, device=features.device).long()
        if (lengths.shape != (features.size(0),) or torch.any(lengths < 1)
                or torch.any(lengths > features.size(-1))):
            raise ValueError('One valid width is required per feature image')
        values = []
        for row, width in zip(features, lengths.detach().cpu().tolist()):
            descriptors = row[:, :, :width].flatten(1).transpose(0, 1)
            projected = descriptors @ directions.to(descriptors)
            ordered = projected.sort(dim=0).values
            # Quantile resampling, not spatial subsampling: location permutation
            # invariance also holds for short images with fewer than 64 entries.
            quantiles = F.interpolate(
                ordered.T.unsqueeze(0), size=self.samples_per_word,
                mode='linear', align_corners=False,
            )[0].T
            values.append(quantiles)
        return torch.cat(values, dim=0)

    def forward(self, real_features, fake_features, real_lengths, fake_lengths):
        if not all(len(items) == self.num_scales for items in
                   (real_features, fake_features, real_lengths, fake_lengths)):
            raise ValueError('Feature and length lists must match configured scales')
        rng = torch.Generator(device='cpu').manual_seed(
            int(self.projection_seed.item()) + int(self.projection_step.item())
        )
        if self.training:
            self.projection_step.add_(1)
        terms = []
        for index, (real, fake, rw, fw) in enumerate(zip(
                real_features, fake_features, real_lengths, fake_lengths)):
            template = getattr(self, f'directions_{index}')
            # Refresh training directions to avoid a permanent low-rank blind
            # subspace. Local RNG/counter are reproducible and checkpointed.
            directions = (F.normalize(torch.randn(template.shape, generator=rng), dim=0).to(template)
                          if self.training else template)
            target = self._quantiles(real.detach(), rw, directions)
            generated = self._quantiles(fake, fw, directions)
            if target.size(0) != generated.size(0):
                raise ValueError('Real/fake batches must contain the same number of words')
            scale = target.square().mean(0).sqrt().clamp_min(1e-2).detach()
            terms.append(((target.sort(0).values - generated.sort(0).values) / scale).square().mean())
        return torch.stack(terms).mean()


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
