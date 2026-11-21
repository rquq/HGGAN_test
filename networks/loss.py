import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils import _len2mask


class SlicedWassersteinLoss(nn.Module):
    """Equal-word, multiscale matching of frozen local feature distributions."""

    def __init__(self, channels, projections=32, samples_per_word=64, seed=123456):
        super().__init__()
        self.samples_per_word = int(samples_per_word)
        self.num_scales = len(channels)
        self.register_buffer('projection_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('projection_seed', torch.tensor(seed, dtype=torch.long))
        rng = torch.Generator(device='cpu').manual_seed(seed)
        for index, channel in enumerate(channels):
            directions = F.normalize(torch.randn(channel, projections, generator=rng), dim=0)
            self.register_buffer(f'directions_{index}', directions)

    def _quantiles(self, features, lengths, directions):
        lengths = torch.as_tensor(lengths, device=features.device).long()
        values = []
        for row, width in zip(features, lengths.detach().cpu().tolist()):
            descriptors = row[:, :, :width].flatten(1).transpose(0, 1)
            projected = descriptors @ directions.to(descriptors)
            ordered = projected.sort(dim=0).values
            quantiles = F.interpolate(
                ordered.T.unsqueeze(0), size=self.samples_per_word,
                mode='linear', align_corners=False,
            )[0].T
            values.append(quantiles)
        return torch.cat(values, dim=0)

    def forward(self, real_features, fake_features, real_lengths, fake_lengths):
        rng = torch.Generator(device='cpu').manual_seed(
            int(self.projection_seed.item()) + int(self.projection_step.item())
        )
        if self.training:
            self.projection_step.add_(1)
        terms = []
        for index, (real, fake, rw, fw) in enumerate(zip(
                real_features, fake_features, real_lengths, fake_lengths)):
            template = getattr(self, f'directions_{index}')
            directions = (
                F.normalize(torch.randn(template.shape, generator=rng), dim=0).to(template)
                if self.training else template
            )
            target = self._quantiles(real.detach(), rw, directions)
            generated = self._quantiles(fake, fw, directions)
            scale = target.square().mean(0).sqrt().clamp_min(1e-2).detach()
            terms.append(((target.sort(0).values - generated.sort(0).values) / scale).square().mean())
        return torch.stack(terms).mean()


def r1_reg(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return 0.5 * grad_dout.pow(2).view(batch_size, -1).sum(1).mean(0)


def recn_l1_loss(img1, img2, img_lens):
    mask = _len2mask(img_lens, img1.size(-1)).to(img1.device)
    diff_img = (img1 - img2) * mask.view(mask.size(0), 1, 1, mask.size(1))
    return diff_img.abs().sum() / (diff_img.size(1) * diff_img.size(2) * torch.clamp(img_lens.sum(), min=1))


def KLloss(mu, logvar):
    loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    return loss.flatten(1).mean(dim=1).mean()
