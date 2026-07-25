import numpy as np
import torch
from torch import nn

from lib.datasets import WriterBalancedBatchSampler
from networks.fusion import StyleContentMamba
from networks.loss import KLloss, supervised_contrastive_style_loss
from networks.model import EMA
from networks.module import StyleBackbone, StyleEncoder


def test_style_encoder_is_compact_and_does_not_replace_parameters_in_forward():
    backbone = StyleBackbone(init='none').eval()
    encoder = StyleEncoder(
        init='none', num_style_tokens=8,
        backbone_channels=(64, 128, 256), n_class=80,
    ).eval()
    images = torch.full((2, 1, 64, 192), -1.0)
    images[..., :128] = torch.randn(2, 1, 64, 128)
    lengths = torch.tensor([128, 128])
    parameter_ids = {id(parameter) for parameter in encoder.parameters()}

    with torch.no_grad():
        styles = encoder(images, lengths, backbone)
        shorter_batch_styles = encoder(images[..., :160], lengths, backbone)

    assert styles.shape == (2, 8, 32)
    torch.testing.assert_close(styles, shorter_batch_styles, atol=1e-3, rtol=1e-3)
    assert {id(parameter) for parameter in encoder.parameters()} == parameter_ids


def test_bidirectional_fusion_does_not_observe_right_padding():
    torch.manual_seed(11)
    fusion = StyleContentMamba(
        d_model=16, style_dim=8, d_state=4, d_conv=3,
        expand=1, vocab_size=20,
    ).eval()
    content = torch.randn(2, 8, 16)
    changed_padding = content.clone()
    changed_padding[0, 3:] = torch.randn_like(changed_padding[0, 3:]) * 100
    changed_padding[1, 5:] = torch.randn_like(changed_padding[1, 5:]) * 100
    style = torch.randn(2, 4, 8)
    lengths = torch.tensor([3, 5])
    character_ids = torch.randint(0, 20, (2, 8))

    with torch.no_grad():
        original = fusion(content, style, character_ids, lengths)
        changed = fusion(changed_padding, style, character_ids, lengths)

    for row, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(
            original[row, :length], changed[row, :length], atol=1e-6, rtol=1e-5
        )


def test_writer_batches_losses_and_ema_buffers():
    writer_ids = np.repeat(np.arange(8), 3)
    sampler = WriterBalancedBatchSampler(
        writer_ids, batch_size=8, samples_per_writer=2, seed=3
    )
    indices = next(iter(sampler))
    batch_writer_ids = writer_ids[indices]
    assert np.all(np.unique(batch_writer_ids, return_counts=True)[1] == 2)

    styles = torch.randn(8, 8, 32, requires_grad=True)
    style_loss = supervised_contrastive_style_loss(
        styles, torch.tensor(batch_writer_ids)
    )
    style_loss.backward()
    assert torch.isfinite(style_loss)

    mu = torch.randn(2, 8, 32)
    logvar = torch.randn_like(mu).clamp(-3, 2)
    expected_kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).mean()
    torch.testing.assert_close(KLloss(mu, logvar), expected_kl)

    current = nn.BatchNorm1d(3)
    averaged = nn.BatchNorm1d(3)
    averaged.load_state_dict(current.state_dict())
    with torch.no_grad():
        current.weight.add_(2)
        current.running_mean.add_(4)
        current.num_batches_tracked.add_(7)
    tracker = EMA(0.5)
    tracker.step = 100
    tracker.step_ema(averaged, current)

    torch.testing.assert_close(averaged.running_mean, current.running_mean)
    assert torch.equal(averaged.num_batches_tracked, current.num_batches_tracked)
