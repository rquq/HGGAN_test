import numpy as np
import torch
from torch import nn

from networks.fusion import StyleContentAttentionFusion
from networks.loss import GramStyleLoss, KLloss, supervised_contrastive_style_loss
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


def test_attention_fusion_does_not_observe_right_padding():
    torch.manual_seed(11)
    fusion = StyleContentAttentionFusion(
        d_model=16, style_dim=8, nhead=4, attn_dim=32,
        ffn_dim=32, max_seq_len=16, vocab_size=20,
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
        cropped = fusion(
            content[:, :5], style, character_ids[:, :5], lengths
        )

    for row, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(
            original[row, :length], changed[row, :length], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            original[row, :length], cropped[row, :length], atol=1e-6, rtol=1e-5
        )

    assert not hasattr(fusion, 'mamba')
    assert not hasattr(fusion, 'cross_attn')
    assert not hasattr(fusion, 'global_style_mod')

def test_attention_and_allograph_have_separate_style_roles():
    torch.manual_seed(17)
    fusion = StyleContentAttentionFusion(
        d_model=16, style_dim=8, nhead=4, attn_dim=32,
        ffn_dim=32, max_seq_len=16, vocab_size=20,
    ).train()
    with torch.no_grad():
        modulation = fusion.content_context.style_mod(torch.zeros(1, 8))
    torch.testing.assert_close(
        torch.sigmoid(modulation[:, -2:]), torch.full((1, 2), 0.25)
    )
    torch.testing.assert_close(
        torch.sigmoid(fusion.local_residual_gate_logits), torch.full((16,), 0.25)
    )
    content = torch.randn(2, 7, 16, requires_grad=True)
    style = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.randint(0, 20, (2, 7))
    lengths = torch.tensor([7, 5])
    observed = {}

    def capture_context(_, inputs):
        observed['global_shape'] = tuple(inputs[1].shape)

    def capture_allograph(_, inputs):
        observed['local_shape'] = tuple(inputs[1].shape)

    context_hook = fusion.content_context.register_forward_pre_hook(capture_context)
    allograph_hook = fusion.allograph_mod.register_forward_pre_hook(capture_allograph)
    output = fusion(content, style, labels, lengths)
    context_hook.remove()
    allograph_hook.remove()
    output.square().mean().backward()

    assert observed['global_shape'] == (2, 8)
    assert observed['local_shape'] == (2, 3, 16)
    assert style.grad[:, 0].abs().sum() > 0
    assert style.grad[:, 1:].abs().sum() > 0
    assert content.grad.abs().sum() > 0


def test_writer_batches_losses_and_ema_buffers():
    batch_writer_ids = np.repeat(np.arange(4), 2)
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


def test_gram_style_loss_is_relative_and_scale_stable():
    gram_loss = GramStyleLoss()
    target = torch.randn(2, 32, 8, 12)

    torch.testing.assert_close(
        gram_loss(target, target), torch.tensor(0.0), atol=1e-7, rtol=0
    )
    moderate = gram_loss(target * 1.5, target)
    amplified = gram_loss(target * 15.0, target * 10.0)

    assert torch.isfinite(moderate)
    torch.testing.assert_close(moderate, amplified, atol=1e-6, rtol=1e-5)


def test_recognizer_cudnn_rnn_backward_in_train_mode():
    if not torch.cuda.is_available():
        return

    from networks.module import Recognizer

    recognizer = Recognizer(
        resolution=8, max_dim=32, in_channel=1, n_class=20,
        rnn_depth=1, bidirectional=True, norm='bn', init='none', dropout=0.0,
    ).cuda()
    recognizer.requires_grad_(False)
    recognizer.eval()
    recognizer.rnn_ctc.train()

    assert recognizer.rnn_ctc.lstm.training
    assert all(
        not module.training for module in recognizer.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )

    images = torch.randn(2, 1, 64, 128, device='cuda', requires_grad=True)
    image_lengths = torch.tensor([96, 128], device='cuda')
    log_probs = recognizer(images, image_lengths, return_log_probs=True)
    targets = torch.tensor([[1, 2, 0], [3, 4, 5]], device='cuda')
    target_lengths = torch.tensor([2, 3], device='cuda')
    input_lengths = torch.full(
        (2,), log_probs.size(0), dtype=torch.long, device='cuda'
    )
    loss = torch.nn.functional.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, zero_infinity=True
    )
    loss.backward()

    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert all(parameter.grad is None for parameter in recognizer.parameters())


def test_content_adversary_covers_every_style_token():
    encoder = StyleEncoder(
        style_dim=32, in_dim=256, num_style_tokens=8,
        backbone_channels=(64, 128, 256), n_class=80,
        content_grl=1.0, init='none',
    )
    styles = torch.randn(3, 8, 32, requires_grad=True)
    logits = encoder.predict_content(styles, reverse=True)
    targets = torch.zeros_like(logits)
    targets[:, :, 1] = 1.0

    torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).backward()

    assert logits.shape == (3, 8, 80)
    assert torch.all(styles.grad.abs().sum(dim=-1) > 0)
