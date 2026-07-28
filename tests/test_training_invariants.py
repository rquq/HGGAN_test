import numpy as np
import torch
from torch import nn
from munch import Munch

from networks.BigGAN_networks import Generator, Discriminator, PatchDiscriminator
from networks.fusion import StyleContentAttentionFusion
from networks.rapidnet import ConditionedRapidBlock
from networks.loss import CXLoss, GramStyleLoss, KLloss, supervised_contrastive_style_loss
from networks.model import BaseModel, EMA
from networks.module import StyleBackbone, StyleEncoder
from networks.utils import (
    get_scheduler, restore_scheduler_state, sample_adaptive_patches,
)


def test_style_encoder_is_compact_and_does_not_replace_parameters_in_forward():
    torch.manual_seed(7)
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
    torch.testing.assert_close(styles, shorter_batch_styles, atol=2e-3, rtol=1e-3)
    assert {id(parameter) for parameter in encoder.parameters()} == parameter_ids
    local_styles = torch.nn.functional.normalize(styles[:, 1:], dim=-1)
    local_similarity = local_styles @ local_styles.transpose(1, 2)
    off_diagonal = ~torch.eye(
        local_styles.size(1), dtype=torch.bool
    ).unsqueeze(0)
    assert local_similarity.masked_select(off_diagonal).mean() < 0.9


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

        observed['local_style'] = inputs[1].detach()
    context_hook = fusion.content_context.register_forward_pre_hook(capture_context)
    allograph_hook = fusion.allograph_mod.register_forward_pre_hook(capture_allograph)
    output = fusion(content, style, labels, lengths)
    context_hook.remove()
    allograph_hook.remove()
    output.square().mean().backward()

    assert observed['global_shape'] == (2, 8)
    assert observed['local_shape'] == (2, 3, 16)
    projected_local = torch.nn.functional.normalize(
        observed['local_style'], dim=-1
    )
    projected_similarity = projected_local @ projected_local.transpose(1, 2)
    off_diagonal = ~torch.eye(3, dtype=torch.bool).unsqueeze(0)
    assert projected_similarity.masked_select(off_diagonal).mean() < 0.9
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


def test_checkpoint_resume_uses_epoch_metadata_at_a_loader_boundary():
    assert BaseModel.resume_position(3, 26114, 26115) == (4, 0, 26115)
    assert BaseModel.resume_position(3, 5, 10) == (3, 6, 6)


def test_discriminator_scheduler_preserves_its_configured_half_rate():
    g_parameter = nn.Parameter(torch.zeros(()))
    d_parameter = nn.Parameter(torch.zeros(()))
    g_optimizer = torch.optim.Adam([g_parameter], lr=2e-4)
    d_optimizer = torch.optim.Adam([d_parameter], lr=1e-4)
    options = Munch(
        lr=2e-4, lr_policy='linear', start_decay_epoch=12, n_epochs_decay=24,
    )

    get_scheduler(g_optimizer, options, base_lr=2e-4)
    d_scheduler = get_scheduler(d_optimizer, options, base_lr=1e-4)
    assert d_optimizer.param_groups[0]['lr'] == 1e-4

    stale_state = d_scheduler.state_dict()
    stale_state['base_lrs'] = [2e-4]
    stale_state['_last_lr'] = [2e-4]
    stale_state['last_epoch'] = 2
    d_optimizer.param_groups[0]['lr'] = 2e-4
    restore_scheduler_state(
        d_scheduler,
        d_optimizer,
        stale_state,
        base_lr=1e-4,
        completed_epochs=3,
    )

    assert d_optimizer.param_groups[0]['lr'] == 1e-4
    assert d_scheduler.base_lrs == [1e-4]
    assert d_scheduler.last_epoch == 3


def test_contextual_loss_ignores_padding_and_centers_each_sample():
    torch.manual_seed(23)
    contextual = CXLoss()
    target = torch.randn(2, 8, 3, 5)
    inferred = torch.randn(2, 8, 3, 6)
    target_lengths = torch.tensor([5, 5])
    input_lengths = torch.tensor([6, 6])
    baseline = contextual(target, inferred)

    padded_target = torch.randn(2, 8, 3, 9) * 100
    padded_inferred = torch.randn(2, 8, 3, 11) * 100
    padded_target[..., :5] = target
    padded_inferred[..., :6] = inferred
    padded = contextual(
        padded_target,
        padded_inferred,
        target_lengths=target_lengths,
        input_lengths=input_lengths,
    )

    torch.testing.assert_close(baseline, padded, atol=1e-6, rtol=1e-5)


def test_fusion_residuals_remain_bounded_under_extreme_style_conditioning():
    torch.manual_seed(31)
    fusion = StyleContentAttentionFusion(
        d_model=16, style_dim=8, nhead=4, attn_dim=32,
        ffn_dim=32, max_seq_len=16, vocab_size=20,
    ).eval()
    with torch.no_grad():
        fusion.content_context.style_mod.weight.fill_(100.0)
        fusion.content_context.style_mod.bias.fill_(100.0)
        fusion.content_context.ffn_in.weight.fill_(50.0)
        fusion.content_context.ffn_out.weight.fill_(50.0)
    content = torch.randn(2, 8, 16)
    style = torch.randn(2, 4, 8) * 100
    labels = torch.randint(0, 20, (2, 8))
    lengths = torch.tensor([8, 5])

    with torch.no_grad():
        output = fusion(content, style, labels, lengths)
    valid = (
        torch.arange(content.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    )
    output_rms = output[valid].square().mean().sqrt()

    assert torch.isfinite(output).all()
    assert output_rms < 3.0

def test_adaptive_patch_sampling_is_bounded_stratified_and_differentiable():
    torch.manual_seed(41)
    images = torch.randn(3, 1, 64, 320, requires_grad=True)
    lengths = torch.tensor([32, 160, 320])

    patches, crop_counts = sample_adaptive_patches(images, lengths)

    assert crop_counts.tolist() == [4, 5, 8]
    assert patches.shape == (17, 1, 32, 32)
    patches.square().mean().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert torch.all(images.grad.abs().sum(dim=(1, 2, 3)) > 0)


def test_rapid_generator_preserves_conditioning_geometry_and_gradients():
    torch.manual_seed(43)
    generator = Generator(
        G_ch=16,
        style_dim=32,
        embed_dim=120,
        bottom_width=4,
        bottom_height=4,
        resolution=64,
        G_attn='0',
        n_class=80,
        init='none',
        input_nc=1,
    ).train()
    style = torch.randn(2, 8, 32, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    lengths = torch.tensor([4, 2])

    output = generator(style, labels, lengths)
    output.square().mean().backward()

    assert output.shape == (2, 1, 64, 128)
    assert all(
        isinstance(blocklist[0], ConditionedRapidBlock)
        for blocklist in generator.blocks
    )
    assert all(
        blocklist[0].upsample.keywords['mode'] == 'bilinear'
        and blocklist[0].upsample.keywords['align_corners'] is False
        and blocklist[0].spatial_transition.groups
            == blocklist[0].spatial_transition.out_channels
        for blocklist in generator.blocks
    )
    assert style.grad is not None and torch.isfinite(style.grad).all()
    assert generator.fusion_gate_logits.grad is not None
    assert all(
        blocklist[0].mldc.mixer_scale.grad is not None
        for blocklist in generator.blocks
    )


def test_rapid_upsampling_does_not_preserve_hard_macro_cells():
    class PassCondition(nn.Module):
        def forward(self, features, _condition):
            return features

    block = ConditionedRapidBlock(
        in_channels=1,
        out_channels=1,
        which_conv=lambda in_channels, out_channels, **kwargs: nn.Conv2d(
            in_channels, out_channels, bias=False, **kwargs
        ),
        which_bn=lambda _channels: PassCondition(),
        activation=nn.Identity(),
        upsample=lambda features: torch.nn.functional.interpolate(
            features, scale_factor=2, mode='bilinear', align_corners=False
        ),
    )
    with torch.no_grad():
        block.project.weight.fill_(1.0)
        block.shortcut.weight.fill_(1.0)
        block.spatial_transition.weight.zero_()
        block.spatial_transition.weight[:, :, 1, 1] = 1.0
        block.mldc.mixer_scale.zero_()
        block.mldc.ffn_scale.zero_()

    seed_cells = torch.tensor(
        [[[[0.0, 1.0], [0.0, 1.0]]]], requires_grad=True
    )
    output = block(seed_cells, torch.zeros(1, 1))
    output.sum().backward()

    assert output.shape == (1, 1, 4, 4)
    assert torch.unique(output).numel() > torch.unique(seed_cells).numel()
    assert 0.0 < output[0, 0, 0, 1] < output[0, 0, 0, -1]
    assert seed_cells.grad is not None and torch.isfinite(seed_cells.grad).all()


def test_global_and_stroke_critics_are_complementary_and_backward_safe():
    torch.manual_seed(47)
    global_critic = Discriminator(
        D_ch=16,
        resolution=64,
        D_attn='0',
        input_nc=1,
        width_context=True,
        width_heads=4,
        init='none',
    ).train()
    stroke_critic = PatchDiscriminator(
        D_ch=16,
        D_max_ch=96,
        D_layers=3,
        input_nc=1,
        init='none',
    ).train()

    words = torch.randn(2, 1, 64, 128, requires_grad=True)
    word_lengths = torch.tensor([128, 96])
    label_lengths = torch.tensor([4, 3])
    global_logits = global_critic(words, word_lengths, label_lengths)

    patches = torch.randn(7, 1, 32, 32, requires_grad=True)
    patch_logits = stroke_critic(patches)
    (global_logits.mean() + patch_logits.mean()).backward()

    assert global_logits.shape == (2, 1)
    assert patch_logits.shape == (7, 1, 4, 4)
    assert words.grad is not None and torch.isfinite(words.grad).all()
    assert patches.grad is not None and torch.isfinite(patches.grad).all()
    assert global_critic.width_context.residual_scale.grad is not None


def test_full_stroke_critic_is_smaller_than_the_global_critic():
    global_critic = Discriminator(
        D_ch=64,
        resolution=64,
        D_attn='0',
        input_nc=1,
        width_context=True,
        width_heads=4,
        init='none',
    )
    stroke_critic = PatchDiscriminator(
        D_ch=32,
        D_max_ch=192,
        D_layers=3,
        input_nc=1,
        init='none',
    )
    count = lambda module: sum(
        parameter.numel() for parameter in module.parameters()
    )

    assert count(stroke_critic) < 700_000
    assert count(stroke_critic) < count(global_critic)

