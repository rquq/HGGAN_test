import torch

from networks.BigGAN_networks import Discriminator, PatchDiscriminator
from networks.utils import sample_adaptive_patches


def test_main_global_and_stroke_critics_are_backward_safe():
    torch.manual_seed(79)
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
    word_lengths = torch.tensor([128, 64])
    label_lengths = torch.tensor([4, 2])
    global_logits = global_critic(words, word_lengths, label_lengths)

    patches, crop_counts = sample_adaptive_patches(words, word_lengths)
    patch_logits = stroke_critic(patches)
    (global_logits.mean() + patch_logits.mean()).backward()

    assert global_logits.shape == (2, 1)
    assert patch_logits.shape == (8, 1, 4, 4)
    assert crop_counts.tolist() == [4, 4]
    assert words.grad is not None and torch.isfinite(words.grad).all()
    assert global_critic.width_context.residual_scale.grad is not None


def test_stroke_critic_remains_lighter_than_the_global_critic():
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
