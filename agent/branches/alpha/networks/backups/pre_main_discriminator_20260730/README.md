# Pre-main-discriminator dev backup

This source-only snapshot was taken from clean dev HEAD
`495f10919cc83a31c8c59610698798de22208da2` before transplanting the
current main discriminator subsystem.

- `BigGAN_networks.py` contains the original dev global D and PatchD.
- `model.py` preserves the original dense-patch critic integration.
- `utils.py` preserves `extract_all_patches`.
- `configs/` contains the original discriminator configuration.

The active dev generator, style encoder, fusion, and non-adversarial losses
remain unchanged. In particular, `networks/fusion.py` is still the original
dev fusion implementation so it can be benchmarked against main's new fusion.
