# Dev architecture backup

This directory preserves the experimental architecture from dev commit
`6955ee7f67a915c03f41eb05d0946483903fa3d4` before the dev tree was reset to
the `random_crop_recog` implementation.

- `BigGAN_networks_star.py` contains `StarFilter`, `StarBSSP`, and the
  Star-conditioned generator.
- `BigGAN_layers_star.py` contains `StarCCBN`.
- `fusion_non_mamba.py` contains the bidirectional-GRU, non-Mamba
  `StyleContentFusion`.
- `fusion_legacy_backup.py` and `fusion_legacy_backup_2.py` preserve the two
  additional fusion experiments that were present in the original dev tree.

These files are archival and are not imported by the active training code.
Any future reintroduction should use zero-initialized residual adapters and
padding masks instead of replacing the active dense conditioning paths.
