### Task 2: Simplify E without changing its global path

**Files:**
- Modify: `networks/module.py`, `StyleEncoder` constructor/forward, currently around lines 301–669.
- Modify: `configs/gan_iam_32.yml`, `configs/gan_iam_64.yml`, `configs/gan_iam_local.yml` in ALPHA only.
- Modify: ignored `tests/test_compact_style_context.py` and `tests/test_luna_review.py`.
- Create: ignored `tests/test_reference_local_evidence.py`.

**Interfaces:**
- Consume Task 1's centered tensor convention; E centers its own local output, while G does not center it again.
- Preserve all existing `forward(img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False, ret_backbone_feat=False, backbone_features=None)` arguments; append `global_only=False` for internal objective calls, not a YAML mode.
- Full forward still returns `(B,N,D)` or VAE `(sample, mu, logvar)` of that shape; optional features keep their existing tuple order. `global_only=True` returns the same contracts with `(B,1,D)` and skips local retrieval/projection entirely.
- Produce private `_encode_global_style(self, feat: torch.Tensor, feat_len: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]` returning global mean/logvar `(B,1,D)`; full and global-only forward call this same seam.
- Produce private `_build_local_attention_inputs(self, selected_feats: list[torch.Tensor], map_lengths: list[torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]` returning keys, values and padding mask for the selected multiscale maps.
- Create constructor-owned `local_projection = nn.Linear(in_dim, style_dim, bias=False)` and bias-free MHA. Keep orthogonal query initialization, query/key affine-free normalization, existing projections/feature scales and four-row pooling.

- [ ] **Write failing assertions:** `test_position_only_changes_keys`, `test_uniform_values_cannot_create_local_style`, `test_weak_visual_deviation_is_not_renormalized`, `test_visual_reference_changes_centered_codes`, `test_global_only_skips_local_modules_and_preserves_features`, `test_zero_and_single_local_slot_contract`. Test both heights, mixed valid widths, padded feature changes and finite evidence gradients; perturb around a nondegenerate channel vector when checking weak contrast.
- [ ] **Run:** `/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_reference_local_evidence.py`. Expected: missing private seam/`global_only` or failures demonstrating the old query-prior behavior.
- [ ] **Factor the unchanged global calculation** into `_encode_global_style`; preserve width context, valid mean, global MLP/norm, `mu`, `logvar` clamp `[-14,4]` and global sampling. Global-only optional features must return masked selected original maps and original backbone sequence, without computing local projections.
- [ ] **Implement local retrieval:** normalize visual tokens once, add 2D PE only to keys, values equal those visual tokens minus their per-map valid-spatial mean, mask padded values, bias-free cross-attention and dedicated bias-free projection, then subtract local-slot mean. Do not append global 1D context tokens to this local stream.
- [ ] **Remove obsolete mechanisms:** reliability MLP/descriptors, sigmoid attention strength, output query residual, local output normalization, shared global `mu` on local appearance and E's checkpoint initialization shim. Remove constructor options/aliases `local_attention_residual_init`, `local_attention_gate_init`, `local_evidence_gate_init`, `local_evidence_gate_hidden`; remove the three currently present YAML gate entries. Keep unrelated option aliases/settings.
- [ ] **Update legacy assertions:** replace the gate-initialization test with absence/strict-load assertions. Local-only gradient tests target queries/attention/projection/input evidence; do not require local-only gradients through the now-global-only sequence context. Retain its separate gradient tests.
- [ ] **Run each focused file:** `/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_reference_local_evidence.py`, `/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_compact_style_context.py`, `/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_luna_review.py`. Expected: all pass, including no NaNs for empty local slots and zero output for one centered local slot. These boundary cases concern E and the prior helper; do not loosen G's existing minimum-token validation.
- [ ] **Review boundary:** root checks E's initialization, feature-length geometry, optional return contracts and that B/W/R definitions are untouched. Do not commit.

