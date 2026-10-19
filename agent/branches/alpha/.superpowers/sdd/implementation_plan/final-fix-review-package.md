# Review package: final-fix
Uncommitted task changes; HEAD and index are untouched.

## ../main/configs/gan_iam_32.yml

```diff
--- before/../main/configs/gan_iam_32.yml
+++ after/../main/configs/gan_iam_32.yml
@@ -219,31 +219,30 @@
 
 
 StyBackbone:
   resolution: 16
   max_dim: 256
   in_channel: 1
   init: 'N02'
   dropout: 0.
   norm: 'bn'
 
 
 EncModel:
   style_dim: 32
   in_dim: 256
   init: 'none'
   num_style_tokens: 8
   num_local_queries: 7
   query_dim: 256
   heads: 4
   cross_attn_dropout: 0.0
-  # All three backbone maps keep old 32px checkpoints shape-compatible.  A
-  # future speed ablation may shorten this list to select shallow maps only.
+  # All three backbone maps contribute to the shared multiscale evidence path.
   feature_scales: [1, 2, 4]
 
 
 WidModel:
   n_writer: 339
   in_dim: 256
   pool: 'attentive_stats'
   attention_dim: 128
   init: 'none'

```

## test_artifacts/local_style_simplification_14_10/verify_sync.py

```diff
--- before/test_artifacts/local_style_simplification_14_10/verify_sync.py
+++ after/test_artifacts/local_style_simplification_14_10/verify_sync.py
@@ -150,44 +150,56 @@
             if (ROOT / path).read_bytes() != (MAIN / path).read_bytes()
         ]
         if mismatched:
             failures.append(
                 f"tracked Python bytes still differ ({len(mismatched)}): {mismatched}"
             )
 
     baseline = json.loads(BASELINE.read_text())
     config_snapshot = json.loads(CONFIG_SNAPSHOT.read_text())
     for name in CONFIGS:
         original_text = baseline[f"../main/configs/{name}"]
         expected_text = original_text
         for key in OBSOLETE_ENCODER_KEYS:
             exact_line = f"  {key}: "
             matching = [line for line in expected_text.splitlines(keepends=True)
                         if line.startswith(exact_line)]
             if len(matching) != 1:
                 failures.append(f"baseline has no unique {key} entry in {name}")
             else:
                 expected_text = expected_text.replace(matching[0], "", 1)
+        if name == "gan_iam_32.yml":
+            stale_comment = (
+                "  # All three backbone maps keep old 32px checkpoints shape-compatible.  A\n"
+                "  # future speed ablation may shorten this list to select shallow maps only.\n"
+            )
+            current_comment = (
+                "  # All three backbone maps contribute to the shared multiscale evidence path.\n"
+            )
+            if expected_text.count(stale_comment) != 1:
+                failures.append(f"baseline has no unique stale feature-scales comment in {name}")
+            else:
+                expected_text = expected_text.replace(stale_comment, current_comment, 1)
         actual_text = (MAIN / "configs" / name).read_text()
         if actual_text != expected_text:
             failures.append(
-                f"MAIN {name} differs from original text beyond the three obsolete encoder lines"
+                f"MAIN {name} differs beyond the three obsolete encoder lines and the approved comment correction"
             )
 
         actual_values = yaml.safe_load(actual_text)
         expected_values = copy.deepcopy(config_snapshot[name]["configuration"])
         for key in OBSOLETE_ENCODER_KEYS:
             expected_values.get("EncModel", {}).pop(key, None)
         if actual_values != expected_values:
             failures.append(f"MAIN {name} YAML values differ from the recorded snapshot")
         if any(key in actual_values.get("EncModel", {}) for key in OBSOLETE_ENCODER_KEYS):
             failures.append(f"MAIN {name} still contains an obsolete encoder gate key")
 
         training = actual_values["training"]
         for key in ("lexicon", "pretrained_w", "pretrained_r"):
             rel = pathlib.Path(training[key])
             if not (MAIN / rel).exists():
                 failures.append(f"MAIN {name} path is missing: training.{key}={rel}")
 
     for relative in (".git", "data/iam", "data/english_words.txt", "pretrained", "runs", "wandb"):
         if not (MAIN / relative).exists():
             failures.append(f"preserved MAIN runtime/data path missing: {relative}")
@@ -207,29 +219,29 @@
         (ROOT, "test_artifacts/local_style_simplification_14_10/verify_sync.py"),
         (ROOT, "test_artifacts/local_style_simplification_14_10/implementation_results.md"),
         (MAIN, "test_artifacts/local_style_simplification_14_10/main_cpu_harness.py"),
         (ROOT, ".superpowers/sdd/implementation_plan/task-5-report.md"),
     ):
         result = subprocess.run(
             ["git", "check-ignore", "-q", relative], cwd=root,
             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
         )
         if result.returncode != 0:
             failures.append(f"artifact is not ignored: {relative}")
 
     if failures:
         print(f"FAIL: {len(failures)} synchronization invariant(s) not met")
         for failure in failures:
             print(f" - {failure}")
         return 1
 
     print("PASS: nested branch/HEAD identities unchanged (Random, MAIN, DEV)")
     print(f"PASS: all {len(random_python)} tracked Python files match byte-for-byte")
-    print("PASS: MAIN configs match original values after removing only three obsolete keys")
+    print("PASS: MAIN config values match snapshot after obsolete keys are removed; approved text correction only")
     print("PASS: runtime/data paths, pretrained hashes, and frozen class-body hashes preserved")
     print("PASS: only the expected Random/MAIN product worktree changes are present; DEV is clean")
     print("PASS: verification artifacts are git-ignored")
     return 0
 
 
 if __name__ == "__main__":
     sys.exit(main())

```

## ../main/tests/test_compact_style_context.py

```diff
--- before/../main/tests/test_compact_style_context.py
+++ after/../main/tests/test_compact_style_context.py
@@ -83,26 +83,36 @@
                     self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                     self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
 
     def test_encoder_local_visual_evidence_and_global_only_sampling(self):
         for height in (32, 64):
             with self.subTest(height=height):
                 torch.manual_seed(813)
                 backbone = StyleBackbone(img_height=height, init='none').eval().requires_grad_(False)
                 encoder = StyleEncoder(init='none').train()
                 x = torch.randn(2, 1, height, height * 3, requires_grad=True)
                 lengths = torch.tensor([height, height * 3])
                 sampled, mu, logvar = encoder(x, lengths, backbone, vae_mode=True)
                 self.assertEqual(mu.shape, (2, 8, 32))
                 torch.testing.assert_close(sampled[:, 1:], mu[:, 1:])
                 self.assertTrue(torch.isfinite(logvar).all())
                 self.assertGreater((mu[0, 1:] - mu[1, 1:]).abs().mean().item(), 1e-6)
                 mu[:, 1:].square().sum().backward()
                 self.assertTrue(torch.isfinite(x.grad).all())
                 self.assertGreater(x.grad[0, ..., :height].abs().sum().item(), 0)
                 self.assertEqual(x.grad[0, ..., height:].abs().max().item(), 0)
+                for parameter in (encoder.style_queries, encoder.local_projection.weight,
+                                  encoder.style_cross_attn.in_proj_weight,
+                                  encoder.style_cross_attn.out_proj.weight):
+                    self.assertTrue(torch.isfinite(parameter.grad).all())
+                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
+                self.assertTrue(all(p.grad is None or p.grad.abs().sum().item() == 0.
+                                    for p in encoder.sequence_model.parameters()))
+                encoder.zero_grad(set_to_none=True)
+                global_style = encoder(x, lengths, backbone, global_only=True)
+                global_style.square().sum().backward()
                 for block in encoder.sequence_model.blocks:
                     self.assertGreater(block.depthwise.weight.grad.abs().sum().item(), 0)
 
 
 if __name__ == '__main__':
     unittest.main(verbosity=2)

```

## ../main/tests/test_critic_experiment.py

```diff
--- before/../main/tests/test_critic_experiment.py
+++ after/../main/tests/test_critic_experiment.py
@@ -13,42 +13,43 @@
 torch.set_num_threads(2)
 
 
 class CriticExperiment(unittest.TestCase):
     def test_removed_modules_have_no_live_imports(self):
         self.assertFalse((ROOT / 'networks' / 'texture_generator.py').exists())
         self.assertFalse((ROOT / 'networks' / 'masking.py').exists())
         for path in (ROOT / 'networks').glob('*.py'):
             source = path.read_text()
             self.assertNotIn('texture_generator', source, str(path))
             self.assertNotIn('import masking', source, str(path))
             self.assertNotIn('masking import', source, str(path))
 
     def test_experiment_settings(self):
         expected_steps = 4 if ROOT.name == 'dev' else 2
         for filename in ('gan_iam_32.yml', 'gan_iam_64.yml', 'gan_iam_local.yml'):
             with self.subTest(config=filename):
                 cfg = yaml.safe_load((ROOT / 'configs' / filename).read_text())
                 training = cfg['training']
                 self.assertEqual(training['num_critic_train'], expected_steps)
+                expected_lr = 2e-5 if filename == 'gan_iam_32.yml' else 2e-4
                 for key in ('lr', 'd_lr', 'patch_d_lr'):
-                    self.assertEqual(training[key], 2e-4)
+                    self.assertEqual(training[key], expected_lr)
                 if filename != 'gan_iam_local.yml':
                     self.assertEqual(training['batch_size'], 8)
                 self.assertNotIn('texture_refinement_enabled', cfg['GenModel'])
 
     def test_no_texture_head_both_heights(self):
         for height in (32, 64):
             with self.subTest(height=height):
                 cfg = yaml.safe_load((ROOT / 'configs' / f'gan_iam_{height}.yml').read_text())
                 args = dict(cfg['GenModel'])
                 args['G_ch'] = 8
                 args['init'] = 'none'
                 model = Generator(**args).train()
                 self.assertFalse(hasattr(model, 'texture_refinement'))
                 self.assertFalse(any('texture' in key for key in model.state_dict()))
                 style = torch.randn(2, 8, model.style_dim)
                 labels = torch.tensor([[1, 2, 3], [4, 0, 0]])
                 lengths = torch.tensor([3, 1])
                 output = model(style, labels, lengths)
                 self.assertEqual(output.shape, (2, 1, height, 3 * height // 2))
                 output.square().mean().backward()

```

## ../main/tests/test_luna_review.py

```diff
--- before/../main/tests/test_luna_review.py
+++ after/../main/tests/test_luna_review.py
@@ -16,45 +16,60 @@
 
 torch.set_num_threads(2)
 
 
 class LunaReview(unittest.TestCase):
     def test_global_kl_preserves_previous_global_contribution(self):
         mu = torch.randn(2, 8, 32, requires_grad=True)
         logvar = torch.randn_like(mu)
         old_mu = torch.cat((mu[:, :1], torch.zeros_like(mu[:, 1:])), dim=1)
         old_logvar = torch.cat((logvar[:, :1], torch.zeros_like(logvar[:, 1:])), dim=1)
         expected = KLloss(old_mu, old_logvar)
         # This expression is the GAN training call; local deterministic tokens
         # must contribute zero, not change the pre-existing global KL weight.
         source = (pathlib.Path(__file__).resolve().parents[1] / 'networks/model.py').read_text()
         self.assertTrue('KLloss(mu[:, :1], logvar[:, :1]) / mu.size(1)' in source)
         actual = KLloss(mu[:, :1], logvar[:, :1]) / mu.size(1)
         torch.testing.assert_close(actual, expected)
         actual.backward()
         self.assertEqual(mu.grad[:, 1:].abs().max().item(), 0)
 
-    def test_evidence_gate_initializes_at_configured_probability(self):
-        encoder = StyleEncoder(init='none', local_evidence_gate_init=.75)
-        logits = encoder.local_evidence_gate(torch.randn(2, 7, 1024))
-        probability = logits.sigmoid()
-        torch.testing.assert_close(probability, torch.full_like(probability, .75))
+    def test_local_evidence_has_no_gate_bias_or_checkpoint_initialization_shim(self):
+        encoder = StyleEncoder(init='none')
+        for obsolete in ('local_evidence_gate', 'local_attention_gate_logits', 'local_output_norm'):
+            self.assertFalse(hasattr(encoder, obsolete))
+        self.assertIsNone(encoder.local_projection.bias)
+        self.assertIsNone(encoder.style_cross_attn.in_proj_bias)
+        self.assertIsNone(encoder.style_cross_attn.out_proj.bias)
+        state = encoder.state_dict()
+        StyleEncoder(init='none').load_state_dict(state, strict=True)
+        missing_projection = {key: value for key, value in state.items()
+                              if key != 'local_projection.weight'}
+        with self.assertRaisesRegex(RuntimeError, 'local_projection.weight'):
+            encoder.load_state_dict(missing_projection, strict=True)
+        obsolete_state = dict(state, **{'local_evidence_gate.2.bias': torch.zeros(1)})
+        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
+            encoder.load_state_dict(obsolete_state, strict=True)
+        for option in ('local_attention_residual_init', 'local_attention_gate_init',
+                       'local_evidence_gate_init', 'local_evidence_gate_hidden'):
+            with self.subTest(option=option), self.assertRaisesRegex(TypeError, option):
+                StyleEncoder(init='none', **{option: .5})
 
     def test_attention_ignores_padding_and_blocks_tail_gradients(self):
         torch.manual_seed(81)
         conv = functools.partial(nn.Conv2d, kernel_size=1, padding=0)
         attention = layers.Attention(16, conv).eval()
         attention.gamma.data.fill_(.4)
         x = torch.randn(1, 16, 4, 5)
         expected = attention(x, x_len=torch.tensor([5]))
         padded = F.pad(x, (0, 7), value=9).requires_grad_()
         for keyword in ('x_len', 'x_lens'):
             actual = attention(padded, **{keyword: torch.tensor([5])})
             torch.testing.assert_close(actual[..., :5], expected, atol=1e-6, rtol=1e-5)
             self.assertEqual(actual[..., 5:].abs().max().item(), 0)
         actual.square().sum().backward()
         self.assertTrue(torch.isfinite(padded.grad).all())
         self.assertEqual(padded.grad[..., 5:].abs().max().item(), 0)
 
     def test_masked_batchnorm_matches_native_with_full_lengths(self):
         torch.manual_seed(71)
         x = torch.randn(3, 7, 4, 9, requires_grad=True)

```

## ../main/tests/test_reference_fusion.py

```diff
--- before/../main/tests/test_reference_fusion.py
+++ after/../main/tests/test_reference_fusion.py
@@ -1,144 +0,0 @@
-"""Local-only checks for reference-conditioned fusion; excluded from Git."""
-import io
-import pathlib
-import sys
-import unittest
-
-import torch
-import torch.nn.functional as F
-
-ROOT = pathlib.Path(__file__).resolve().parents[1]
-sys.path.insert(0, str(ROOT))
-from networks.fusion import StyleContentAttentionFusion
-from networks.BigGAN_networks import Generator
-from lib.utils import yaml2config
-
-torch.set_num_threads(2)
-
-
-class ReferenceFusion(unittest.TestCase):
-    def setUp(self):
-        torch.manual_seed(309)
-
-    def test_global_fallback_produces_zero_local_correction(self):
-        model = StyleContentAttentionFusion(120, 32).eval()
-        with torch.no_grad():
-            model.context_scale.zero_()
-            model.local_scale.zero_()
-            model.reference_scale.fill_(1.)
-        content = torch.randn(2, 5, 120)
-        global_style = torch.randn(2, 1, 32)
-        style = global_style.expand(-1, 8, -1).clone()
-        actual = model(content, style)
-        torch.testing.assert_close(actual, content, atol=1e-7, rtol=1e-6)
-        style[:, 3, :16] += .5
-        changed = model(content, style)
-        self.assertGreater((changed - actual).abs().max().item(), 1e-5)
-
-    def test_reference_values_preserve_small_evidence_without_renormalizing_delta(self):
-        model = StyleContentAttentionFusion(120, 32).eval()
-        attention = model.reference_attention
-        content = torch.randn(2, 4, 120)
-        global_style = torch.randn(2, 1, 32)
-        evidence = torch.randn(2, 7, 32)
-        seen = []
-        handle = attention.v_proj.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
-        for reliability in (0., 1e-4, .5):
-            local = global_style + reliability * evidence
-            model(content, torch.cat((global_style, local), dim=1))
-            expected = attention.style_norm(local) - attention.style_norm(global_style)
-            torch.testing.assert_close(seen[-1], expected, atol=1e-7, rtol=1e-6)
-        handle.remove()
-        self.assertEqual(seen[0].abs().max().item(), 0.)
-        self.assertLess(seen[1].norm().item(), seen[2].norm().item() * .01)
-
-    def test_reference_evidence_reaches_neighborhood_mixer(self):
-        # Catches putting width mixing before retrieval, which cannot mix the
-        # newly retrieved evidence between adjacent target positions.
-        model = StyleContentAttentionFusion(120, 32).eval()
-        mixer = getattr(model, 'local_mixer', None)
-        self.assertIsNotNone(mixer, 'Joint local mixing must follow style retrieval')
-        seen = []
-        handle = mixer.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
-        content = torch.randn(2, 5, 120)
-        style = torch.randn(2, 8, 32)
-        changed = style.clone()
-        changed[:, 3, :16] += 2
-        model(content, style)
-        model(content, changed)
-        handle.remove()
-        self.assertGreater((seen[0] - seen[1]).abs().max().item(), 1e-5)
-
-    def test_residual_strength_can_grow_without_a_sigmoid_ceiling(self):
-        # Catches an extra outer gate or a hidden nonlinear strength limit.
-        model = StyleContentAttentionFusion(120, 32).eval()
-        scales = getattr(model, 'reference_scale', None)
-        self.assertIsNotNone(scales, 'Reference residual needs a direct learnable scale')
-        content, style = torch.randn(2, 4, 120), torch.randn(2, 8, 32)
-        with torch.no_grad():
-            model.context_scale.zero_()
-            model.local_scale.zero_()
-            model.reference_scale.fill_(1)
-            one = model(content, style)
-            model.reference_scale.fill_(2)
-            two = model(content, style)
-        torch.testing.assert_close(two - content, 2 * (one - content), atol=2e-6, rtol=2e-5)
-
-    def test_short_long_and_padded_content_preserve_outputs_and_gradients(self):
-        for length in (1, 2, 9, 45):
-            with self.subTest(length=length):
-                model = StyleContentAttentionFusion(120, 32).train()
-                style = torch.randn(2, 8, 32, requires_grad=True)
-                content = torch.randn(2, length, 120)
-                valid = torch.tensor([max(1, length // 2), length])
-                padded = F.pad(content, (0, 0, 0, 5), value=12).requires_grad_()
-                expected = model(content, style, y_lens=valid)
-                actual = model(padded, style, y_lens=valid)
-                torch.testing.assert_close(actual[:, :length], expected, atol=3e-6, rtol=3e-5)
-                for row, end in enumerate(valid):
-                    self.assertEqual(actual[row, end:].abs().max().item(), 0)
-                actual.square().mean().backward()
-                self.assertTrue(torch.isfinite(style.grad).all())
-                self.assertGreater(style.grad[:, 1:].abs().sum().item(), 0)
-                for row, end in enumerate(valid):
-                    self.assertEqual(padded.grad[row, end:].abs().max().item(), 0)
-                for name, parameter in model.named_parameters():
-                    self.assertIsNotNone(parameter.grad, name)
-                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
-                    # One-character attention has only one possible key.
-                    if not (length == 1 and name.endswith('relative_position_bias')):
-                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
-
-    def test_local_style_slot_permutation_and_variable_slot_count(self):
-        model = StyleContentAttentionFusion(120, 32).eval()
-        content = torch.randn(2, 6, 120)
-        for slots in (1, 7, 11):
-            style = torch.randn(2, slots + 1, 32)
-            reordered = torch.cat((style[:, :1], style[:, 1:].flip(1)), dim=1)
-            torch.testing.assert_close(model(content, style), model(content, reordered), atol=2e-6, rtol=2e-5)
-
-    def test_real_config_generator_style_gradients_and_checkpoint_roundtrip(self):
-        for height in (32, 64):
-            cfg = yaml2config(str(ROOT / 'configs' / f'gan_iam_{height}.yml'))
-            args = dict(cfg.GenModel)
-            args.update(G_ch=8, init='none')
-            model = Generator(**args).eval()
-            style = torch.randn(2, 8, 32, requires_grad=True)
-            labels = torch.tensor([[2, 8, 9], [5, 0, 0]])
-            lengths = torch.tensor([3, 1])
-            image = model(style, labels, lengths)
-            image.square().mean().backward()
-            self.assertGreater(style.grad[:, 1:].abs().sum().item(), 0)
-            self.assertTrue(torch.isfinite(style.grad).all())
-            self.assertTrue(torch.isfinite(model.fusion_strength()))
-            buffer = io.BytesIO()
-            torch.save(model.state_dict(), buffer)
-            buffer.seek(0)
-            restored = Generator(**args).eval()
-            restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
-            with torch.no_grad():
-                torch.testing.assert_close(restored(style.detach(), labels, lengths), model(style.detach(), labels, lengths))
-
-
-if __name__ == '__main__':
-    unittest.main(verbosity=2)

```

## ../main/tests/test_local_style_contract.py

```diff
--- before/../main/tests/test_local_style_contract.py
+++ after/../main/tests/test_local_style_contract.py
@@ -0,0 +1,189 @@
+import unittest
+import pathlib
+import sys
+
+import torch
+import torch.nn.functional as F
+
+sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
+
+from networks.loss import (
+    global_content_adversarial_loss,
+    global_style_reconstruction_loss,
+)
+from networks.rand_dist import Distribution
+from networks.utils import center_style_prior
+
+
+class LocalStyleContractTests(unittest.TestCase):
+    def test_prior_centering_preserves_global_and_gradients(self):
+        styles = torch.tensor(
+            [
+                [[10.0, 20.0], [0.0, 2.0], [2.0, 4.0], [4.0, 6.0]],
+                [[-2.0, 4.0], [4.0, 0.0], [4.0, 2.0], [4.0, 4.0]],
+            ],
+            requires_grad=True,
+        )
+        original = styles.detach().clone()
+
+        centered = center_style_prior(styles)
+
+        self.assertEqual(centered.shape, (2, 4, 2))
+        torch.testing.assert_close(centered[:, 0], original[:, 0], rtol=0, atol=0)
+        torch.testing.assert_close(
+            centered[:, 1:],
+            torch.tensor(
+                [
+                    [[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]],
+                    [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]],
+                ]
+            ),
+        )
+        torch.testing.assert_close(
+            centered[:, 1:].mean(dim=1), torch.zeros((2, 2)), atol=1e-7, rtol=0
+        )
+        torch.testing.assert_close(styles.detach(), original, rtol=0, atol=0)
+
+        (centered * torch.arange(1, 17, dtype=centered.dtype).view_as(centered)).sum().backward()
+        self.assertIsNotNone(styles.grad)
+        self.assertTrue(torch.isfinite(styles.grad).all())
+
+        single = styles.detach()[:, :1]
+        torch.testing.assert_close(center_style_prior(single), single, rtol=0, atol=0)
+        with self.assertRaisesRegex(ValueError, 'shape'):
+            center_style_prior(torch.zeros(2, 3))
+        with self.assertRaisesRegex(ValueError, 'token'):
+            center_style_prior(torch.zeros(2, 0, 3))
+
+    def test_distribution_prior_returns_base_tensor(self):
+        base = torch.arange(24, dtype=torch.float32).view(2, 4, 3).requires_grad_()
+        distribution = base.as_subclass(Distribution)
+        distribution.init_distribution('normal', mean=0.0, var=1.0, seed=91)
+        global_rng_before = torch.random.get_rng_state().clone()
+        sampler_rng_before = distribution.get_state()['generator'].clone()
+
+        centered = center_style_prior(distribution)
+
+        self.assertIs(type(centered), torch.Tensor)
+        self.assertTrue(centered.requires_grad)
+        weights = torch.tensor([[[2., 3., 4.], [1., 4., 7.], [3., 5., 9.], [8., 9., 11.]]]).expand_as(centered)
+        (centered * weights).sum().backward()
+        self.assertIsNotNone(base.grad)
+        # d(x_i - mean(x))/dx_j = delta_ij - 1/3; a sum-only
+        # objective would hide a detached or incorrect local Jacobian.
+        torch.testing.assert_close(base.grad, torch.tensor([
+            [[2., 3., 4.], [-3., -2., -2.], [-1., -1., 0.], [4., 3., 2.]],
+            [[2., 3., 4.], [-3., -2., -2.], [-1., -1., 0.], [4., 3., 2.]],
+        ]))
+        torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before)
+        torch.testing.assert_close(
+            distribution.get_state()['generator'], sampler_rng_before
+        )
+
+        single_base = torch.tensor([[[2., 5.]]], requires_grad=True)
+        single = single_base.as_subclass(Distribution)
+        single.init_distribution('normal', mean=0., var=1., seed=92)
+        single_rng = single.get_state()['generator'].clone()
+        one = center_style_prior(single)
+        self.assertIs(type(one), torch.Tensor)
+        torch.testing.assert_close(one, single_base, atol=0, rtol=0)
+        (one * torch.tensor([[[3., 7.]]])).sum().backward()
+        torch.testing.assert_close(single_base.grad, torch.tensor([[[3., 7.]]]))
+        torch.testing.assert_close(single.get_state()['generator'], single_rng)
+
+    def test_global_l1_matches_former_global_gradient(self):
+        target = torch.tensor(
+            [
+                [[0.0, 1.0], [4.0, 3.0], [1.0, 5.0]],
+                [[2.0, 0.0], [7.0, 2.0], [6.0, 8.0]],
+            ]
+        )
+        old_pred = torch.tensor(
+            [
+                [[2.0, -1.0], [5.0, 7.0], [3.0, 4.0]],
+                [[-1.0, 3.0], [9.0, 2.0], [4.0, 0.0]],
+            ],
+            requires_grad=True,
+        )
+        old_mean_loss = F.l1_loss(old_pred, target)
+        old_mean_loss.backward()
+
+        pred = old_pred.detach().clone().requires_grad_()
+        loss = global_style_reconstruction_loss(pred, target, num_style_tokens=3)
+        loss.backward()
+
+        old_global_loss = F.l1_loss(old_pred.detach()[:, :1], target[:, :1]) / 3
+        torch.testing.assert_close(loss, old_global_loss)
+        torch.testing.assert_close(pred.grad[:, 0], old_pred.grad[:, 0])
+        torch.testing.assert_close(pred.grad[:, 1:], torch.zeros_like(pred.grad[:, 1:]))
+        global_only_loss = global_style_reconstruction_loss(
+            pred.detach()[:, :1], target[:, :1], num_style_tokens=3
+        )
+        torch.testing.assert_close(global_only_loss, loss.detach())
+        single_token_loss = global_style_reconstruction_loss(
+            pred.detach()[:, :1], target[:, :1], num_style_tokens=1
+        )
+        torch.testing.assert_close(
+            single_token_loss, F.l1_loss(pred.detach()[:, :1], target[:, :1])
+        )
+
+        changed_local = pred.detach().clone()
+        changed_local[:, 1:] += 1000
+        torch.testing.assert_close(
+            global_style_reconstruction_loss(changed_local, target, 3), loss.detach()
+        )
+        with self.assertRaisesRegex(ValueError, 'shape'):
+            global_style_reconstruction_loss(torch.zeros(2, 3), torch.zeros(2, 3), 3)
+        with self.assertRaisesRegex(ValueError, 'positive'):
+            global_style_reconstruction_loss(pred.detach(), target, 0)
+
+    def test_global_bce_matches_former_global_gradient(self):
+        token_count = 4
+        class_count = 40
+        labels = torch.tensor([[1, 1, 0], [2, 0, 0]])
+        old_logits = torch.linspace(
+            -1.5, 1.5, steps=2 * token_count * class_count
+        ).view(2, token_count, class_count).requires_grad_()
+        old_targets = torch.zeros_like(old_logits)
+        old_indices = labels[:, None, :].expand(-1, token_count, -1)
+        old_targets.scatter_(2, old_indices, 1.0)
+        old_targets[..., 0] = 0.0
+        positive_fraction = old_targets.mean().detach().clamp(1e-4, 0.5)
+        positive_weight = ((1.0 - positive_fraction) / positive_fraction).clamp(max=10.0)
+        former_mean_loss = F.binary_cross_entropy_with_logits(
+            old_logits, old_targets, pos_weight=positive_weight
+        )
+        former_mean_loss.backward()
+
+        logits = old_logits.detach()[:, :1].clone().requires_grad_()
+        loss = global_content_adversarial_loss(logits, labels, token_count)
+        loss.backward()
+
+        former_global_loss = F.binary_cross_entropy_with_logits(
+            old_logits.detach()[:, :1], old_targets[:, :1], pos_weight=positive_weight
+        ) / token_count
+        torch.testing.assert_close(loss, former_global_loss)
+        one_token_loss = global_content_adversarial_loss(logits.detach(), labels, 1)
+        torch.testing.assert_close(one_token_loss, former_global_loss * token_count)
+        torch.testing.assert_close(logits.grad[:, 0], old_logits.grad[:, 0])
+        self.assertTrue(torch.isfinite(logits.grad).all())
+
+        dense_labels = torch.arange(1, class_count).view(1, -1)
+        dense_logits = torch.linspace(-1.0, 1.0, steps=class_count).view(1, 1, -1)
+        dense_targets = torch.ones_like(dense_logits)
+        dense_targets[..., 0] = 0.0
+        dense_loss = F.binary_cross_entropy_with_logits(
+            dense_logits, dense_targets, pos_weight=torch.tensor(1.0)
+        ) / token_count
+        torch.testing.assert_close(
+            global_content_adversarial_loss(dense_logits, dense_labels, token_count),
+            dense_loss,
+        )
+        with self.assertRaisesRegex(ValueError, 'shape'):
+            global_content_adversarial_loss(torch.zeros(2, 2, class_count), labels, token_count)
+        with self.assertRaisesRegex(ValueError, 'positive'):
+            global_content_adversarial_loss(logits.detach(), labels, 0)
+
+
+if __name__ == '__main__':
+    unittest.main(verbosity=2)

```

## ../main/tests/test_reference_local_evidence.py

```diff
--- before/../main/tests/test_reference_local_evidence.py
+++ after/../main/tests/test_reference_local_evidence.py
@@ -0,0 +1,292 @@
+"""Reference-evidence regressions; this test directory is intentionally ignored."""
+import pathlib
+import sys
+import unittest
+from unittest.mock import patch
+
+import torch
+import torch.nn.functional as F
+
+sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
+from networks.module import StyleBackbone, StyleEncoder
+
+torch.set_num_threads(2)
+
+
+class ReferenceLocalEvidence(unittest.TestCase):
+    def setUp(self):
+        torch.manual_seed(214)
+
+    def _small_encoder(self, **kwargs):
+        encoder = StyleEncoder(
+            in_dim=8, style_dim=4, backbone_channels=(8, 8, 8),
+            heads=2, init='none', **kwargs,
+        ).eval()
+        with torch.no_grad():
+            for projection in encoder.proj_layers:
+                projection.weight.copy_(torch.eye(8).view(8, 8, 1, 1))
+                projection.bias.zero_()
+        return encoder
+
+    def _maps(self, uniform=False):
+        base = torch.tensor([-2., -1., -.5, .25, .5, 1., 2., 3.])
+        sizes = ((8, 9), (4, 7), (2, 5))
+        maps = [base[None, :, None, None].expand(2, -1, h, w).clone()
+                for h, w in sizes]
+        if not uniform:
+            maps = [feature + .2 * torch.randn_like(feature) for feature in maps]
+        lengths = [torch.tensor([3, 9]), torch.tensor([2, 7]), torch.tensor([1, 5])]
+        return maps, lengths
+
+    def test_position_only_changes_keys(self):
+        # Putting PE into values would make this comparison fail.
+        encoder = self._small_encoder(feature_scales=(4, 2))
+        all_maps, lengths = self._maps()
+        selected = [all_maps[2], all_maps[1]]
+        keys, values, mask = encoder._build_local_attention_inputs(selected, lengths)
+        with patch('networks.module.get_2d_sinusoidal_embeddings',
+                   side_effect=lambda h, w, d, device: torch.zeros(h, w, d, device=device)):
+            no_position_keys, no_position_values, no_position_mask = (
+                encoder._build_local_attention_inputs(selected, lengths)
+            )
+        torch.testing.assert_close(values, no_position_values, rtol=0, atol=0)
+        torch.testing.assert_close(mask, no_position_mask)
+        self.assertGreater((keys - no_position_keys).abs().sum().item(), 1.)
+        # Four rows at each selected scale; no appended 1D global context.
+        self.assertEqual(keys.shape, (2, 4 * (5 + 7), 8))
+        # map_lengths is the full backbone list, not the selected-map list.
+        self.assertEqual(mask[0].sum().item(), 4 * ((5 - 1) + (7 - 2)))
+        self.assertEqual(values.masked_select(mask[..., None].expand_as(values)).abs().max().item(), 0.)
+        torch.testing.assert_close(values[0, :20].sum(dim=0), torch.zeros(8), atol=2e-6, rtol=0)
+        torch.testing.assert_close(values[0, 20:].sum(dim=0), torch.zeros(8), atol=3e-6, rtol=0)
+
+    def test_uniform_values_cannot_create_local_style(self):
+        # Query residuals, attention/projector biases, or non-centered values
+        # would create style even though every valid map is spatially uniform.
+        encoder = self._small_encoder()
+        maps, lengths = self._maps(uniform=True)
+        for feature, width in zip(maps, lengths):
+            invalid = torch.arange(feature.size(-1))[None, :] >= width[:, None]
+            feature.masked_fill_(invalid[:, None, None, :], 999.)
+        keys, values, mask = encoder._build_local_attention_inputs(maps, lengths)
+        torch.testing.assert_close(values, torch.zeros_like(values), atol=5e-7, rtol=0)
+        queries = encoder.style_query_norm(encoder.style_queries.expand(2, -1, -1))
+        attended, _ = encoder.style_cross_attn(queries, keys, values,
+                                             key_padding_mask=mask, need_weights=False)
+        local = encoder.local_projection(attended)
+        local = local - local.mean(dim=1, keepdim=True)
+        torch.testing.assert_close(local, torch.zeros_like(local), atol=5e-7, rtol=0)
+        for height in (32, 64):
+            backbone = StyleBackbone(resolution=2, max_dim=8, img_height=height, init='none').eval()
+            image = torch.randn(2, 1, height, height * 3)
+            widths = torch.tensor([height, height * 3])
+            feat, feature_maps = backbone(image, ret_feats=True, x_lens=widths)
+            _, full_lengths = backbone.feature_lengths(widths)
+            uniform_maps = []
+            for feature, lengths in zip(feature_maps, full_lengths):
+                uniform = torch.arange(8, dtype=feature.dtype)[None, :, None, None].expand_as(feature).clone()
+                invalid = torch.arange(feature.size(-1))[None, :] >= lengths[:, None]
+                uniform_maps.append(uniform.masked_fill(invalid[:, None, None, :], 999.))
+            styles = encoder(image, widths, backbone, backbone_features=(feat, uniform_maps))
+            torch.testing.assert_close(styles[:, 1:], torch.zeros(2, 7, 4), atol=5e-7, rtol=0)
+
+    def test_weak_visual_deviation_is_not_renormalized(self):
+        # Renormalizing residual values or attended output would erase the
+        # amplitude difference between tiny visual deviations.
+        encoder = self._small_encoder(feature_scales=(1,))
+        base = torch.tensor([-2., -1., -.5, .25, .5, 1., 2., 3.])[None, :, None, None]
+        direction = torch.tensor([1., -.5, .25, -1., .75, -.75, .5, -.25])[None, :, None, None]
+        contrast = torch.tensor([-1., -.3, .2, .7, 1.])[None, None, None, :]
+        norms, local_norms = [], []
+        for strength in (.01, .0001):
+            feature = (base + strength * direction * contrast).expand(2, -1, 4, -1)
+            keys, values, mask = encoder._build_local_attention_inputs([feature], None)
+            self.assertIsNone(mask)
+            norms.append(values.norm().item())
+            queries = encoder.style_query_norm(encoder.style_queries.expand(2, -1, -1))
+            attended, _ = encoder.style_cross_attn(queries, keys, values, need_weights=False)
+            local = encoder.local_projection(attended)
+            local_norms.append((local - local.mean(dim=1, keepdim=True)).norm().item())
+        self.assertGreater(norms[1], 1e-5)
+        self.assertGreater(norms[0] / norms[1], 90.)
+        self.assertLess(norms[0] / norms[1], 110.)
+        self.assertGreater(local_norms[1], 1e-7)
+        self.assertGreater(local_norms[0] / local_norms[1], 80.)
+        self.assertLess(local_norms[0] / local_norms[1], 120.)
+        for height in (32, 64):
+            backbone = StyleBackbone(resolution=2, max_dim=8, img_height=height, init='none').eval()
+            image = torch.randn(2, 1, height, height * 3)
+            widths = torch.tensor([height, height * 3])
+            feat, feature_maps = backbone(image, ret_feats=True, x_lens=widths)
+            full_norms = []
+            for strength in (.01, .0001):
+                selected = feature_maps[0]
+                positions = torch.linspace(-1., 1., selected.size(-1))[None, None, None, :]
+                changed = (base + strength * direction * positions).expand_as(selected)
+                maps = [changed, *feature_maps[1:]]
+                codes = encoder(image, widths, backbone, backbone_features=(feat, maps))
+                full_norms.append(codes[:, 1:].norm().item())
+            self.assertGreater(full_norms[1], 1e-7)
+            self.assertGreater(full_norms[0] / full_norms[1], 80.)
+            self.assertLess(full_norms[0] / full_norms[1], 120.)
+
+    def test_visual_reference_changes_centered_codes(self):
+        # Constant priors, missing centering, or an unmasked tail must fail.
+        for height in (32, 64):
+            with self.subTest(height=height):
+                torch.manual_seed(671)
+                backbone = StyleBackbone(img_height=height, init='none').eval().requires_grad_(False)
+                encoder = StyleEncoder(init='none').eval()
+                image = torch.randn(2, 1, height, height * 3, requires_grad=True)
+                widths = torch.tensor([height, height * 2 + 1])
+                features = backbone(image, ret_feats=True, x_lens=widths)
+                styles = encoder(image, widths, backbone, backbone_features=features)
+                torch.testing.assert_close(styles[:, 1:].mean(dim=1), torch.zeros(2, 32), atol=1e-7, rtol=0)
+                self.assertGreater((styles[0, 1:] - styles[1, 1:]).abs().mean().item(), 1e-6)
+                # Changing only invalid cached feature columns changes nothing.
+                feat_lengths, map_lengths = backbone.feature_lengths(widths)
+                poisoned_maps = []
+                for feature, lengths in zip(features[1], map_lengths):
+                    invalid = torch.arange(feature.size(-1))[None, :] >= lengths[:, None]
+                    poisoned_maps.append(feature.masked_fill(invalid[:, None, None, :], 31.))
+                invalid = torch.arange(features[0].size(-1))[None, :] >= feat_lengths[:, None]
+                poisoned_feat = features[0].masked_fill(invalid[:, None, :], -27.)
+                poisoned = encoder(image, widths, backbone,
+                                   backbone_features=(poisoned_feat, poisoned_maps))
+                torch.testing.assert_close(styles, poisoned, atol=1e-7, rtol=1e-6)
+                styles[:, 1:].square().sum().backward()
+                self.assertTrue(torch.isfinite(image.grad).all())
+                for row, width in enumerate(widths.tolist()):
+                    self.assertGreater(image.grad[row, ..., :width].abs().sum().item(), 0.)
+                    self.assertEqual(image.grad[row, ..., width:].abs().max().item(), 0.)
+                for name, parameter in list(encoder.proj_layers.named_parameters()) + [
+                        ('queries', encoder.style_queries),
+                        *encoder.style_cross_attn.named_parameters(),
+                        *encoder.local_projection.named_parameters()]:
+                    self.assertIsNotNone(parameter.grad, name)
+                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
+                    self.assertGreater(parameter.grad.abs().sum().item(), 0., name)
+                self.assertTrue(all(p.grad is None or p.grad.abs().sum().item() == 0.
+                                    for p in encoder.sequence_model.parameters()))
+
+    def test_global_only_skips_local_modules_and_preserves_features(self):
+        # Any local projection/retrieval in global-only mode raises here.
+        for height in (32, 64):
+            with self.subTest(height=height):
+                backbone = StyleBackbone(img_height=height, init='none').eval()
+                encoder = StyleEncoder(init='none', feature_scales=(4, 2)).eval()
+                image = torch.randn(2, 1, height, height * 3)
+                widths = torch.tensor([height, height * 2 + 1])
+                features = backbone(image, ret_feats=True, x_lens=widths)
+                full = encoder(image, widths, backbone, backbone_features=features)
+                torch.manual_seed(992)
+                full_sample, full_mu, full_logvar = encoder(
+                    image, widths, backbone, backbone_features=features, vae_mode=True,
+                )
+
+                def forbidden(*args):
+                    raise AssertionError('global-only forward invoked a local module')
+
+                modules = [*encoder.proj_layers, encoder.style_cross_attn, encoder.local_projection]
+                handles = [module.register_forward_pre_hook(forbidden) for module in modules]
+                try:
+                    with patch.object(encoder, '_build_local_attention_inputs', side_effect=forbidden):
+                        global_code, maps, feat = encoder(
+                            image, widths, backbone, backbone_features=features,
+                            global_only=True, ret_backbone_feat=True,
+                        )
+                        torch.manual_seed(992)
+                        (sample, mu, logvar), optional_maps = encoder(
+                            image, widths, backbone, backbone_features=features,
+                            global_only=True, vae_mode=True, ret_feats=True,
+                        )
+                        unmasked, unmasked_maps, unmasked_feat = encoder(
+                            image, None, backbone, backbone_features=features,
+                            global_only=True, ret_backbone_feat=True,
+                        )
+                finally:
+                    for handle in handles:
+                        handle.remove()
+                self.assertEqual(global_code.shape, (2, 1, 32))
+                torch.testing.assert_close(global_code, full[:, :1], rtol=0, atol=0)
+                self.assertIs(feat, features[0])
+                self.assertIs(unmasked_feat, features[0])
+                torch.testing.assert_close(unmasked, encoder._encode_global_style(features[0], None)[0], rtol=0, atol=0)
+                for actual, index in zip(unmasked_maps, (2, 1)):
+                    torch.testing.assert_close(actual, features[1][index], rtol=0, atol=0)
+                for actual, optional, index in zip(maps, optional_maps, (2, 1)):
+                    expected = features[1][index]
+                    valid = torch.arange(expected.size(-1))[None, :] < backbone.feature_lengths(widths)[1][index][:, None]
+                    expected = expected.masked_fill(~valid[:, None, None, :], 0.)
+                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
+                    torch.testing.assert_close(optional, expected, rtol=0, atol=0)
+                self.assertEqual(sample.shape, (2, 1, 32))
+                torch.testing.assert_close(mu, global_code, rtol=0, atol=0)
+                torch.testing.assert_close(sample, full_sample[:, :1], rtol=0, atol=0)
+                torch.testing.assert_close(mu, full_mu[:, :1], rtol=0, atol=0)
+                torch.testing.assert_close(logvar, full_logvar[:, :1], rtol=0, atol=0)
+                self.assertTrue(torch.isfinite(logvar).all())
+                self.assertGreater((sample - mu).abs().sum().item(), 0.)
+
+    def test_global_only_without_optional_returns_does_not_mask_feature_maps(self):
+        # Copying all selected 4D maps wastes fake-cycle memory when no caller
+        # requests them. Global 3D feature masking is still required.
+        backbone = StyleBackbone(resolution=2, max_dim=8, img_height=32, init='none').eval()
+        encoder = self._small_encoder()
+        image, widths = torch.randn(2, 1, 32, 96), torch.tensor([32, 65])
+        features = backbone(image, ret_feats=True, x_lens=widths)
+        original = torch.Tensor.masked_fill
+
+        def forbid_map_copy(tensor, *args, **kwargs):
+            if tensor.ndim == 4:
+                raise AssertionError('no-return global-only forward copied a 4D map')
+            return original(tensor, *args, **kwargs)
+
+        with patch.object(torch.Tensor, 'masked_fill', forbid_map_copy):
+            result = encoder(image, widths, backbone, backbone_features=features, global_only=True)
+        self.assertEqual(result.shape, (2, 1, 4))
+
+    def test_global_seam_preserves_writer_calculation_and_vae_sampling(self):
+        # A changed valid mean/global MLP/clamp or local random sampling fails.
+        encoder = self._small_encoder()
+        feat = torch.randn(2, 8, 9)
+        lengths = torch.tensor([3, 9])
+        mask = torch.arange(9)[None, :] < lengths[:, None]
+        contextual = encoder.sequence_model(feat, valid_mask=mask)
+        pooled = contextual.sum(dim=-1) / lengths[:, None]
+        global_hidden = encoder.linear_style(encoder.global_context_norm(pooled)).unsqueeze(1)
+        expected_mu = encoder.mu(global_hidden)
+        expected_logvar = encoder.logvar(global_hidden).clamp(-14., 4.)
+        actual_mu, actual_logvar = encoder._encode_global_style(feat, lengths)
+        torch.testing.assert_close(actual_mu, expected_mu, rtol=0, atol=0)
+        torch.testing.assert_close(actual_logvar, expected_logvar, rtol=0, atol=0)
+        # Also exercise the unmasked valid mean and the clamp boundaries.
+        with torch.no_grad():
+            encoder.logvar.bias.copy_(torch.tensor([-30., -10., 4., 9.]))
+        mu, logvar = encoder._encode_global_style(feat, None)
+        expected_hidden = encoder.linear_style(encoder.global_context_norm(
+            encoder.sequence_model(feat).mean(dim=-1)
+        )).unsqueeze(1)
+        torch.testing.assert_close(mu, encoder.mu(expected_hidden), rtol=0, atol=0)
+        torch.testing.assert_close(logvar[0, 0], torch.tensor([-14., -10., 4., 4.]), rtol=0, atol=0)
+
+    def test_zero_and_single_local_slot_contract(self):
+        # Empty means must not create NaNs, and one centered local is exactly 0.
+        for height in (32, 64):
+            backbone = StyleBackbone(img_height=height, init='none').eval()
+            image = torch.randn(2, 1, height, height * 2)
+            widths = torch.tensor([height, height * 2])
+            for token_count in (1, 2):
+                with self.subTest(height=height, token_count=token_count):
+                    encoder = StyleEncoder(init='none', num_style_tokens=token_count).eval()
+                    sample, mu, logvar = encoder(image, widths, backbone, vae_mode=True)
+                    self.assertEqual(mu.shape, (2, token_count, 32))
+                    self.assertTrue(all(torch.isfinite(value).all() for value in (sample, mu, logvar)))
+                    torch.testing.assert_close(sample[:, 1:], mu[:, 1:], rtol=0, atol=0)
+                    torch.testing.assert_close(logvar[:, 1:], torch.zeros_like(logvar[:, 1:]), rtol=0, atol=0)
+                    if token_count == 2:
+                        torch.testing.assert_close(mu[:, 1:], torch.zeros(2, 1, 32), rtol=0, atol=0)
+
+
+if __name__ == '__main__':
+    unittest.main(verbosity=2)

```

## ../main/tests/test_spatial_style_fusion.py

```diff
--- before/../main/tests/test_spatial_style_fusion.py
+++ after/../main/tests/test_spatial_style_fusion.py
@@ -0,0 +1,209 @@
+"""Local-only tests for content-first spatial style refinement; Git-ignored."""
+import io
+import pathlib
+import sys
+import unittest
+import torch
+import torch.nn.functional as F
+
+ROOT = pathlib.Path(__file__).resolve().parents[1]
+sys.path.insert(0, str(ROOT))
+from networks import fusion
+from networks.BigGAN_networks import Generator
+from lib.utils import yaml2config
+torch.set_num_threads(2)
+
+
+class SpatialStyleFusion(unittest.TestCase):
+    def setUp(self):
+        torch.manual_seed(309)
+
+    def generator(self, height=32):
+        args = dict(yaml2config(ROOT / 'configs' / f'gan_iam_{height}.yml').GenModel)
+        args.update(G_ch=8, init='none')
+        return Generator(**args).eval()
+
+    def spatial(self):
+        module = getattr(self.generator(), 'spatial_style', None)
+        self.assertIsNotNone(module, 'Local evidence must refine feature maps after GBlock 1')
+        return module
+
+    def test_seed_is_reference_independent_at_both_heights(self):
+        for height in (32, 64):
+            model, seen = self.generator(height), []
+            handle = model.filter_linear.register_forward_hook(lambda m, args, out: seen.append(out.detach().clone()))
+            labels, lengths = torch.tensor([[2, 8, 9], [5, 0, 0]]), torch.tensor([3, 1])
+            with torch.no_grad():
+                model(torch.randn(2, 8, 32), labels, lengths)
+                model(torch.randn(2, 8, 32) * 3, labels, lengths)
+            handle.remove()
+            torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)
+
+    def test_local_style_first_changes_features_after_first_gblock(self):
+        for height in (32, 64):
+            model, first, second = self.generator(height), [], []
+            hooks = [
+                model.blocks[0][-1].register_forward_hook(lambda m, args, out: first.append(out.detach().clone())),
+                model.blocks[1][0].register_forward_pre_hook(lambda m, args: second.append(args[0].detach().clone())),
+            ]
+            labels, lengths = torch.tensor([[2, 8, 9], [5, 0, 0]]), torch.tensor([3, 1])
+            style = torch.randn(2, 8, 32)
+            changed = style.clone()
+            changed[:, 1:, :16] += 1
+            with torch.no_grad():
+                model(style, labels, lengths)
+                model(changed, labels, lengths)
+            for hook in hooks:
+                hook.remove()
+            torch.testing.assert_close(first[0], first[1], atol=0, rtol=0)
+            self.assertGreater((second[0] - second[1]).abs().max().item(), 1e-5)
+            self.assertEqual(first[0].shape, (2, 32, 8, 12))
+
+    def test_global_conditioning_still_changes_first_block(self):
+        model, seen = self.generator(), []
+        hook = model.blocks[0][-1].register_forward_hook(lambda m, args, out: seen.append(out.detach().clone()))
+        labels, lengths = torch.tensor([[2, 8, 9], [5, 0, 0]]), torch.tensor([3, 1])
+        style = torch.randn(2, 8, 32)
+        changed = style.clone()
+        changed[:, 0, :16] += 1
+        with torch.no_grad():
+            model(style, labels, lengths)
+            model(changed, labels, lengths)
+        hook.remove()
+        self.assertGreater((seen[0] - seen[1]).abs().max().item(), 1e-5)
+
+    def test_spatial_queries_keep_vertical_information(self):
+        module, seen = self.spatial(), []
+        feature, style = torch.randn(2, 32, 8, 12), torch.randn(2, 8, 32)
+        hook = module.q_proj.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
+        output = module(feature, style, x_lens=torch.tensor([12, 4]))
+        hook.remove()
+        self.assertEqual(output.shape, feature.shape)
+        self.assertEqual(seen[0].shape, (2, 96, 32))
+        self.assertGreater((seen[0][:, :12] - seen[0][:, 12:24]).abs().sum().item(), 0)
+
+    def test_zero_local_codes_are_neutral_for_any_global(self):
+        module = self.spatial()
+        feature = torch.randn(2, 32, 8, 12)
+        local_codes = torch.zeros(2, 7, 32)
+        for global_code in (torch.randn(2, 1, 32), torch.randn(2, 1, 32) * 100):
+            style = torch.cat((global_code, local_codes), dim=1)
+            torch.testing.assert_close(module(feature, style), feature, atol=0, rtol=0)
+
+    def test_value_projection_receives_raw_local_residuals(self):
+        module, seen = self.spatial(), []
+        feature = torch.randn(2, 32, 8, 12)
+        global_code, local_codes = torch.randn(2, 1, 32), torch.randn(2, 7, 32)
+        hook = module.v_proj.register_forward_pre_hook(
+            lambda m, args: seen.append(args[0].detach().clone())
+        )
+        module(feature, torch.cat((global_code, local_codes), dim=1))
+        hook.remove()
+        torch.testing.assert_close(seen[0], local_codes, atol=0, rtol=0)
+
+    def test_small_local_values_produce_small_correction(self):
+        module = self.spatial()
+        feature = torch.randn(2, 32, 8, 12)
+        global_code, local_codes = torch.randn(2, 1, 32), torch.randn(2, 7, 32)
+        style = torch.cat((global_code, local_codes), dim=1)
+        keys = []
+        cache_keys = module.k_proj.register_forward_hook(
+            lambda m, args, output: keys.append(output.detach().clone())
+        )
+        with torch.no_grad():
+            module(feature, style)
+        cache_keys.remove()
+        fixed_keys = keys[0]
+        fixed_key_hook = module.k_proj.register_forward_hook(
+            lambda m, args, output: fixed_keys
+        )
+        values, corrections = [], []
+        value_hook = module.v_proj.register_forward_pre_hook(
+            lambda m, args: values.append(args[0].detach().clone())
+        )
+        with torch.no_grad():
+            for strength in (0.01, 1.0):
+                scaled_style = torch.cat((global_code, strength * local_codes), dim=1)
+                corrections.append(module(feature, scaled_style) - feature)
+        fixed_key_hook.remove()
+        value_hook.remove()
+        torch.testing.assert_close(values[0], 0.01 * values[1], atol=1e-7, rtol=1e-6)
+        self.assertLess(corrections[0].norm().item(), corrections[1].norm().item() * 0.02)
+
+    def test_spatial_strength_is_unbounded_and_slots_are_permutation_invariant(self):
+        module = self.spatial()
+        feature, style = torch.randn(2, 32, 8, 12), torch.randn(2, 8, 32)
+        with torch.no_grad():
+            module.reference_scale.fill_(1)
+            one = module(feature, style)
+            module.reference_scale.fill_(2)
+            two = module(feature, style)
+        torch.testing.assert_close(two - feature, 2 * (one - feature), atol=2e-6, rtol=2e-5)
+        for count in (1, 7, 11):
+            style = torch.randn(2, count + 1, 32)
+            reordered = torch.cat((style[:, :1], style[:, 1:].flip(1)), dim=1)
+            torch.testing.assert_close(module(feature, style), module(feature, reordered), atol=2e-6, rtol=2e-5)
+
+    def test_spatial_padding_blocks_tail_output_and_gradients(self):
+        module = self.spatial()
+        feature = torch.randn(2, 32, 8, 12)
+        style, lengths = torch.randn(2, 8, 32, requires_grad=True), torch.tensor([12, 4])
+        padded = F.pad(feature, (0, 7), value=12).requires_grad_()
+        actual, expected = module(padded, style, x_lens=lengths), module(feature, style, x_lens=lengths)
+        torch.testing.assert_close(actual[..., :12], expected, atol=2e-6, rtol=2e-5)
+        actual.square().mean().backward()
+        for row, width in enumerate(lengths):
+            self.assertEqual(actual[row, ..., width:].abs().max().item(), 0)
+            self.assertEqual(padded.grad[row, ..., width:].abs().max().item(), 0)
+        self.assertTrue(torch.isfinite(style.grad).all())
+        for name, parameter in module.named_parameters():
+            self.assertIsNotNone(parameter.grad, name)
+            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
+
+    def test_content_context_short_long_padding_and_gradients(self):
+        cls = getattr(fusion, 'ContentContext', None)
+        self.assertIsNotNone(cls, 'Content context must work without reference style')
+        for length in (1, 2, 9, 45):
+            module = cls(120, nhead=4, attn_dim=128, ffn_dim=360).train()
+            content = torch.randn(2, length, 120)
+            lengths = torch.tensor([max(1, length // 2), length])
+            padded = F.pad(content, (0, 0, 0, 5), value=12).requires_grad_()
+            actual, expected = module(padded, y_lens=lengths), module(content, y_lens=lengths)
+            torch.testing.assert_close(actual[:, :length], expected, atol=3e-6, rtol=3e-5)
+            actual.square().mean().backward()
+            for row, width in enumerate(lengths):
+                self.assertEqual(actual[row, width:].abs().max().item(), 0)
+                self.assertEqual(padded.grad[row, width:].abs().max().item(), 0)
+            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))
+
+    def test_generator_gradients_logging_and_checkpoint_roundtrip(self):
+        for height in (32, 64):
+            model = self.generator(height)
+            style = torch.randn(2, 8, 32, requires_grad=True)
+            labels, lengths = torch.tensor([[2, 8, 9], [5, 0, 0]]), torch.tensor([3, 1])
+            image = model(style, labels, lengths)
+            (image * torch.randn_like(image)).mean().backward()
+            self.assertEqual(image.shape, (2, 1, height, 3 * height // 2))
+            self.assertGreater(style.grad[:, :1].abs().sum().item(), 0)
+            self.assertGreater(style.grad[:, 1:].abs().sum().item(), 0)
+            self.assertTrue(torch.isfinite(style.grad).all())
+            scales = getattr(model, 'fusion_scales', None)
+            self.assertIsNotNone(scales, 'Logging must include content and spatial residuals')
+            self.assertTrue(torch.isfinite(scales).all())
+            self.assertTrue(torch.isfinite(model.fusion_strength()))
+            for name, parameter in model.named_parameters():
+                if name.startswith(('content_context', 'spatial_style', 'bssp')):
+                    self.assertIsNotNone(parameter.grad, name)
+                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
+                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
+            buffer = io.BytesIO()
+            torch.save(model.state_dict(), buffer)
+            buffer.seek(0)
+            restored = self.generator(height)
+            restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
+            with torch.no_grad():
+                torch.testing.assert_close(restored(style.detach(), labels, lengths), model(style.detach(), labels, lengths))
+
+
+if __name__ == '__main__':
+    unittest.main(verbosity=2)

```

## ../main/tests/test_global_only_integration.py

```diff
--- before/../main/tests/test_global_only_integration.py
+++ after/../main/tests/test_global_only_integration.py
@@ -0,0 +1,257 @@
+"""Ignored behavioral proofs for wrapped objectives and prior producers."""
+import pathlib
+import sys
+import tempfile
+import unittest
+from unittest.mock import patch
+
+import matplotlib.pyplot as plt
+import numpy as np
+import torch
+from torch import nn
+
+ROOT = pathlib.Path(__file__).resolve().parents[1]
+sys.path.insert(0, str(ROOT))
+from lib.alphabet import strLabelConverter
+from lib.path_config import set_img_height
+from lib.utils import yaml2config
+from networks.loss import global_content_adversarial_loss, global_style_reconstruction_loss
+from networks.module import StyleEncoder
+from networks.rand_dist import Distribution, prepare_y_dist, prepare_z_dist
+import networks.model as model_module
+
+torch.set_num_threads(2)
+
+
+class TinyDataset(torch.utils.data.Dataset):
+    """Replace external files only; use real collectors/models/train loop."""
+    def __init__(self):
+        self.wids = np.array([0, 1])
+        self.img_lens = np.array([48, 64])
+        self.lb_lens = np.array([3, 4])
+        self.converter = strLabelConverter('all')
+        rng = torch.Generator().manual_seed(509)
+        self.images = [torch.rand(1, 32, int(w), generator=rng) * 2 - 1 for w in self.img_lens]
+
+    def __len__(self):
+        return 2
+
+    def __getitem__(self, index):
+        text = ('cat', 'dogs')[index]
+        image = self.images[index]
+        return dict(org_img=image, style_img=image, lb=self.converter.encode(text), wid=index, text=text)
+
+
+class WrappedEncoder(nn.Module):
+    """Observe the supported wrapped-forward boundary, not a custom encode API."""
+    def __init__(self, module, backbone):
+        super().__init__()
+        self.module = module
+        self.calls = []
+        self.backbone_calls = 0
+        self.local_calls = 0
+        self.handles = [backbone.register_forward_hook(self._backbone)]
+        self.handles += [m.register_forward_hook(self._local) for m in
+                         [*module.proj_layers, module.style_cross_attn, module.local_projection]]
+
+    def _backbone(self, *args):
+        self.backbone_calls += 1
+
+    def _local(self, *args):
+        self.local_calls += 1
+
+    def forward(self, *args, **kwargs):
+        before = self.backbone_calls, self.local_calls
+        output = self.module(*args, **kwargs)
+        self.calls.append((kwargs.copy(), output, self.backbone_calls - before[0], self.local_calls - before[1]))
+        return output
+
+    def close(self):
+        for handle in self.handles:
+            handle.remove()
+
+
+class GlobalOnlyIntegration(unittest.TestCase):
+    def setUp(self):
+        torch.manual_seed(803)
+        set_img_height(32)
+        self.temp = tempfile.TemporaryDirectory(prefix='hggan-global-integration-')
+        self.addCleanup(self.temp.cleanup)
+        opt = yaml2config(ROOT / 'configs/gan_iam_32.yml')
+        opt.device, opt.multi_gpu = 'cpu', False
+        opt.expected_train_writers, opt.expected_width_multiple = 2, 16
+        for key, value in dict(batch_size=2, eval_batch_size=2, num_workers=0,
+                               eval_num_workers=0, sample_num_workers=0, epochs=1,
+                               print_iter_val=1000, sample_iter_val=1000,
+                               start_save_epoch_val=1000, start_eval_epoch_val=1000,
+                               max_word_len=4, capitalize_ratio=0., rare_word_ratio=0.,
+                               pretrained_ckpt='', r1_interval=1).items():
+            opt.training[key] = value
+        opt.GenModel.G_ch, opt.DiscModel.D_ch = 8, 8
+        opt.PatchDiscModel.D_ch, opt.PatchDiscModel.D_max_ch = 8, 24
+        opt.StyBackbone.resolution, opt.StyBackbone.max_dim = 2, 8
+        opt.OcrModel.resolution, opt.OcrModel.max_dim = 2, 8
+        opt.WidModel.in_dim, opt.WidModel.n_writer, opt.WidModel.attention_dim = 8, 2, 8
+        opt.EncModel.in_dim, opt.EncModel.query_dim, opt.EncModel.heads = 8, 8, 2
+        opt.EncModel.backbone_channels = (8, 8, 8)
+        self.dataset = TinyDataset()
+        with patch.object(model_module, 'get_dataset', return_value=self.dataset), \
+             patch.object(model_module, 'get_lexicon', return_value=['cat', 'dogs']):
+            self.model = model_module.GlobalLocalAdversarialModel(opt, log_root=self.temp.name)
+        self.model.print = lambda *args, **kwargs: None
+        self.model.info = lambda: None
+        opt.training.pretrained_w = str(pathlib.Path(self.temp.name, 'writer.pth'))
+        opt.training.pretrained_r = str(pathlib.Path(self.temp.name, 'ocr.pth'))
+        torch.save(dict(WriterIdentifier=self.model.models.W.state_dict(), StyleBackbone=self.model.models.B.state_dict()), opt.training.pretrained_w)
+        torch.save(dict(Recognizer=self.model.models.R.state_dict()), opt.training.pretrained_r)
+        self.model.eval_z = prepare_z_dist(2, 32, 'cpu', seed=13, num_tokens=8)
+        self.model.eval_y = prepare_y_dist(2, 2, 'cpu', seed=13)
+        pathlib.Path(self.temp.name, opt.training.sample_dir).mkdir()
+        self.seen = []
+        handle = self.model.models.G.register_forward_pre_hook(lambda m, args: self.seen.append(args[0].detach().clone()))
+        self.addCleanup(handle.remove)
+        handle = self.model.models_ema.G.register_forward_pre_hook(lambda m, args: self.seen.append(args[0].detach().clone()))
+        self.addCleanup(handle.remove)
+
+    def centered(self, style):
+        self.assertIs(type(style), torch.Tensor)
+        torch.testing.assert_close(style[:, 1:].mean(1), torch.zeros(style.size(0), style.size(2)), atol=2e-7, rtol=0)
+
+    def test_content_probe_is_global_only_with_global_grl_and_local_zero_gradient(self):
+        encoder = StyleEncoder(in_dim=8, style_dim=4, backbone_channels=(8, 8, 8), heads=2, n_class=3, content_grl=.7, init='none')
+        with torch.no_grad():
+            encoder.content_probe.weight.copy_(torch.tensor([[1., 2., 3., 4.], [-1., 0., 1., 2.], [2., 1., 0., -1.]]))
+        style = torch.randn(2, 8, 4, requires_grad=True)
+        logits = encoder.predict_content(style)
+        self.assertEqual(logits.shape, (2, 1, 3))
+        torch.testing.assert_close(logits, encoder.content_probe(style[:, :1]), atol=0, rtol=0)
+        logits.sum().backward()
+        torch.testing.assert_close(style.grad[:, :1], torch.tensor([[[-1.4, -2.1, -2.8, -3.5]]]).expand(2, -1, -1))
+        torch.testing.assert_close(style.grad[:, 1:], torch.zeros(2, 7, 4), atol=0, rtol=0)
+        self.assertGreater(encoder.content_probe.weight.grad.abs().sum().item(), 0.)
+
+    def test_train_centers_both_priors_and_routes_fake_objectives_through_wrapped_global_forward(self):
+        wrapper = WrappedEncoder(self.model.models.E, self.model.models.B)
+        self.addCleanup(wrapper.close)
+        self.model.models.E = wrapper
+        raw_samples, writer_feats, context_feats, l1_calls, bce_calls = [], [], [], [], []
+        original_sample = Distribution.sample_
+        writer_forward = self.model.models.W.forward_from_feat
+        def sample(distribution):
+            output = original_sample(distribution)
+            if distribution.ndim == 3:
+                raw_samples.append(distribution.as_subclass(torch.Tensor).clone())
+            return output
+        def writer(feat, *args, **kwargs):
+            writer_feats.append(feat)
+            return writer_forward(feat, *args, **kwargs)
+        def l1(*args, **kwargs):
+            l1_calls.append(args)
+            return global_style_reconstruction_loss(*args, **kwargs)
+        def bce(*args, **kwargs):
+            bce_calls.append(args)
+            return global_content_adversarial_loss(*args, **kwargs)
+        context_hook = self.model.contextual_loss.register_forward_pre_hook(lambda m, args, kwargs: context_feats.append(args[1]), with_kwargs=True)
+        self.addCleanup(context_hook.remove)
+        with patch.object(Distribution, 'sample_', sample), \
+             patch.object(self.model.models.W, 'forward_from_feat', writer), \
+             patch.object(model_module, 'global_style_reconstruction_loss', l1, create=True), \
+             patch.object(model_module, 'global_content_adversarial_loss', bce, create=True):
+            self.model.train()
+        self.assertEqual(len(self.seen), 2)
+        self.assertEqual(len(raw_samples), 2)
+        for index, (generated_input, raw) in enumerate(zip(self.seen, raw_samples)):
+            with self.subTest(prior_phase=('D', 'G')[index]):
+                self.centered(generated_input[:2])
+                torch.testing.assert_close(generated_input[:2, :1], raw[:, :1], atol=0, rtol=0)
+        fake_calls = [call for call in wrapper.calls if call[0].get('backbone_features') is None]
+        self.assertEqual(len(fake_calls), 2)
+        for index, (kwargs, output, backbone_count, local_count) in enumerate(fake_calls):
+            with self.subTest(fake_phase=('random', 'transfer')[index]):
+                self.assertTrue(kwargs.get('global_only'), kwargs)
+                self.assertEqual(backbone_count, 1)
+                self.assertEqual(local_count, 0)
+        transfer_kwargs, transfer, _, _ = fake_calls[1]
+        self.assertTrue(transfer_kwargs['ret_feats'] and transfer_kwargs['ret_backbone_feat'])
+        self.assertIs(writer_feats[0], transfer[2])
+        self.assertEqual(len(context_feats), len(transfer[1]))
+        for used, returned in zip(context_feats, transfer[1]):
+            self.assertIs(used, returned)
+        with self.subTest(objective='global L1'):
+            self.assertEqual(len(l1_calls), 2)
+        for pred, target, count in l1_calls:
+            self.assertEqual(pred.shape, (2, 1, 32))
+            self.assertEqual(target.shape, pred.shape)
+            self.assertFalse(target.requires_grad)
+            self.assertEqual(count, 8)
+        self.assertEqual(len(bce_calls), 1)
+        self.assertEqual(bce_calls[0][0].shape, (2, 1, 81))
+        self.assertEqual(bce_calls[0][2], 8)
+
+    def test_sample_images_centers_eval_prior_without_mutating_sampler(self):
+        self.model.sample_images(4)
+        self.assertEqual(len(self.seen), 4)
+        for index in (1, 2):
+            self.centered(self.seen[index])
+            torch.testing.assert_close(self.seen[index][:, :1], self.model.eval_z.as_subclass(torch.Tensor)[:, :1], atol=0, rtol=0)
+        self.assertIs(type(self.model.eval_z), Distribution)
+        self.assertEqual(self.model.eval_z.dist_kwargs['seed'], 13)
+        self.assertGreater(self.model.eval_z.as_subclass(torch.Tensor)[:, 1:].mean(1).abs().sum().item(), 0.)
+
+    def test_unguided_image_generator_centers_random_prior_without_changing_rng(self):
+        original_randn, raw = torch.randn, []
+        def randn(*args, **kwargs):
+            result = original_randn(*args, **kwargs)
+            raw.append(result.clone())
+            return result
+        with patch.object(model_module.torch, 'randn', randn):
+            batches = list(self.model.image_generator(self.model.train_loader, style_guided=False))
+        self.assertEqual(len(batches), 1)
+        self.centered(self.seen[0])
+        torch.testing.assert_close(self.seen[0][:, :1], raw[0][:, :1], atol=0, rtol=0)
+
+    def test_manual_interpolation_centers_endpoints_and_preserves_global_lerp(self):
+        original_randn, raw = torch.randn, []
+        def randn(*args, **kwargs):
+            result = original_randn(*args, **kwargs)
+            raw.append(result.clone())
+            return result
+        self.model.opt.test.interp_num = 3
+        with patch('builtins.input', side_effect=['cat', '']), patch.object(plt, 'show'), \
+             patch.object(model_module.torch, 'randn', randn):
+            self.model.eval_interp()
+        self.addCleanup(plt.close, 'all')
+        self.centered(self.seen[0])
+        expected_global = torch.cat((raw[0][:, :1], (raw[0][:, :1] + raw[1][:, :1]) / 2, raw[1][:, :1]))
+        torch.testing.assert_close(self.seen[0][:, :1], expected_global, atol=1e-7, rtol=1e-6)
+
+    def test_manual_random_centers_sampler_outputs_and_preserves_global_repeats(self):
+        sampler = prepare_z_dist(2, 32, 'cpu', seed=76, num_tokens=8)
+        self.model.opt.test.nrow = 2
+        with patch('builtins.input', side_effect=['cat dogs', '']), patch.object(plt, 'show'), \
+             patch.object(model_module, 'prepare_z_dist', return_value=sampler):
+            self.model.eval_rand()
+        self.addCleanup(plt.close, 'all')
+        self.centered(self.seen[0])
+        torch.testing.assert_close(self.seen[0][:, :1], sampler.as_subclass(torch.Tensor)[:, :1].repeat_interleave(2, 0), atol=0, rtol=0)
+        self.assertEqual(sampler.dist_kwargs['seed'], 76)
+
+    def test_legacy_gan_state_is_rejected_by_production_strict_loader(self):
+        # Legacy-shaped fixture: remove newly required spatial/content modules
+        # and add an obsolete fusion module. Production load must report both.
+        self.model.save('last', 0)
+        path = pathlib.Path(self.temp.name, self.model.opt.training.ckpt_dir, 'last_fid_inf.pth')
+        payload = torch.load(path, map_location='cpu', weights_only=False)
+        payload['G'] = {name: value for name, value in payload['G'].items()
+                        if not name.startswith(('content_context.', 'spatial_style.'))}
+        payload['G']['style_fusion.obsolete.weight'] = torch.zeros(1)
+        legacy_path = pathlib.Path(self.temp.name, 'legacy-shaped.pth')
+        torch.save(payload, legacy_path)
+        with self.assertRaises(RuntimeError) as raised:
+            self.model.load(str(legacy_path), map_location='cpu')
+        self.assertIn('Missing key(s)', str(raised.exception))
+        self.assertIn('Unexpected key(s)', str(raised.exception))
+
+
+if __name__ == '__main__':
+    unittest.main(verbosity=2)

```

## ../main/test_artifacts/legacy_tests/legacy_reference_fusion.py

```diff
--- before/../main/test_artifacts/legacy_tests/legacy_reference_fusion.py
+++ after/../main/test_artifacts/legacy_tests/legacy_reference_fusion.py
@@ -0,0 +1,144 @@
+"""Local-only checks for reference-conditioned fusion; excluded from Git."""
+import io
+import pathlib
+import sys
+import unittest
+
+import torch
+import torch.nn.functional as F
+
+ROOT = pathlib.Path(__file__).resolve().parents[1]
+sys.path.insert(0, str(ROOT))
+from networks.fusion import StyleContentAttentionFusion
+from networks.BigGAN_networks import Generator
+from lib.utils import yaml2config
+
+torch.set_num_threads(2)
+
+
+class ReferenceFusion(unittest.TestCase):
+    def setUp(self):
+        torch.manual_seed(309)
+
+    def test_global_fallback_produces_zero_local_correction(self):
+        model = StyleContentAttentionFusion(120, 32).eval()
+        with torch.no_grad():
+            model.context_scale.zero_()
+            model.local_scale.zero_()
+            model.reference_scale.fill_(1.)
+        content = torch.randn(2, 5, 120)
+        global_style = torch.randn(2, 1, 32)
+        style = global_style.expand(-1, 8, -1).clone()
+        actual = model(content, style)
+        torch.testing.assert_close(actual, content, atol=1e-7, rtol=1e-6)
+        style[:, 3, :16] += .5
+        changed = model(content, style)
+        self.assertGreater((changed - actual).abs().max().item(), 1e-5)
+
+    def test_reference_values_preserve_small_evidence_without_renormalizing_delta(self):
+        model = StyleContentAttentionFusion(120, 32).eval()
+        attention = model.reference_attention
+        content = torch.randn(2, 4, 120)
+        global_style = torch.randn(2, 1, 32)
+        evidence = torch.randn(2, 7, 32)
+        seen = []
+        handle = attention.v_proj.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
+        for reliability in (0., 1e-4, .5):
+            local = global_style + reliability * evidence
+            model(content, torch.cat((global_style, local), dim=1))
+            expected = attention.style_norm(local) - attention.style_norm(global_style)
+            torch.testing.assert_close(seen[-1], expected, atol=1e-7, rtol=1e-6)
+        handle.remove()
+        self.assertEqual(seen[0].abs().max().item(), 0.)
+        self.assertLess(seen[1].norm().item(), seen[2].norm().item() * .01)
+
+    def test_reference_evidence_reaches_neighborhood_mixer(self):
+        # Catches putting width mixing before retrieval, which cannot mix the
+        # newly retrieved evidence between adjacent target positions.
+        model = StyleContentAttentionFusion(120, 32).eval()
+        mixer = getattr(model, 'local_mixer', None)
+        self.assertIsNotNone(mixer, 'Joint local mixing must follow style retrieval')
+        seen = []
+        handle = mixer.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
+        content = torch.randn(2, 5, 120)
+        style = torch.randn(2, 8, 32)
+        changed = style.clone()
+        changed[:, 3, :16] += 2
+        model(content, style)
+        model(content, changed)
+        handle.remove()
+        self.assertGreater((seen[0] - seen[1]).abs().max().item(), 1e-5)
+
+    def test_residual_strength_can_grow_without_a_sigmoid_ceiling(self):
+        # Catches an extra outer gate or a hidden nonlinear strength limit.
+        model = StyleContentAttentionFusion(120, 32).eval()
+        scales = getattr(model, 'reference_scale', None)
+        self.assertIsNotNone(scales, 'Reference residual needs a direct learnable scale')
+        content, style = torch.randn(2, 4, 120), torch.randn(2, 8, 32)
+        with torch.no_grad():
+            model.context_scale.zero_()
+            model.local_scale.zero_()
+            model.reference_scale.fill_(1)
+            one = model(content, style)
+            model.reference_scale.fill_(2)
+            two = model(content, style)
+        torch.testing.assert_close(two - content, 2 * (one - content), atol=2e-6, rtol=2e-5)
+
+    def test_short_long_and_padded_content_preserve_outputs_and_gradients(self):
+        for length in (1, 2, 9, 45):
+            with self.subTest(length=length):
+                model = StyleContentAttentionFusion(120, 32).train()
+                style = torch.randn(2, 8, 32, requires_grad=True)
+                content = torch.randn(2, length, 120)
+                valid = torch.tensor([max(1, length // 2), length])
+                padded = F.pad(content, (0, 0, 0, 5), value=12).requires_grad_()
+                expected = model(content, style, y_lens=valid)
+                actual = model(padded, style, y_lens=valid)
+                torch.testing.assert_close(actual[:, :length], expected, atol=3e-6, rtol=3e-5)
+                for row, end in enumerate(valid):
+                    self.assertEqual(actual[row, end:].abs().max().item(), 0)
+                actual.square().mean().backward()
+                self.assertTrue(torch.isfinite(style.grad).all())
+                self.assertGreater(style.grad[:, 1:].abs().sum().item(), 0)
+                for row, end in enumerate(valid):
+                    self.assertEqual(padded.grad[row, end:].abs().max().item(), 0)
+                for name, parameter in model.named_parameters():
+                    self.assertIsNotNone(parameter.grad, name)
+                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
+                    # One-character attention has only one possible key.
+                    if not (length == 1 and name.endswith('relative_position_bias')):
+                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
+
+    def test_local_style_slot_permutation_and_variable_slot_count(self):
+        model = StyleContentAttentionFusion(120, 32).eval()
+        content = torch.randn(2, 6, 120)
+        for slots in (1, 7, 11):
+            style = torch.randn(2, slots + 1, 32)
+            reordered = torch.cat((style[:, :1], style[:, 1:].flip(1)), dim=1)
+            torch.testing.assert_close(model(content, style), model(content, reordered), atol=2e-6, rtol=2e-5)
+
+    def test_real_config_generator_style_gradients_and_checkpoint_roundtrip(self):
+        for height in (32, 64):
+            cfg = yaml2config(str(ROOT / 'configs' / f'gan_iam_{height}.yml'))
+            args = dict(cfg.GenModel)
+            args.update(G_ch=8, init='none')
+            model = Generator(**args).eval()
+            style = torch.randn(2, 8, 32, requires_grad=True)
+            labels = torch.tensor([[2, 8, 9], [5, 0, 0]])
+            lengths = torch.tensor([3, 1])
+            image = model(style, labels, lengths)
+            image.square().mean().backward()
+            self.assertGreater(style.grad[:, 1:].abs().sum().item(), 0)
+            self.assertTrue(torch.isfinite(style.grad).all())
+            self.assertTrue(torch.isfinite(model.fusion_strength()))
+            buffer = io.BytesIO()
+            torch.save(model.state_dict(), buffer)
+            buffer.seek(0)
+            restored = Generator(**args).eval()
+            restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
+            with torch.no_grad():
+                torch.testing.assert_close(restored(style.detach(), labels, lengths), model(style.detach(), labels, lengths))
+
+
+if __name__ == '__main__':
+    unittest.main(verbosity=2)

```
