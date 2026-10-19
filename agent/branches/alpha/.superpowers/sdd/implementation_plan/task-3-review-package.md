# Review package: task-3
Uncommitted task changes; HEAD and index are untouched.

## networks/fusion.py

```diff
--- before/networks/fusion.py
+++ after/networks/fusion.py
@@ -90,45 +90,43 @@
         for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
             nn.init.xavier_uniform_(layer.weight)
         with torch.no_grad():
             self.reference_scale.fill_(self.residual_init)
 
     def forward(self, feature, style_seq, x_lens=None):
         if feature.ndim != 4 or feature.size(1) != self.d_model:
             raise ValueError('spatial style refinement requires BCHW features with matching channels')
         style_seq = ensure_dim3(style_seq)
         if style_seq.size(0) != feature.size(0) or style_seq.size(1) < 2:
             raise ValueError('spatial refinement requires one global and at least one local token per image')
         batch_size, channels, height, width = feature.shape
         spatial_seq = feature.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
         mask = None
         if x_lens is not None:
             lengths = x_lens.to(feature.device).long().clamp(1, width)
             width_mask = torch.arange(width, device=feature.device)[None, :] < lengths[:, None]
             mask = width_mask[:, None, :].expand(-1, height, -1).reshape(batch_size, height * width)
         spatial_seq = _mask_tokens(spatial_seq, mask)
         length = spatial_seq.size(1)
-        global_style, local_style_seq = style_seq[:, 0], style_seq[:, 1:]
+        local_style_seq = style_seq[:, 1:]
         style = self.style_norm(local_style_seq)
-        # Subtract in the same space as the existing value projection. Do not
-        # normalize the difference again: weak evidence must remain weak.
-        value_style = style - self.style_norm(global_style).unsqueeze(1)
+        value_style = local_style_seq
         query = self.q_proj(self.feature_norm(spatial_seq)).view(
             batch_size, length, self.nhead, self.head_dim
         ).transpose(1, 2)
         key = self.k_proj(style).view(
             batch_size, style.size(1), self.nhead, self.head_dim
         ).transpose(1, 2)
         value = self.v_proj(value_style).view(
             batch_size, style.size(1), self.nhead, self.head_dim
         ).transpose(1, 2)
         attended = F.scaled_dot_product_attention(
             query, key, value, dropout_p=0.0, is_causal=False
         )
         attended = attended.transpose(1, 2).reshape(batch_size, length, -1)
         refined = _mask_tokens(
             spatial_seq + self.reference_scale * self.out_proj(attended), mask
         )
         return refined.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
 
 
 class LocalGatedFeedForward(nn.Module):

```

## tests/test_spatial_style_fusion.py

```diff
--- before/tests/test_spatial_style_fusion.py
+++ after/tests/test_spatial_style_fusion.py
@@ -65,61 +65,87 @@
         labels, lengths = torch.tensor([[2, 8, 9], [5, 0, 0]]), torch.tensor([3, 1])
         style = torch.randn(2, 8, 32)
         changed = style.clone()
         changed[:, 0, :16] += 1
         with torch.no_grad():
             model(style, labels, lengths)
             model(changed, labels, lengths)
         hook.remove()
         self.assertGreater((seen[0] - seen[1]).abs().max().item(), 1e-5)
 
     def test_spatial_queries_keep_vertical_information(self):
         module, seen = self.spatial(), []
         feature, style = torch.randn(2, 32, 8, 12), torch.randn(2, 8, 32)
         hook = module.q_proj.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
         output = module(feature, style, x_lens=torch.tensor([12, 4]))
         hook.remove()
         self.assertEqual(output.shape, feature.shape)
         self.assertEqual(seen[0].shape, (2, 96, 32))
         self.assertGreater((seen[0][:, :12] - seen[0][:, 12:24]).abs().sum().item(), 0)
 
-    def test_global_fallback_has_zero_local_correction(self):
-        module = self.spatial()
-        feature, global_style = torch.randn(2, 32, 8, 12), torch.randn(2, 1, 32)
-        style = global_style.expand(-1, 8, -1).clone()
-        torch.testing.assert_close(module(feature, style), feature, atol=0, rtol=0)
-        style[:, 3, :16] += .5
-        self.assertGreater((module(feature, style) - feature).abs().max().item(), 1e-5)
-
-    def test_weak_evidence_values_are_not_renormalized(self):
+    def test_zero_local_codes_are_neutral_for_any_global(self):
+        module = self.spatial()
+        feature = torch.randn(2, 32, 8, 12)
+        local_codes = torch.zeros(2, 7, 32)
+        for global_code in (torch.randn(2, 1, 32), torch.randn(2, 1, 32) * 100):
+            style = torch.cat((global_code, local_codes), dim=1)
+            torch.testing.assert_close(module(feature, style), feature, atol=0, rtol=0)
+
+    def test_value_projection_receives_raw_local_residuals(self):
         module, seen = self.spatial(), []
         feature = torch.randn(2, 32, 8, 12)
-        global_style, evidence = torch.randn(2, 1, 32), torch.randn(2, 7, 32)
-        hook = module.v_proj.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
-        for reliability in (0., 1e-4, .5):
-            local = global_style + reliability * evidence
-            module(feature, torch.cat((global_style, local), dim=1))
-            expected = F.layer_norm(local, (32,)) - F.layer_norm(global_style, (32,))
-            torch.testing.assert_close(seen[-1], expected, atol=1e-7, rtol=1e-6)
+        global_code, local_codes = torch.randn(2, 1, 32), torch.randn(2, 7, 32)
+        hook = module.v_proj.register_forward_pre_hook(
+            lambda m, args: seen.append(args[0].detach().clone())
+        )
+        module(feature, torch.cat((global_code, local_codes), dim=1))
         hook.remove()
-        self.assertEqual(seen[0].abs().max().item(), 0)
-        self.assertLess(seen[1].norm().item(), seen[2].norm().item() * .01)
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
 
     def test_spatial_strength_is_unbounded_and_slots_are_permutation_invariant(self):
         module = self.spatial()
         feature, style = torch.randn(2, 32, 8, 12), torch.randn(2, 8, 32)
         with torch.no_grad():
             module.reference_scale.fill_(1)
             one = module(feature, style)
             module.reference_scale.fill_(2)
             two = module(feature, style)
         torch.testing.assert_close(two - feature, 2 * (one - feature), atol=2e-6, rtol=2e-5)
         for count in (1, 7, 11):
             style = torch.randn(2, count + 1, 32)
             reordered = torch.cat((style[:, :1], style[:, 1:].flip(1)), dim=1)
             torch.testing.assert_close(module(feature, style), module(feature, reordered), atol=2e-6, rtol=2e-5)
 
     def test_spatial_padding_blocks_tail_output_and_gradients(self):
         module = self.spatial()
         feature = torch.randn(2, 32, 8, 12)
         style, lengths = torch.randn(2, 8, 32, requires_grad=True), torch.tensor([12, 4])
         padded = F.pad(feature, (0, 7), value=12).requires_grad_()

```
