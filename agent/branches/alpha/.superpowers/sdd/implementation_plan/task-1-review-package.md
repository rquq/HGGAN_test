# Review package: task-1
Uncommitted task changes; HEAD and index are untouched.

## networks/utils.py

```diff
--- before/networks/utils.py
+++ after/networks/utils.py
@@ -222,40 +222,59 @@
     nParams = sum([p.nelement() for p in model.parameters()])
     mSize = nParams * 4.0 / 1024 / 1024
     if tag is None:
         tag = type(model).__name__
     res = "%-12s P:%6dK  S:%8.4fMB" % (tag,  nParams / 1000, mSize)
     return res
 
 
 def set_requires_grad(nets, requires_grad=False):
     """Set requires_grad=False for all the networks to avoid unnecessary computations
     Parameters:
         nets (network list)   -- a list of networks
         requires_grad (bool)  -- whether the networks require gradients or not
     """
     if not isinstance(nets, list):
         nets = [nets]
     for net in nets:
         if net is not None:
             for param in net.parameters():
                 param.requires_grad = requires_grad
+
+
+def center_style_prior(styles: torch.Tensor) -> torch.Tensor:
+    """Center local style-prior tokens while leaving the global token intact."""
+    if not isinstance(styles, torch.Tensor):
+        raise TypeError('styles must be a torch.Tensor')
+    if styles.ndim != 3 or styles.size(1) < 1:
+        raise ValueError(
+            'styles must have shape (B, N, D) with token count N >= 1'
+        )
+    if type(styles) is not torch.Tensor:
+        styles = styles.as_subclass(torch.Tensor)
+    if styles.size(1) == 1:
+        return styles
+
+    global_style = styles[:, :1]
+    local_style = styles[:, 1:]
+    centered_local_style = local_style - local_style.mean(dim=1, keepdim=True)
+    return torch.cat((global_style, centered_local_style), dim=1)
 
 
 def _rare_lexicon_sampler(lexicon):
     """Return corpus words weighted toward genuinely rare alphabet symbols.
 
     Fabricated punctuation/digit strings can shift the training text
     distribution away from IAM. This sampler only reuses words in the configured
     corpus, while inverse-square-root character weights give Q/X/Z/J and other
     low-frequency glyphs more chances to reach the generator.
     """
     cache = getattr(idx_to_words, '_rare_cache', None)
     if cache is None:
         cache = {}
         idx_to_words._rare_cache = cache
     cache_key = (id(lexicon), len(lexicon))
     if cache_key in cache and cache[cache_key][0] is lexicon:
         return cache[cache_key][1]
 
     counts = {}
     for raw_word in lexicon:

```

## networks/loss.py

```diff
--- before/networks/loss.py
+++ after/networks/loss.py
@@ -216,40 +216,96 @@
             a, b, c, d = input.size()
 
             if feat_len is not None:
                 # mask for varying lengths
                 mask = _len2mask(feat_len, d).view(a, 1, 1, d)
                 input = input * mask
                 denom = (c * torch.clamp(feat_len, min=1)).view(a, 1, 1) * b
             else:
                 denom = float(b * c * d)
 
             features = input.view(a, b, c * d)
             G = torch.bmm(features, features.transpose(1, 2))
 
             return G / denom
 
 
 def _global_style(style):
     return style if style.dim() == 2 else style[:, 0]
 
 
+def global_style_reconstruction_loss(
+    pred: torch.Tensor, target: torch.Tensor, num_style_tokens: int
+) -> torch.Tensor:
+    """Apply the existing mean L1 to global style, retaining its all-token scale."""
+    if (
+        isinstance(num_style_tokens, bool)
+        or not isinstance(num_style_tokens, int)
+        or num_style_tokens <= 0
+    ):
+        raise ValueError('num_style_tokens must be a positive integer')
+    if (
+        pred.ndim != 3
+        or target.ndim != 3
+        or pred.shape != target.shape
+        or pred.size(0) < 1
+        or pred.size(1) not in (1, num_style_tokens)
+        or pred.size(2) < 1
+    ):
+        raise ValueError(
+            'pred and target must have matching shape (B, 1 or N, D)'
+        )
+    return F.l1_loss(pred[:, :1], target[:, :1]) / num_style_tokens
+
+
+def global_content_adversarial_loss(
+    logits: torch.Tensor, labels: torch.Tensor, num_style_tokens: int
+) -> torch.Tensor:
+    """Apply the existing bag-of-character BCE to global logits at all-token scale."""
+    if (
+        isinstance(num_style_tokens, bool)
+        or not isinstance(num_style_tokens, int)
+        or num_style_tokens <= 0
+    ):
+        raise ValueError('num_style_tokens must be a positive integer')
+    if (
+        logits.ndim != 3
+        or logits.size(0) < 1
+        or logits.size(1) != 1
+        or logits.size(2) < 1
+    ):
+        raise ValueError('logits must have shape (B, 1, C)')
+    if labels.ndim != 2 or labels.size(0) != logits.size(0):
+        raise ValueError('labels must have shape (B, L) matching logits batch size')
+
+    content_targets = torch.zeros_like(logits)
+    content_indices = labels[:, None, :].expand(-1, 1, -1)
+    content_indices = content_indices.clamp(0, logits.size(2) - 1)
+    content_targets.scatter_(2, content_indices, 1.0)
+    content_targets[..., 0] = 0.0
+    positive_fraction = content_targets.mean().detach().clamp(1e-4, 0.5)
+    positive_weight = ((1.0 - positive_fraction) / positive_fraction).clamp(max=10.0)
+    return F.binary_cross_entropy_with_logits(
+        logits, content_targets, pos_weight=positive_weight
+    ) / num_style_tokens
+
+
 
 
 def supervised_contrastive_style_loss(styles, writer_ids, temperature=0.1):
     """Pull same-writer global style tokens together and repel other writers."""
     features = F.normalize(_global_style(styles), dim=-1)
     writer_ids = writer_ids.view(-1)
     count = features.size(0)
     if count < 2:
         return features.sum() * 0.0
 
     self_mask = torch.eye(count, dtype=torch.bool, device=features.device)
     positive_mask = writer_ids[:, None].eq(writer_ids[None, :]) & ~self_mask
     valid_anchors = positive_mask.any(dim=1)
     if not valid_anchors.any():
         return features.sum() * 0.0
 
     logits = torch.matmul(features, features.t()) / temperature
     logits = logits - logits.max(dim=1, keepdim=True).values.detach()
     exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
     log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))

```

## tests/test_local_style_contract.py

```diff
--- before/tests/test_local_style_contract.py
+++ after/tests/test_local_style_contract.py
@@ -0,0 +1,172 @@
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
+        centered.sum().backward()
+        self.assertIsNotNone(base.grad)
+        self.assertTrue(torch.isfinite(base.grad).all())
+        torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before)
+        torch.testing.assert_close(
+            distribution.get_state()['generator'], sampler_rng_before
+        )
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
