# Review package: task-4
Uncommitted task changes; HEAD and index are untouched.

## networks/model.py

```diff
--- before/networks/model.py
+++ after/networks/model.py
@@ -1,48 +1,48 @@
 import csv
 import io
 import glob
 import torch, os
 import wandb
 from PIL import Image
 from munch import Munch
 from itertools import chain
 import matplotlib.pyplot as plt
 import numpy as np
 from distance import levenshtein
 from tqdm import tqdm
 from torch.utils.data.dataloader import DataLoader
 from torch.utils.data import WeightedRandomSampler
 from torch.nn import CTCLoss, CrossEntropyLoss
 import torch.distributed as dist
 import torch.nn.functional as F
 from metric.val_metrics import calculate_fid_kid_is
 from metric.mssim_psnr import calculate_mssim_psnr
 from networks.utils import _info, set_requires_grad, get_scheduler, idx_to_words, rescale_images, rescale_images2, \
-                            words_to_images, ctc_greedy_decoder, sample_stroke_patches, run_patch_discriminator, augment_word_batch, frozen_bn, restore_scheduler_state
+                            words_to_images, ctc_greedy_decoder, sample_stroke_patches, run_patch_discriminator, augment_word_batch, frozen_bn, restore_scheduler_state, center_style_prior
 from networks.BigGAN_networks import Generator, Discriminator, PatchDiscriminator
 from networks.module import Recognizer, WriterIdentifier, StyleEncoder, StyleBackbone
 from lib.datasets import get_dataset, get_collect_fn, Hdf5Dataset
 from lib.alphabet import strLabelConverter, get_lexicon, get_true_alphabet, Alphabets
 from lib.utils import draw_image, get_logger, AverageMeterManager, option_to_string, AverageMeter, plot_heatmap, write_wandb_log
 from networks.rand_dist import prepare_z_dist, prepare_y_dist
-from networks.loss import recn_l1_loss, CXLoss, KLloss, r1_reg
+from networks.loss import recn_l1_loss, CXLoss, KLloss, r1_reg, global_style_reconstruction_loss, global_content_adversarial_loss
 
 
 class EMA(object):
     def __init__(self, beta):
         super().__init__()
         self.beta = beta
         self.step = 0
 
     @torch.no_grad()
     def update_model_average(self, ma_model, current_model, beta):
         current = getattr(current_model, 'module', current_model)
         ema_parameters = dict(ma_model.named_parameters())
         for name, parameter in current.named_parameters():
             ema_parameters[name].mul_(beta).add_(parameter, alpha=1.0 - beta)
         # Running means, variances, spectral-norm buffers, and counters must match
         # the current model; parameter-only EMA left these buffers stale.
         ema_buffers = dict(ma_model.named_buffers())
         for name, buffer in current.named_buffers():
             ema_buffers[name].copy_(buffer)
 
@@ -985,41 +985,41 @@
     def train(self):
         raise NotImplementedError()
 
     def sample_images(self, iteration_done=0):
         self.set_mode('eval')
 
         device = self.device
         batchA = next(iter(self.tst_loader))
         batchB = next(iter(self.tst_loader2))
         batch = Hdf5Dataset.merge_batch(batchA, batchB, device)
 
         real_imgs, real_img_lens = batch['style_imgs'], batch['style_img_lens']
         style_refs, style_ref_lens = batch['org_imgs'], batch['org_img_lens']
         real_lbs, real_lb_lens = batch['lbs'], batch['lb_lens']
         use_ema = getattr(self, 'use_ema', False)
         generator = self.models_ema.G if use_ema else self.models.G
         encoder = self.models_ema.E if use_ema else self.models.E
 
         with torch.no_grad():
             self.eval_z.sample_()
-            eval_z_in = self.eval_z
+            eval_z_in = center_style_prior(self.eval_z)
 
             recn_imgs = None
             if 'E' in self.models:
                 enc_z = encoder(style_refs, style_ref_lens, self.models.B)
                 recn_imgs = generator(enc_z, real_lbs, real_lb_lens)
 
             fake_real_imgs = generator(eval_z_in, real_lbs, real_lb_lens)
 
             self.eval_y.sample_()
             sampled_words = idx_to_words(self.eval_y, self.lexicon, 0,
                                          self.opt.training.capitalize_ratio,
                                          self.opt.training.blank_ratio,
                                          rare_ratio=0.0,
                                          rare_lexicon=self.rare_lexicon)
             sampled_words[-2] = sampled_words[-1]
             fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
             fake_lbs, fake_lb_lens = fake_lbs.to(device), fake_lb_lens.to(device)
             fake_imgs = generator(eval_z_in, fake_lbs, fake_lb_lens)
             style_imgs = generator(enc_z, fake_lbs, fake_lb_lens)
 
@@ -1085,44 +1085,44 @@
                         sampled_words = idx_to_words(word_idx_sampler[:style_imgs.size(0)],
                                                      self.lexicon, 0, self.opt.training.capitalize_ratio,
                                                      blank_ratio=0,
                                                      rare_ratio=float(getattr(
                                                          self.opt.training,
                                                          'rare_word_ratio', 0.15,
                                                      )),
                                                      rare_lexicon=self.rare_lexicon)
                         content_lbs, content_lb_lens = self.label_converter.encode(sampled_words)
                     else:
                         content_lbs, content_lb_lens = style_lbs, style_lb_lens
 
                     content_lbs = content_lbs.to(device)
                     content_lb_lens = content_lb_lens.to(device)
                     fake_batch['lbs'], fake_batch['lb_lens'] = content_lbs, content_lb_lens
 
                     if style_guided:
                         enc_z = self.models.E(style_refs, style_ref_lens, self.models.B)
                     else:
                         num_tokens = getattr(self.opt.EncModel, 'num_style_tokens', 8)
-                        enc_z = torch.randn(
+                        enc_z = center_style_prior(torch.randn(
                             style_lb_lens.size(0), num_tokens, self.models.G.style_dim,
                             device=device,
-                        )
+                        ))
 
                     fake_batch['style_imgs'] = self.models.G(enc_z, content_lbs, content_lb_lens)
                     fake_batch['style_img_lens'] = fake_batch['lb_lens'] * self.opt.char_width
                     fake_batch['wids'] = batch['wids']
 
                     fake_batch['org_imgs'], fake_batch['org_img_lens'] =\
                                         rescale_images(fake_batch['style_imgs'],
                                         fake_batch['style_img_lens'],
                                         batch['org_img_lens'])
 
                     yield fake_batch
 
     def _cmmd_cache_path(self, dataset):
         """Key real embeddings by input geometry and actual dataset revision."""
         import hashlib
         subsets = []
         while not hasattr(dataset, 'file_path') and hasattr(dataset, 'dataset'):
             if hasattr(dataset, 'indices'):
                 subsets.append(tuple(int(index) for index in dataset.indices))
             dataset = dataset.dataset
@@ -1555,42 +1555,42 @@
         self.print('WID_wier:{:.2f}'.format(wier))
         return wier
 
     def eval_interp(self):
         self.set_mode('eval')
 
         with torch.no_grad():
             interp_num = self.opt.test.interp_num
             nrow, ncol = 1, interp_num
             while True:
                 text = input('input text: ')
                 if len(text) == 0:
                     break
 
                 fake_lbs = self.label_converter.encode(text)
                 fake_lbs = torch.LongTensor(fake_lbs).unsqueeze(0)
                 fake_lb_lens = torch.IntTensor([len(text)])
 
                 num_tokens = getattr(self.opt.EncModel, 'num_style_tokens', 8)
                 style_dim = getattr(self.opt.EncModel, 'style_dim', 32)
-                style0 = torch.randn((1, num_tokens, style_dim))
-                style1 = torch.randn(style0.size())
+                style0 = center_style_prior(torch.randn((1, num_tokens, style_dim)))
+                style1 = center_style_prior(torch.randn(style0.size()))
 
                 styles = [torch.lerp(style0, style1, i / (interp_num - 1)) for i in range(interp_num)]
                 styles = torch.cat(styles, dim=0).float().to(self.device)
 
                 fake_lbs, fake_lb_lens = fake_lbs.repeat(nrow * ncol, 1).to(self.device),\
                                          fake_lb_lens.repeat(nrow * ncol).to(self.device)
                 gen_imgs = self.models.G(styles, fake_lbs, fake_lb_lens)
                 gen_imgs = (1 - gen_imgs).squeeze(1).cpu().numpy() * 127
                 plt.figure()
                 for i in range(nrow * ncol):
                     plt.subplot(nrow, ncol, i + 1)
                     plt.imshow(gen_imgs[i], cmap='gray')
                     plt.axis('off')
                 plt.tight_layout()
                 plt.show()
 
     def eval_style(self):
         self.set_mode('eval')
 
         tst_loader = DataLoader(
@@ -1649,41 +1649,41 @@
 
     def eval_rand(self):
         self.set_mode('eval')
 
         with torch.no_grad():
             nrow, ncol = self.opt.test.nrow, 2
             rand_z = prepare_z_dist(nrow, self.opt.EncModel.style_dim, self.device, num_tokens=getattr(self.opt.EncModel, 'num_style_tokens', 8))
             while True:
                 text = input('input text: ')
                 if len(text) == 0:
                     break
 
                 texts = text.split(' ')
                 ncol = len(texts)
                 fake_lbs, fake_lb_lens = self.label_converter.encode(texts)
 
                 fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                 fake_lb_lens = fake_lb_lens.repeat(nrow, ).to(self.device)
 
                 rand_z.sample_()
-                rand_styles = rand_z.unsqueeze(1).repeat(1, ncol, 1, 1).view(nrow * ncol, rand_z.size(1), -1)
+                rand_styles = center_style_prior(rand_z).unsqueeze(1).repeat(1, ncol, 1, 1).view(nrow * ncol, rand_z.size(1), -1)
                 gen_imgs = self.models.G(rand_styles, fake_lbs, fake_lb_lens)
                 gen_imgs = (1 - gen_imgs).squeeze(1).cpu().numpy() * 127
                 plt.figure()
                 for i in range(nrow):
                     for j in range(ncol):
                         ax = plt.subplot(nrow, ncol, i * ncol + 1 + j)
                         gen_img = gen_imgs[i * ncol + j]
                         ax.imshow(gen_img, cmap='gray')
                         ax.axis('off')
                 plt.tight_layout()
                 plt.show()
 
     def eval_text(self):
         self.set_mode('eval')
 
         tst_loader = DataLoader(
             get_dataset(self.opt.dataset, self.opt.training.dset_split, process_style=True),
             batch_size=self.opt.test.nrow,
             shuffle=True,
             collate_fn=self.collect_fn,
@@ -2096,41 +2096,41 @@
                 #############################
                 # Optimizing Discriminator
                 #############################
                 self.optimizers.D.zero_grad(set_to_none=True)
                 self.optimizers.P.zero_grad(set_to_none=True)
                 set_requires_grad([self.models.G, self.models.E, self.models.R, self.models.W, self.models.B], False)
                 set_requires_grad([self.models.D, self.models.P], True)
                 # self.models.B.frozen_bn()
 
                 with torch.no_grad():
                     self.y.sample_()
                     sampled_words = idx_to_words(self.y, self.lexicon, max_label_len,
                                                  self.opt.training.capitalize_ratio,
                                                  self.opt.training.blank_ratio,
                                                  rare_ratio=rare_word_ratio,
                                                  rare_lexicon=self.rare_lexicon)
                     fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words, max_label_len)
                     fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()
 
                     self.z.sample_()
-                    z_in = self.z
+                    z_in = center_style_prior(self.z)
 
                     # B is frozen and in eval mode. Cache only its real-image
                     # features for this batch; E must run again with gradients
                     # during the G update.
                     reference_features = self.models.B(
                         style_refs, ret_feats=True, x_lens=style_ref_lens
                     )
                     if self.vae_mode:
                         enc_z, _, _ = self.models.E(
                             style_refs, style_ref_lens, self.models.B, vae_mode=True,
                             backbone_features=reference_features,
                         )
                     else:
                         enc_z = self.models.E(
                             style_refs, style_ref_lens, self.models.B, vae_mode=False,
                             backbone_features=reference_features,
                         )
 
                     # Batch forward all fake/generated types to avoid multiple GPU kernel launches
                     cat_z = torch.cat([z_in, enc_z, enc_z], dim=0)
@@ -2239,41 +2239,41 @@
                     self.optimizers.G.zero_grad(set_to_none=True)
                     set_requires_grad([self.models.D, self.models.P, self.models.R, self.models.W, self.models.B], False)
                     set_requires_grad([self.models.G, self.models.E], True)
                     # self.models.B.frozen_bn()
 
                     ##########################
                     # Prepare Fake Inputs
                     ##########################
                     self.y.sample_()
                     sampled_words = idx_to_words(self.y, self.lexicon, max_label_len,
                                                  self.opt.training.capitalize_ratio,
                                                  self.opt.training.blank_ratio,
                                                  sort=True,
                                                  rare_ratio=rare_word_ratio,
                                                  rare_lexicon=self.rare_lexicon)
 
                     fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words, max_label_len)
                     fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()
 
                     self.z.sample_()
-                    z_in = self.z
+                    z_in = center_style_prior(self.z)
 
                     # Keep reference strokes intact for style encoding.
                     if self.vae_mode:
                         enc_z, mu, logvar = self.models.E(
                             style_refs, style_ref_lens, self.models.B,
                             vae_mode=True,
                             backbone_features=reference_features,
                         )
                     else:
                         enc_z = self.models.E(
                             style_refs, style_ref_lens, self.models.B,
                             vae_mode=False,
                             backbone_features=reference_features,
                         )
 
                     # Batch forward all fake/generated types through G to avoid multiple GPU kernel launches
                     cat_z = torch.cat([z_in, enc_z, enc_z], dim=0)
                     cat_fake_lb_lens = torch.cat([fake_lb_lens, fake_lb_lens, real_lb_lens], dim=0)
                     cat_y = torch.cat([fake_lbs, fake_lbs, real_lbs], dim=0)
                     cat_fake_imgs = self.models.G(cat_z, cat_y, cat_fake_lb_lens)
@@ -2332,71 +2332,67 @@
                     # or sign inversion here; those are display-only elsewhere.
                     ctc_imgs = torch.cat([fake_imgs, style_imgs], dim=0)
                     ctc_img_lens = torch.cat([fake_img_lens, style_img_lens], dim=0)
                     ctc_log_probs = self.models.R(
                         ctc_imgs, ctc_img_lens, return_log_probs=True
                     )
                     fake_ctc_rand, fake_ctc_style = torch.chunk(ctc_log_probs, 2, dim=1)
 
                     fake_ctc_loss_rand = self.ctc_loss(fake_ctc_rand, fake_lbs,
                                                        torch.div(fake_img_lens, ctc_len_scale, rounding_mode='trunc'),
                                                        fake_lb_lens)
 
                     fake_ctc_loss_style = self.ctc_loss(fake_ctc_style, fake_lbs,
                                                         torch.div(style_img_lens, ctc_len_scale, rounding_mode='trunc'),
                                                         fake_lb_lens)
                     fake_ctc_loss = fake_ctc_loss_rand + fake_ctc_loss_style
 
 
                     ### Style Reconstruction, Transfer Cycle, and Disentanglement ###
                     styles = self.models.E(
-                        fake_imgs, fake_lb_lens * self.opt.char_width, self.models.B
+                        fake_imgs, fake_lb_lens * self.opt.char_width, self.models.B,
+                        global_only=True,
                     )
                     transferred_styles, style_img_feats, style_backbone_feat = (
                         self.models.E(
                             style_imgs,
                             style_img_lens,
                             self.models.B,
                             ret_feats=True,
                             ret_backbone_feat=True,
+                            global_only=True,
                         )
                     )
-                    info_loss = torch.mean(torch.abs(styles - z_in.detach()))
+                    encoder = self.unwrap_model(self.models.E)
+                    info_loss = global_style_reconstruction_loss(
+                        styles, z_in[:, :1].detach(), encoder.num_style_tokens
+                    )
                     real_style_for_loss = mu if self.vae_mode else enc_z
-                    style_cycle_loss = F.l1_loss(
-                        transferred_styles, real_style_for_loss.detach()
-                    )
-
-                    encoder = self.unwrap_model(self.models.E)
+                    style_cycle_loss = global_style_reconstruction_loss(
+                        transferred_styles, real_style_for_loss[:, :1].detach(),
+                        encoder.num_style_tokens,
+                    )
+
                     content_code = mu if self.vae_mode else enc_z
                     content_logits = encoder.predict_content(content_code, reverse=True)
-                    content_targets = torch.zeros_like(content_logits)
-                    content_indices = real_lbs[:, None, :].expand(
-                        -1, content_logits.size(1), -1
-                    )
-                    content_indices = content_indices.clamp(0, content_logits.size(2) - 1)
-                    content_targets.scatter_(2, content_indices, 1.0)
-                    content_targets[..., 0] = 0.0
-                    positive_fraction = content_targets.mean().detach().clamp(1e-4, 0.5)
-                    content_pos_weight = ((1.0 - positive_fraction) / positive_fraction).clamp(max=10.0)
-                    content_adv_loss = F.binary_cross_entropy_with_logits(
-                        content_logits, content_targets, pos_weight=content_pos_weight
+                    content_adv_loss = global_content_adversarial_loss(
+                        content_logits, real_lbs, encoder.num_style_tokens
                     )
 
                     ### Content Restruction ###
                     recn_loss = recn_l1_loss(recn_imgs, real_imgs, real_img_lens)
 
                     ### Writer identity and non-aligned style supervision ###
                     writer = self.unwrap_model(self.models.W)
                     style_wid_logits = writer.forward_from_feat(
                         style_backbone_feat, style_img_lens, self.models.B
                     )
                     fake_wid_loss = self.classify_loss(style_wid_logits, real_wids)
 
                     # Non-aligned feature matching directly supervises transfer
                     # to new text. Reuse B's cached real features and the fake
                     # features already computed by the style-cycle/writer path.
                     ctx_loss = recn_imgs.new_zeros(())
                     if context_weight > 0:
                         encoder = self.unwrap_model(self.models.E)
                         real_img_feats = [
                             reference_features[1][index]

```

## networks/module.py

```diff
--- before/networks/module.py
+++ after/networks/module.py
@@ -413,41 +413,41 @@
         )
         # Affine-free norms add no checkpoint state. They prevent the frozen
         # backbone's very different feature scales from dominating attention.
         self.style_key_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
         self.style_query_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
         # Local values describe visual deviations only. No affine offset or
         # output normalization can manufacture evidence from a uniform map.
         self.local_projection = nn.Linear(in_dim, style_dim, bias=False)
         self.content_probe = nn.Linear(style_dim, n_class)
 
         if init != 'none':
             init_weights(self, init)
         nn.init.constant_(self.logvar.weight, 0.)
         nn.init.constant_(self.logvar.bias, -10.)
 
     @staticmethod
     def global_token(style_tokens):
         return style_tokens[:, 0]
 
     def predict_content(self, style_tokens, reverse=True):
-        style_for_probe = style_tokens
+        style_for_probe = style_tokens[:, :1]
         if reverse:
             style_for_probe = _gradient_reverse(style_for_probe, self.content_grl)
         return self.content_probe(style_for_probe)
 
     def _encode_global_style(
         self, feat: torch.Tensor, feat_len: torch.Tensor | None,
     ) -> tuple[torch.Tensor, torch.Tensor]:
         """The original writer-summary path, shared by full/global-only calls."""
         feat_mask = (
             torch.arange(feat.size(-1), device=feat.device)[None, :] < feat_len[:, None]
         ) if feat_len is not None else None
         contextual = self.sequence_model(feat, valid_mask=feat_mask)
         if feat_len is None:
             global_context = contextual.mean(dim=-1)
         else:
             global_context = contextual.sum(dim=-1) / feat_len.unsqueeze(1).to(feat.dtype)
         global_style = self.linear_style(
             self.global_context_norm(global_context)
         ).unsqueeze(1)
         return self.mu(global_style), self.logvar(global_style).clamp(-14.0, 4.0)
@@ -517,49 +517,50 @@
                 'and EncModel.backbone_channels consistently.'
             )
         for index, (proj_layer, feature) in enumerate(zip(self.proj_layers, selected_all_feats)):
             if proj_layer.in_channels != feature.size(1):
                 raise RuntimeError(
                     f'Backbone feature {index} has {feature.size(1)} channels, but the '
                     f'configured projection expects {proj_layer.in_channels}.'
                 )
 
         backbone = getattr(cnn_backbone, 'module', cnn_backbone)
         if img_len is not None:
             feat_len, map_lengths = backbone.feature_lengths(img_len)
             feat_len = feat_len.clamp(min=1, max=feat.size(-1))
         else:
             feat_len, map_lengths = None, None
 
         global_mu, global_logvar = self._encode_global_style(feat, feat_len)
         # Optional returned maps preserve their original channels and selected
         # order even when local retrieval is skipped by objective-only calls.
         masked_all_feats = []
-        for feature_index, feature in zip(self.feature_indices, selected_all_feats):
-            width_mask = (
-                torch.arange(feature.size(-1), device=feature.device)[None, :]
-                < map_lengths[feature_index][:, None]
-            ) if map_lengths is not None else None
-            masked_all_feats.append(
-                feature.masked_fill(~width_mask[:, None, None, :], 0.0)
-                if width_mask is not None else feature
-            )
+        if ret_feats or ret_backbone_feat:
+            for feature_index, feature in zip(self.feature_indices, selected_all_feats):
+                width_mask = (
+                    torch.arange(feature.size(-1), device=feature.device)[None, :]
+                    < map_lengths[feature_index][:, None]
+                ) if map_lengths is not None else None
+                masked_all_feats.append(
+                    feature.masked_fill(~width_mask[:, None, None, :], 0.0)
+                    if width_mask is not None else feature
+                )
 
         if not global_only and self.num_local_queries:
             style_keys, style_values, key_padding_mask = self._build_local_attention_inputs(
                 selected_all_feats, map_lengths,
             )
             style_queries = self.style_query_norm(self.style_queries.expand(img.size(0), -1, -1))
             local_attended, _ = self.style_cross_attn(
                 query=style_queries,
                 key=style_keys,
                 value=style_values,
                 key_padding_mask=key_padding_mask,
                 need_weights=False,
             )
             local_mu = self.local_projection(local_attended)
             local_mu = local_mu - local_mu.mean(dim=1, keepdim=True)
         else:
             local_mu = global_mu.new_empty(img.size(0), 0, self.style_dim)
         style_tokens_mu = torch.cat([global_mu, local_mu], dim=1)
 
         if vae_mode:

```

## tests/test_local_style_contract.py

```diff
--- before/tests/test_local_style_contract.py
+++ after/tests/test_local_style_contract.py
@@ -49,47 +49,64 @@
         self.assertTrue(torch.isfinite(styles.grad).all())
 
         single = styles.detach()[:, :1]
         torch.testing.assert_close(center_style_prior(single), single, rtol=0, atol=0)
         with self.assertRaisesRegex(ValueError, 'shape'):
             center_style_prior(torch.zeros(2, 3))
         with self.assertRaisesRegex(ValueError, 'token'):
             center_style_prior(torch.zeros(2, 0, 3))
 
     def test_distribution_prior_returns_base_tensor(self):
         base = torch.arange(24, dtype=torch.float32).view(2, 4, 3).requires_grad_()
         distribution = base.as_subclass(Distribution)
         distribution.init_distribution('normal', mean=0.0, var=1.0, seed=91)
         global_rng_before = torch.random.get_rng_state().clone()
         sampler_rng_before = distribution.get_state()['generator'].clone()
 
         centered = center_style_prior(distribution)
 
         self.assertIs(type(centered), torch.Tensor)
         self.assertTrue(centered.requires_grad)
-        centered.sum().backward()
+        weights = torch.tensor([[[2., 3., 4.], [1., 4., 7.], [3., 5., 9.], [8., 9., 11.]]]).expand_as(centered)
+        (centered * weights).sum().backward()
         self.assertIsNotNone(base.grad)
-        self.assertTrue(torch.isfinite(base.grad).all())
+        # d(x_i - mean(x))/dx_j = delta_ij - 1/3; a sum-only
+        # objective would hide a detached or incorrect local Jacobian.
+        torch.testing.assert_close(base.grad, torch.tensor([
+            [[2., 3., 4.], [-3., -2., -2.], [-1., -1., 0.], [4., 3., 2.]],
+            [[2., 3., 4.], [-3., -2., -2.], [-1., -1., 0.], [4., 3., 2.]],
+        ]))
         torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before)
         torch.testing.assert_close(
             distribution.get_state()['generator'], sampler_rng_before
         )
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
 
     def test_global_l1_matches_former_global_gradient(self):
         target = torch.tensor(
             [
                 [[0.0, 1.0], [4.0, 3.0], [1.0, 5.0]],
                 [[2.0, 0.0], [7.0, 2.0], [6.0, 8.0]],
             ]
         )
         old_pred = torch.tensor(
             [
                 [[2.0, -1.0], [5.0, 7.0], [3.0, 4.0]],
                 [[-1.0, 3.0], [9.0, 2.0], [4.0, 0.0]],
             ],
             requires_grad=True,
         )
         old_mean_loss = F.l1_loss(old_pred, target)
         old_mean_loss.backward()
 
         pred = old_pred.detach().clone().requires_grad_()
         loss = global_style_reconstruction_loss(pred, target, num_style_tokens=3)

```

## tests/check_training_background.py

```diff
--- before/tests/check_training_background.py
+++ after/tests/check_training_background.py
@@ -1,24 +1,25 @@
 """Bounded real-data/real-teacher smoke test of the actual GAN train loop."""
 import os
 import sys
 import tempfile
+from unittest.mock import patch
 import numpy as np
 import torch
 from torch.utils.data import DataLoader
 
 sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
 from lib.path_config import set_img_height
 from lib.utils import yaml2config
 import networks.model as model_module
 
 height = int(sys.argv[1])
 device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
 torch.set_num_threads(2)
 torch.manual_seed(123)
 if device.startswith('cuda'):
     torch.cuda.set_per_process_memory_fraction(.3)
 set_img_height(height)
 opt = yaml2config(f'configs/gan_iam_{height}.yml')
 opt.device = device
 opt.multi_gpu = False
 opt.training.batch_size = 2
@@ -38,38 +39,71 @@
 
 def cached_dataset(*args, **kwargs):
     key = (args, tuple(sorted(kwargs.items())))
     if key not in cache:
         cache[key] = get_dataset(*args, **kwargs)
     return cache[key]
 
 model_module.get_dataset = cached_dataset
 root = tempfile.mkdtemp(prefix=f'hggan-background-{height}-')
 model = model_module.GlobalLocalAdversarialModel(opt, log_root=root)
 
 def small_loader(loader, batch_size=2):
     data = loader.dataset
     indices = np.flatnonzero((data.lb_lens >= 3) & (data.lb_lens <= 6))[:16].tolist()
     return DataLoader(data, batch_size=batch_size, sampler=indices,
                       collate_fn=model.collect_fn, num_workers=0)
 
 model.train_loader = small_loader(model.train_loader)
 model.tst_loader = small_loader(model.tst_loader, batch_size=1)
 model.tst_loader2 = small_loader(model.tst_loader2, batch_size=1)
-model.train()
+assert len(model.train_loader) == 8
+initial_evidence = {}
+for module_name, prefixes in (
+        ('G', ('content_context', 'spatial_style', 'bssp')),
+        ('E', ('style_queries', 'proj_layers', 'style_cross_attn', 'local_projection', 'content_probe'))):
+    for name, parameter in model.models[module_name].named_parameters():
+        if name.startswith(prefixes):
+            initial_evidence[(module_name, name)] = parameter.detach().cpu().clone()
+
+original_update = model_module.AverageMeterManager.update_many
+finite_loss_batches = []
+def checked_update(meters, values):
+    for name, value in values.items():
+        assert torch.isfinite(torch.as_tensor(value)).all(), name
+    finite_loss_batches.append(tuple(values))
+    return original_update(meters, values)
+
+original_step = torch.optim.Adam.step
+finite_gradient_steps = []
+def checked_step(optimizer, *args, **kwargs):
+    for group in optimizer.param_groups:
+        for parameter in group['params']:
+            if parameter.grad is not None:
+                assert parameter.grad.dtype == torch.float32
+                assert torch.isfinite(parameter.grad).all()
+    finite_gradient_steps.append(1)
+    return original_step(optimizer, *args, **kwargs)
+
+with patch.object(model_module.AverageMeterManager, 'update_many', checked_update), \
+     patch.object(torch.optim.Adam, 'step', checked_step):
+    model.train()
 assert model.completed_epoch == 1
 expected_d_steps = len(model.train_loader)
 expected_g_steps = (expected_d_steps - 1) // opt.training.num_critic_train + 1
 assert not any('texture' in name for name in model.models.G.state_dict())
 for key, expected in (('D', expected_d_steps), ('P', expected_d_steps), ('G', expected_g_steps)):
     steps = {int(v['step'].item()) for v in model.optimizers[key].state.values() if 'step' in v}
     assert steps == {expected}, (key, steps, expected)
 assert model.ema_tracker.step == expected_g_steps
+assert (expected_d_steps, expected_g_steps) == (8, 4)
+assert len(finite_loss_batches) == expected_d_steps + expected_g_steps
+assert len(finite_gradient_steps) == expected_d_steps * 2 + expected_g_steps
 assert next(model.models.G.parameters()).dtype == torch.float32
 for key in ('G', 'E', 'D', 'P'):
     assert all(torch.isfinite(p).all() for p in model.models[key].parameters())
     print(key, 'finite; optimizer steps', sorted({int(v['step'].item()) for v in model.optimizers['G' if key == 'E' else key].state.values() if 'step' in v}), flush=True)
 os.makedirs(os.path.join(root, opt.training.sample_dir), exist_ok=True)
 model.sample_images(4)
 if device.startswith('cuda'):
     print('CUDA peak MiB', torch.cuda.max_memory_allocated() / 2**20, flush=True)
 print('ACTUAL TRAIN LOOP + SAMPLING PASSED', height, device, root, flush=True)

```

## tests/check_spatial_training.py

```diff
--- before/tests/check_spatial_training.py
+++ after/tests/check_spatial_training.py
@@ -1,61 +1,148 @@
 """Bounded native-height train/sample/checkpoint proof, not a quality benchmark."""
 import pathlib
+import copy
 import runpy
 import sys
 
 height = int(sys.argv[1]) if len(sys.argv) > 1 else 64
 sys.argv = ['tests/check_training_background.py', str(height), 'cuda:0']
 scope = runpy.run_path('tests/check_training_background.py')
-torch, model, opt, root = [scope[key] for key in ('torch', 'model', 'opt', 'root')]
+torch, model, opt, root, initial_evidence = [scope[key] for key in ('torch', 'model', 'opt', 'root', 'initial_evidence')]
 
 # New parameters must be owned by the GAN optimizer and updated on G steps.
-for name, parameter in model.models.G.named_parameters():
-    if name.startswith(('content_context', 'spatial_style', 'bssp')):
-        state = model.optimizers.G.state[parameter]
-        assert int(state['step'].item()) == 4, name
-        assert torch.isfinite(state['exp_avg']).all(), name
-        assert torch.isfinite(state['exp_avg_sq']).all(), name
+optimizer_ids = {id(parameter) for group in model.optimizers.G.param_groups for parameter in group['params']}
+evidence_changes = {}
+for (module_name, name), before in initial_evidence.items():
+    parameter = dict(model.models[module_name].named_parameters())[name]
+    assert id(parameter) in optimizer_ids, (module_name, name)
+    state = model.optimizers.G.state[parameter]
+    assert int(state['step'].item()) == 4, (module_name, name)
+    for moment in ('exp_avg', 'exp_avg_sq'):
+        assert torch.isfinite(state[moment]).all(), (module_name, name, moment)
+        assert state[moment].abs().sum().item() > 0, (module_name, name, moment)
+    change = (parameter.detach().cpu() - before).abs().max().item()
+    assert change > 0, (module_name, name, 'parameter never updated')
+    evidence_changes[(module_name, name)] = change
 assert model.ema_tracker.step == 4
-model.models_ema.G.load_state_dict(model.models.G.state_dict(), strict=True)
-print('NEW FUSION OPTIMIZER OWNERSHIP / STEPS AND EMA SHAPES PASS', flush=True)
+for module_name in ('G', 'E'):
+    changes = [value for (owner, _), value in evidence_changes.items() if owner == module_name]
+    print(module_name, 'EVIDENCE OPTIMIZER OWNERSHIP / 4 STEPS / ACTUAL UPDATES PASS',
+          'parameters', len(changes), 'max-delta range', min(changes), max(changes), flush=True)
 
 # Exercise maximum configured target length with short real style references.
 model.set_mode('eval')
 dataset = model.tst_loader.dataset
 indices = (dataset.lb_lens == 1).nonzero()[0][:2].tolist()
 assert len(indices) == 2
 batch = model.collect_fn([dataset[i] for i in indices])
 labels, lengths = model.label_converter.encode(['abcdefghijklmnopqrst', '/.'])
 with torch.no_grad():
     style = model.models.E(batch['org_imgs'].to(model.device),
                            batch['org_img_lens'].to(model.device), model.models.B)
     generated = model.models.G(style, labels.to(model.device), lengths.to(model.device))
 assert generated.shape == (2, 1, height, height * 10)
 assert generated.dtype == torch.float32 and torch.isfinite(generated).all()
 assert (generated[1, ..., height:] == -1).all()
 print('SHORT REAL REFERENCE / LONG TARGET / VALID TAIL FP32 PASS', height, flush=True)
 
 # Save a full newly generated checkpoint, then restore optimizer/EMA/progress.
 # No measured metrics are invented for this untrained smoke-test run.
-model.save('last', 1, iter_count=8, best_fid=float('inf'))
+model.save('last', model.current_epoch, iter_count=model.last_completed_iter, best_fid=float('inf'))
 checkpoint = pathlib.Path(root, opt.training.ckpt_dir, 'last_fid_inf.pth')
 assert checkpoint.is_file()
-expected = {name: tensor.detach().clone() for name, tensor in model.models.G.state_dict().items()}
-epoch = model.load(str(checkpoint), map_location=model.device)
+def cpu_snapshot(value):
+    if isinstance(value, torch.Tensor):
+        return value.detach().cpu().clone()
+    if isinstance(value, dict):
+        return {key: cpu_snapshot(item) for key, item in value.items()}
+    if isinstance(value, list):
+        return [cpu_snapshot(item) for item in value]
+    if isinstance(value, tuple):
+        return tuple(cpu_snapshot(item) for item in value)
+    return copy.deepcopy(value)
+
+def exact(actual, expected, path=''):
+    if isinstance(expected, torch.Tensor):
+        torch.testing.assert_close(actual.detach().cpu(), expected, atol=0, rtol=0, msg=path)
+    elif isinstance(expected, dict):
+        assert actual.keys() == expected.keys(), path
+        for key, value in expected.items():
+            exact(actual[key], value, path + '.' + str(key))
+    elif isinstance(expected, (tuple, list)):
+        assert len(actual) == len(expected), path
+        for index, value in enumerate(expected):
+            exact(actual[index], value, path + '.' + str(index))
+    else:
+        assert actual == expected, (path, actual, expected)
+
+expected_live = {owner: cpu_snapshot(model.models[owner].state_dict()) for owner in ('G', 'E')}
+expected_ema = {owner: cpu_snapshot(model.models_ema[owner].state_dict()) for owner in ('G', 'E')}
+expected_optimizers = {owner: cpu_snapshot(optimizer.state_dict()) for owner, optimizer in model.optimizers.items()}
+expected_schedulers = {owner: cpu_snapshot(scheduler.state_dict()) for owner, scheduler in model.lr_schedulers.items()}
+expected_lrs = {owner: [group['lr'] for group in optimizer.param_groups] for owner, optimizer in model.optimizers.items()}
+expected_progress = (model.current_epoch, model.last_completed_iter, model.ema_tracker.step)
+assert expected_progress == (1, 7, 4)
+
+# Make restoration observable instead of reloading equivalent live state.
+with torch.no_grad():
+    for owner in ('G', 'E'):
+        next(model.models[owner].parameters()).add_(1)
+        next(model.models_ema[owner].parameters()).add_(2)
+    for optimizer in model.optimizers.values():
+        for state in optimizer.state.values():
+            for value in state.values():
+                if isinstance(value, torch.Tensor):
+                    value.zero_()
+        for group in optimizer.param_groups:
+            group['lr'] = .123
+for scheduler in model.lr_schedulers.values():
+    scheduler.last_epoch = 999
+    scheduler._last_lr = [.123 for _ in scheduler.get_last_lr()]
+model.current_epoch, model.last_completed_iter, model.ema_tracker.step = 99, 999, -1
+epoch = model.load(str(checkpoint), map_location='cpu')
 assert epoch == 1
-for name, tensor in model.models.G.state_dict().items():
-    torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
-assert model.ema_tracker.step == 4
+for owner in ('G', 'E'):
+    exact(model.models[owner].state_dict(), expected_live[owner], owner)
+    exact(model.models_ema[owner].state_dict(), expected_ema[owner], owner + '_EMA')
+for owner, optimizer in model.optimizers.items():
+    exact(optimizer.state_dict(), expected_optimizers[owner], 'OPT.' + owner)
+    exact([group['lr'] for group in optimizer.param_groups], expected_lrs[owner], 'LR.' + owner)
+for owner, scheduler in model.lr_schedulers.items():
+    exact(scheduler.state_dict(), expected_schedulers[owner], 'SCHED.' + owner)
+assert (model.current_epoch, model.last_completed_iter, model.ema_tracker.step) == expected_progress
+assert model.restored_metadata['Epoch'] == 1
+assert model.restored_metadata['iter_count'] == 7
+assert model.restored_metadata['ema_step'] == 4
+assert model.resume_position(epoch, model.last_completed_iter, len(model.train_loader)) == (2, 0, 8)
 assert len(model.eval_history) == 0
-print('FULL NEW CHECKPOINT / OPTIMIZER / EMA RESTORE PASS', height, flush=True)
+print('E/G FULL STATE + E/G EMA + ALL OPTIMIZER MOMENTS/STEPS + SCHEDULER/LR + EXACT PROGRESS RESTORE PASS', height, flush=True)
+
+# A legacy-shaped GAN checkpoint must fail through production strict loading;
+# no old checkpoint happens to be installed in pretrained/bestIAM.pth here.
+legacy = torch.load(checkpoint, map_location='cpu', weights_only=False)
+legacy['G'] = {name: value for name, value in legacy['G'].items()
+               if not name.startswith(('content_context.', 'spatial_style.'))}
+legacy['G']['style_fusion.obsolete.weight'] = torch.zeros(1)
+legacy_path = pathlib.Path(root, 'legacy-shaped.pth')
+torch.save(legacy, legacy_path)
+del legacy
+try:
+    model.load(str(legacy_path), map_location='cpu')
+except RuntimeError as error:
+    assert 'Missing key(s)' in str(error) and 'Unexpected key(s)' in str(error), error
+else:
+    raise AssertionError('production load silently accepted legacy GAN state')
+model.load(str(checkpoint), map_location='cpu')
+print('LEGACY-SHAPED GAN STRICT MISSING/UNEXPECTED-KEY REJECTION PASS', height, flush=True)
 
 # Sampling and load must not leave the cuDNN OCR teacher in an invalid mode.
 model.set_mode('train')
 x = torch.randn(2, 1, height, height * 3, device=model.device, requires_grad=True)
 log_probs = model.models.R(x, torch.tensor([height * 3, height * 2], device=model.device),
                           return_log_probs=True)
 log_probs.square().mean().backward()
 assert x.grad is not None and torch.isfinite(x.grad).all()
 print('POST-SAMPLE/LOAD OCR INPUT BACKWARD PASS', height, flush=True)
 print('SPATIAL STYLE RUNTIME PASS', height, 'peak MiB',
-      torch.cuda.max_memory_allocated() / 2**20, 'temporary output', root, flush=True)
+      torch.cuda.max_memory_allocated() / 2**20, 'reserved peak MiB',
+      torch.cuda.max_memory_reserved() / 2**20, 'temporary output', root, flush=True)

```

## tests/test_global_only_integration.py

```diff
--- before/tests/test_global_only_integration.py
+++ after/tests/test_global_only_integration.py
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

## tests/test_reference_local_evidence.py

```diff
--- before/tests/test_reference_local_evidence.py
+++ after/tests/test_reference_local_evidence.py
@@ -211,40 +211,58 @@
                 torch.testing.assert_close(global_code, full[:, :1], rtol=0, atol=0)
                 self.assertIs(feat, features[0])
                 self.assertIs(unmasked_feat, features[0])
                 torch.testing.assert_close(unmasked, encoder._encode_global_style(features[0], None)[0], rtol=0, atol=0)
                 for actual, index in zip(unmasked_maps, (2, 1)):
                     torch.testing.assert_close(actual, features[1][index], rtol=0, atol=0)
                 for actual, optional, index in zip(maps, optional_maps, (2, 1)):
                     expected = features[1][index]
                     valid = torch.arange(expected.size(-1))[None, :] < backbone.feature_lengths(widths)[1][index][:, None]
                     expected = expected.masked_fill(~valid[:, None, None, :], 0.)
                     torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                     torch.testing.assert_close(optional, expected, rtol=0, atol=0)
                 self.assertEqual(sample.shape, (2, 1, 32))
                 torch.testing.assert_close(mu, global_code, rtol=0, atol=0)
                 torch.testing.assert_close(sample, full_sample[:, :1], rtol=0, atol=0)
                 torch.testing.assert_close(mu, full_mu[:, :1], rtol=0, atol=0)
                 torch.testing.assert_close(logvar, full_logvar[:, :1], rtol=0, atol=0)
                 self.assertTrue(torch.isfinite(logvar).all())
                 self.assertGreater((sample - mu).abs().sum().item(), 0.)
 
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
     def test_global_seam_preserves_writer_calculation_and_vae_sampling(self):
         # A changed valid mean/global MLP/clamp or local random sampling fails.
         encoder = self._small_encoder()
         feat = torch.randn(2, 8, 9)
         lengths = torch.tensor([3, 9])
         mask = torch.arange(9)[None, :] < lengths[:, None]
         contextual = encoder.sequence_model(feat, valid_mask=mask)
         pooled = contextual.sum(dim=-1) / lengths[:, None]
         global_hidden = encoder.linear_style(encoder.global_context_norm(pooled)).unsqueeze(1)
         expected_mu = encoder.mu(global_hidden)
         expected_logvar = encoder.logvar(global_hidden).clamp(-14., 4.)
         actual_mu, actual_logvar = encoder._encode_global_style(feat, lengths)
         torch.testing.assert_close(actual_mu, expected_mu, rtol=0, atol=0)
         torch.testing.assert_close(actual_logvar, expected_logvar, rtol=0, atol=0)
         # Also exercise the unmasked valid mean and the clamp boundaries.
         with torch.no_grad():
             encoder.logvar.bias.copy_(torch.tensor([-30., -10., 4., 9.]))
         mu, logvar = encoder._encode_global_style(feat, None)
         expected_hidden = encoder.linear_style(encoder.global_context_norm(
             encoder.sequence_model(feat).mean(dim=-1)

```
