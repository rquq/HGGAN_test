# Review package: task-2
Uncommitted task changes; HEAD and index are untouched.

## networks/module.py

```diff
--- before/networks/module.py
+++ after/networks/module.py
@@ -284,390 +284,305 @@
 
 
 class _GradientReverse(torch.autograd.Function):
     @staticmethod
     def forward(ctx, x, scale):
         ctx.scale = scale
         return x.view_as(x)
 
     @staticmethod
     def backward(ctx, grad_output):
         return -ctx.scale * grad_output, None
 
 
 def _gradient_reverse(x, scale):
     return _GradientReverse.apply(x, scale)
 
 
 class StyleEncoder(nn.Module):
     def __init__(self, style_dim=32, in_dim=256, init='N02', num_style_tokens=None,
                  backbone_channels=(64, 128, 256), n_class=None, content_grl=1.0,
-                 local_attention_residual_init=0.25,
-                 local_evidence_gate_init=0.75,
-                 local_evidence_gate_hidden=64,
                  # These names are kept as aliases because older DEV YAML files
                  # used them.  Previously they landed in **kwargs and were
                  # silently ignored, making the printed configuration untrue.
                  num_local_queries=None, query_dim=None, heads=4,
-                 cross_attn_dropout=0.0, local_attention_gate_init=None,
+                 cross_attn_dropout=0.0,
                  feature_scales=None, **kwargs):
         super(StyleEncoder, self).__init__()
         if n_class is None:
             n_class = get_alphabet_size('all')
         if kwargs:
             unknown = ', '.join(sorted(str(key) for key in kwargs))
             raise TypeError(f'Unknown StyleEncoder option(s): {unknown}')
         if (num_style_tokens is not None and num_local_queries is not None
                 and int(num_style_tokens) != int(num_local_queries) + 1):
             raise ValueError(
                 'num_style_tokens must equal num_local_queries + 1'
             )
         if num_style_tokens is None:
             num_style_tokens = (
                 int(num_local_queries) + 1
                 if num_local_queries is not None else 8
             )
         if query_dim is not None and int(query_dim) != int(in_dim):
             raise ValueError(
                 f'query_dim ({query_dim}) must equal in_dim ({in_dim}); '
                 'separate query projections are not implemented.'
             )
-        if local_attention_gate_init is not None:
-            local_attention_residual_init = local_attention_gate_init
         if not 1 <= int(heads) <= int(in_dim):
             raise ValueError('heads must be in [1, in_dim]')
         if int(in_dim) % int(heads) != 0:
             raise ValueError('in_dim must be divisible by heads')
         if not 0.0 <= float(cross_attn_dropout) < 1.0:
             raise ValueError('cross_attn_dropout must be in [0, 1)')
         if feature_scales is not None:
             feature_scales = tuple(feature_scales)
             if not feature_scales:
                 raise ValueError('feature_scales cannot be empty')
             if len(feature_scales) > len(backbone_channels):
                 raise ValueError(
                     'feature_scales cannot request more maps than '
                     'backbone_channels'
                 )
             if any(float(scale) <= 0 for scale in feature_scales):
                 raise ValueError('feature_scales must contain positive values')
             available_scales = tuple(2 ** i for i in range(len(backbone_channels)))
             if (len(set(feature_scales)) != len(feature_scales)
                     or any(scale not in available_scales for scale in feature_scales)):
                 raise ValueError(
                     f'feature_scales must be distinct members of {available_scales}'
                 )
         self.style_dim = style_dim
         self._in_dim = in_dim
         self.num_style_tokens = int(num_style_tokens)
         self.num_local_queries = self.num_style_tokens - 1
         self.attention_heads = int(heads)
         self.cross_attn_dropout = float(cross_attn_dropout)
         # The backbone exposes three feature maps.  Keep all of them by
         # default. Scale IDs 1/2/4 select the corresponding shallow/deeper maps;
         # [2, 4] must not silently select the first two maps.
         self.feature_scales = feature_scales
         self.feature_indices = tuple(
             available_scales.index(scale) for scale in feature_scales
         ) if feature_scales is not None else tuple(range(len(backbone_channels)))
         self.content_grl = content_grl
-        self.local_attention_residual_init = float(local_attention_residual_init)
-        self.local_evidence_gate_init = float(local_evidence_gate_init)
-        self.local_evidence_gate_hidden = int(local_evidence_gate_hidden)
         if num_style_tokens < 1:
             raise ValueError('num_style_tokens must be at least 1')
-        if not 0.0 < self.local_attention_residual_init < 1.0:
-            raise ValueError(
-                'local_attention_residual_init must be strictly between 0 and 1'
-            )
-        if not 0.0 < self.local_evidence_gate_init < 1.0:
-            raise ValueError(
-                'local_evidence_gate_init must be strictly between 0 and 1'
-            )
-        if self.local_evidence_gate_hidden < 1:
-            raise ValueError('local_evidence_gate_hidden must be positive')
 
         self.linear_style = nn.Sequential(
             nn.Linear(in_dim, in_dim),
             nn.LeakyReLU(),
             nn.Linear(in_dim, in_dim),
             nn.LeakyReLU(),
         )
         # Global pooled features have a different scale from attention tokens.
-        # Normalize them before the shared global-style projector so short or
+        # Normalize them before the global-style projector so short or
         # low-ink references cannot drive an extreme conditioning vector.
         self.global_context_norm = nn.LayerNorm(
             in_dim, elementwise_affine=False
         )
         self.mu = nn.Linear(in_dim, style_dim)
         self.logvar = nn.Linear(in_dim, style_dim)
         self.sequence_model = GatedStyleContext(in_dim)
 
         # Build every trainable projection before the optimizer is created. The old
         # forward-time replacement silently left new parameters unoptimised.
         self.proj_layers = nn.ModuleList([
             nn.Conv2d(backbone_channels[index], in_dim, kernel_size=1)
             for index in self.feature_indices
         ])
         for layer in self.proj_layers:
             nn.init.normal_(layer.weight, 0.0, 0.02)
             if layer.bias is not None:
                 nn.init.zeros_(layer.bias)
 
         # Token zero is an explicit global style summary. The remaining compact
         # query set captures local stroke details without a 32x32 content-rich code.
         query_count = self.num_local_queries
         style_query_init = torch.empty(1, query_count, in_dim)
         if query_count:
             # Orthogonal rows start as distinct local stroke slots while matching
             # the old N(0, 0.02) per-component scale in expectation.
             nn.init.orthogonal_(style_query_init[0])
             style_query_init.mul_(0.02 * (in_dim ** 0.5))
         self.style_queries = nn.Parameter(style_query_init)
         # Slot queries are learned after orthogonal initialization.
 
-        attention_gate_logit = torch.logit(torch.tensor(
-            self.local_attention_residual_init
-        )).item()
-        self.local_attention_gate_logits = nn.Parameter(
-            torch.full((in_dim,), attention_gate_logit)
-        )
         self.style_cross_attn = nn.MultiheadAttention(
             embed_dim=in_dim, num_heads=self.attention_heads,
-            dropout=self.cross_attn_dropout, batch_first=True
+            dropout=self.cross_attn_dropout, batch_first=True, bias=False,
         )
         # Affine-free norms add no checkpoint state. They prevent the frozen
         # backbone's very different feature scales from dominating attention.
         self.style_key_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
         self.style_query_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
-        self.local_output_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
-        # Predict local-token reliability from visual evidence, not from a
-        # character ID, transcription length, image height, or assumed glyph
-        # aspect ratio.  The fourth descriptor is the masked feature variance,
-        # so the same rule applies to other resolutions, scripts, and datasets.
-        self.local_evidence_gate = nn.Sequential(
-            nn.Linear(in_dim * 4, self.local_evidence_gate_hidden),
-            nn.SiLU(),
-            nn.Linear(self.local_evidence_gate_hidden, 1),
-        )
+        # Local values describe visual deviations only. No affine offset or
+        # output normalization can manufacture evidence from a uniform map.
+        self.local_projection = nn.Linear(in_dim, style_dim, bias=False)
         self.content_probe = nn.Linear(style_dim, n_class)
 
         if init != 'none':
             init_weights(self, init)
-        nn.init.zeros_(self.local_evidence_gate[-1].weight)
-        nn.init.constant_(
-            self.local_evidence_gate[-1].bias,
-            torch.logit(torch.tensor(
-                self.local_evidence_gate_init
-            )).item(),
-        )
         nn.init.constant_(self.logvar.weight, 0.)
         nn.init.constant_(self.logvar.bias, -10.)
-
-    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
-                              missing_keys, unexpected_keys, error_msgs):
-        # Texture ablations should remain possible with checkpoints created
-        # before the evidence gate existed.  Only the newly introduced gate is
-        # initialized locally; every historical model key remains strict.
-        for name, value in self.local_evidence_gate.state_dict().items():
-            key = prefix + 'local_evidence_gate.' + name
-            if key not in state_dict:
-                state_dict[key] = value.detach().clone()
-        super()._load_from_state_dict(
-            state_dict, prefix, local_metadata, strict,
-            missing_keys, unexpected_keys, error_msgs,
-        )
 
     @staticmethod
     def global_token(style_tokens):
         return style_tokens[:, 0]
 
     def predict_content(self, style_tokens, reverse=True):
         style_for_probe = style_tokens
         if reverse:
             style_for_probe = _gradient_reverse(style_for_probe, self.content_grl)
         return self.content_probe(style_for_probe)
 
+    def _encode_global_style(
+        self, feat: torch.Tensor, feat_len: torch.Tensor | None,
+    ) -> tuple[torch.Tensor, torch.Tensor]:
+        """The original writer-summary path, shared by full/global-only calls."""
+        feat_mask = (
+            torch.arange(feat.size(-1), device=feat.device)[None, :] < feat_len[:, None]
+        ) if feat_len is not None else None
+        contextual = self.sequence_model(feat, valid_mask=feat_mask)
+        if feat_len is None:
+            global_context = contextual.mean(dim=-1)
+        else:
+            global_context = contextual.sum(dim=-1) / feat_len.unsqueeze(1).to(feat.dtype)
+        global_style = self.linear_style(
+            self.global_context_norm(global_context)
+        ).unsqueeze(1)
+        return self.mu(global_style), self.logvar(global_style).clamp(-14.0, 4.0)
+
+    def _build_local_attention_inputs(
+        self, selected_feats: list[torch.Tensor],
+        map_lengths: list[torch.Tensor] | None,
+    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
+        """Position selects evidence; values carry only per-map deviations.
+
+        map_lengths retains the full backbone map order. selected_feats and
+        proj_layers instead follow feature_indices, including subset/reordering.
+        """
+        keys, values, padding_masks = [], [], []
+        for feature_index, projection, feature in zip(
+            self.feature_indices, self.proj_layers, selected_feats,
+        ):
+            width_mask = (
+                torch.arange(feature.size(-1), device=feature.device)[None, :]
+                < map_lengths[feature_index][:, None]
+            ) if map_lengths is not None else None
+            if width_mask is not None:
+                feature = feature.masked_fill(~width_mask[:, None, None, :], 0.0)
+            # A pointwise projection commutes with average pooling. Keep the
+            # same four height rows without projecting the full feature map.
+            pooled = projection(F.adaptive_avg_pool2d(feature, (4, feature.size(-1))))
+            height, width = pooled.shape[-2:]
+            visual = self.style_key_norm(pooled.flatten(2).transpose(1, 2))
+            position = get_2d_sinusoidal_embeddings(
+                height, width, self._in_dim, pooled.device,
+            ).reshape(1, height * width, self._in_dim)
+            if width_mask is None:
+                visual_mean = visual.mean(dim=1, keepdim=True)
+                map_keys = visual + position
+                map_values = visual - visual_mean
+            else:
+                valid = width_mask[:, None, :].expand(-1, height, -1).reshape(feature.size(0), -1)
+                visual = visual.masked_fill(~valid.unsqueeze(-1), 0.0)
+                visual_mean = visual.sum(dim=1, keepdim=True) / valid.sum(
+                    dim=1, keepdim=True,
+                ).clamp_min(1).unsqueeze(-1).to(visual.dtype)
+                map_keys = (visual + position).masked_fill(~valid.unsqueeze(-1), 0.0)
+                map_values = (visual - visual_mean).masked_fill(~valid.unsqueeze(-1), 0.0)
+                padding_masks.append(~valid)
+            keys.append(map_keys)
+            values.append(map_values)
+        padding_mask = torch.cat(padding_masks, dim=1) if padding_masks else None
+        return torch.cat(keys, dim=1), torch.cat(values, dim=1), padding_mask
+
     def forward(self, img, img_len, cnn_backbone=None, ret_feats=False,
-                vae_mode=False, ret_backbone_feat=False, backbone_features=None):
+                vae_mode=False, ret_backbone_feat=False, backbone_features=None,
+                global_only=False):
         feat, all_feats = (
             cnn_backbone(img, ret_feats=True, x_lens=img_len)
             if backbone_features is None else backbone_features
         )
         if self.feature_indices and max(self.feature_indices) >= len(all_feats):
             raise RuntimeError(
                 f'StyleEncoder requested feature map indices {self.feature_indices}, '
                 f'but the backbone returned {len(all_feats)} maps.'
             )
         selected_all_feats = [all_feats[index] for index in self.feature_indices]
         if len(self.proj_layers) != len(selected_all_feats):
             raise RuntimeError(
                 f'StyleEncoder expected {len(self.proj_layers)} backbone feature maps, '
                 f'but received {len(selected_all_feats)}. Set EncModel.feature_scales '
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
-            feat_mask = torch.arange(feat.size(-1), device=img.device)[None, :] < feat_len[:, None]
         else:
-            feat_mask, feat_len, map_lengths = None, None, None
-        feat_mask_f = feat_mask.unsqueeze(1).to(feat.dtype) if feat_mask is not None else 1.0
-        feat_m = self.sequence_model(feat, valid_mask=feat_mask)
-        if feat_mask is None:
-            global_context = feat_m.mean(dim=-1)
-            feature_variation = (
-                feat_m - global_context.unsqueeze(-1)
-            ).square().mean(dim=-1)
-        else:
-            global_context = feat_m.sum(dim=-1) / feat_len.unsqueeze(1).to(feat.dtype)
-            centered = (
-                feat_m - global_context.unsqueeze(-1)
-            ) * feat_mask_f
-            feature_variation = centered.square().sum(dim=-1) / feat_len.unsqueeze(1).to(
-                feat.dtype
-            )
-
-        feat_m_trans = feat_m.transpose(1, 2)
-        pe_1d = get_1d_sinusoidal_embeddings(
-            feat_m_trans.size(1), self._in_dim, feat_m_trans.device
-        )
-        feat_m_trans = feat_m_trans + pe_1d.unsqueeze(0)
-        if feat_mask is not None:
-            feat_m_trans = feat_m_trans * feat_mask.unsqueeze(-1).to(feat_m_trans.dtype)
-
-        spatial_tokens = [feat_m_trans]
-        padding_masks = [~feat_mask] if feat_mask is not None else []
+            feat_len, map_lengths = None, None
+
+        global_mu, global_logvar = self._encode_global_style(feat, feat_len)
+        # Optional returned maps preserve their original channels and selected
+        # order even when local retrieval is skipped by objective-only calls.
         masked_all_feats = []
-        for feature_index, proj_layer, feature in zip(self.feature_indices, self.proj_layers, selected_all_feats):
+        for feature_index, feature in zip(self.feature_indices, selected_all_feats):
             width_mask = (
-                torch.arange(feature.size(-1), device=img.device)[None, :]
+                torch.arange(feature.size(-1), device=feature.device)[None, :]
                 < map_lengths[feature_index][:, None]
             ) if map_lengths is not None else None
-            width_mask_f = (
-                width_mask[:, None, None, :].to(feature.dtype)
-                if width_mask is not None else 1.0
-            )
-            feature_masked = feature * width_mask_f
-            masked_all_feats.append(feature_masked)
-            # A pointwise affine projection commutes with average pooling.
-            # Project four height rows instead of the entire feature map.
-            feature_pooled = proj_layer(F.adaptive_avg_pool2d(
-                feature_masked, (4, feature.size(-1))
-            ))
-            height, width = feature_pooled.shape[-2:]
-            pe_2d = get_2d_sinusoidal_embeddings(
-                height, width, self._in_dim, feature_pooled.device
-            )
-            feature_pooled = feature_pooled + pe_2d.permute(2, 0, 1).unsqueeze(0)
-            if width_mask is not None:
-                feature_pooled = feature_pooled * width_mask[:, None, None, :].to(
-                    feature_pooled.dtype
-                )
-            spatial_tokens.append(feature_pooled.flatten(2).transpose(1, 2))
-            if width_mask is not None:
-                padding_masks.append(
-                    ~width_mask[:, None, :].expand(-1, height, -1).reshape(feature.size(0), -1)
-                )
-
-        style_keys = self.style_key_norm(torch.cat(spatial_tokens, dim=1))
-        key_padding_mask = torch.cat(padding_masks, dim=1) if padding_masks else None
-
-        batch_size = img.size(0)
-        style_queries = self.style_queries.expand(batch_size, -1, -1)
-        if style_queries.size(1):
-            style_queries = self.style_query_norm(style_queries)
+            masked_all_feats.append(
+                feature.masked_fill(~width_mask[:, None, None, :], 0.0)
+                if width_mask is not None else feature
+            )
+
+        if not global_only and self.num_local_queries:
+            style_keys, style_values, key_padding_mask = self._build_local_attention_inputs(
+                selected_all_feats, map_lengths,
+            )
+            style_queries = self.style_query_norm(self.style_queries.expand(img.size(0), -1, -1))
             local_attended, _ = self.style_cross_attn(
                 query=style_queries,
                 key=style_keys,
-                value=style_keys,
+                value=style_values,
                 key_padding_mask=key_padding_mask,
                 need_weights=False,
             )
-            # Cross-attention weights can grow substantially during long GAN
-            # runs. Normalize the attended evidence before mixing it with the
-            # learned slot queries, and keep local evidence out of the global
-            # MLP whose negative activations were suppressing its variation.
-            local_attended = self.style_key_norm(local_attended)
-            attention_strength = torch.sigmoid(
-                self.local_attention_gate_logits
-            ).view(1, 1, -1)
-            local_style = self.local_output_norm(
-                style_queries + attention_strength * local_attended
-            )
+            local_mu = self.local_projection(local_attended)
+            local_mu = local_mu - local_mu.mean(dim=1, keepdim=True)
         else:
-            local_style = style_queries
-
-        global_style = self.linear_style(
-            self.global_context_norm(global_context)
-        ).unsqueeze(1)
-
-        if local_style.size(1):
-            expanded_global = global_style.expand_as(local_style)
-            normalized_global = F.layer_norm(
-                expanded_global, (expanded_global.size(-1),)
-            )
-            normalized_local = F.layer_norm(
-                local_style, (local_style.size(-1),)
-            )
-            normalized_variation = F.layer_norm(
-                feature_variation, (feature_variation.size(-1),)
-            ).unsqueeze(1).expand_as(local_style)
-            evidence_descriptor = torch.cat([
-                normalized_global,
-                normalized_local,
-                torch.abs(normalized_local - normalized_global),
-                normalized_variation,
-            ], dim=-1)
-            # Learned visual evidence can select any mixture, including a
-            # global fallback for an uninformative local reference.
-            local_reliability = torch.sigmoid(
-                self.local_evidence_gate(evidence_descriptor)
-            )
-        else:
-            local_reliability = local_style.new_ones(
-                batch_size, 0, 1
-            )
-        global_mu = self.mu(global_style)
-
-        if local_style.size(1):
-            local_data_mu = self.mu(local_style)
-            local_mu = global_mu + local_reliability * (local_data_mu - global_mu)
-        else:
-            local_mu = self.mu(local_style)
+            local_mu = global_mu.new_empty(img.size(0), 0, self.style_dim)
         style_tokens_mu = torch.cat([global_mu, local_mu], dim=1)
 
         if vae_mode:
             # Keep stochastic variation in the global writer code. Local
             # tokens describe observed strokes and remain deterministic; their
             # unit-Gaussian KL pressure had driven their posterior to the prior.
-            global_logvar = torch.clamp(
-                self.logvar(global_style), min=-14.0, max=4.0
-            )
             global_std = torch.exp(0.5 * global_logvar)
             global_sample = torch.randn_like(global_std) * global_std + global_mu
             style_tokens_sampled = torch.cat([global_sample, local_mu], dim=1)
             local_logvar = torch.zeros_like(local_mu)
             logvar = torch.cat([global_logvar, local_logvar], dim=1)
             style_tokens = (style_tokens_sampled, style_tokens_mu, logvar)
         else:
             style_tokens = style_tokens_mu
 
         if ret_backbone_feat and not ret_feats:
             ret_feats = True
         if ret_backbone_feat:
             return style_tokens, masked_all_feats, feat
         if ret_feats:
             return style_tokens, masked_all_feats
         return style_tokens
 
 
 class MaskedAttentiveStatsPool(nn.Module):
     """Pool writer features while ignoring right-padding.

```

## configs/gan_iam_32.yml

```diff
--- before/configs/gan_iam_32.yml
+++ after/configs/gan_iam_32.yml
@@ -215,34 +215,30 @@
 
 
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
-  local_attention_residual_init: 0.5
-  local_evidence_gate_init: 0.75
-  local_evidence_gate_hidden: 64
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

## configs/gan_iam_64.yml

```diff
--- before/configs/gan_iam_64.yml
+++ after/configs/gan_iam_64.yml
@@ -221,32 +221,29 @@
 
 
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
-  local_attention_residual_init: 0.5
-  local_evidence_gate_init: 0.75
-  local_evidence_gate_hidden: 64
   feature_scales: [1, 2, 4]
 
 
 WidModel:
   n_writer: 339
   in_dim: 256
   pool: 'attentive_stats'
   attention_dim: 128
   init: 'none'

```

## configs/gan_iam_local.yml

```diff
--- before/configs/gan_iam_local.yml
+++ after/configs/gan_iam_local.yml
@@ -232,33 +232,30 @@
   rnn_depth: 2
   bidirectional: true
 
 
 StyBackbone:
   resolution: 16
   max_dim: 256
   in_channel: 1
   init: 'N02'
   dropout: 0.0
   norm: 'bn'
 
 
 EncModel:
   style_dim: 32
   in_dim: 256
   num_style_tokens: 8
   backbone_channels: [64, 128, 256]
   n_class: 81
   content_grl: 1.0
-  local_attention_residual_init: 0.25
-  local_evidence_gate_init: 0.75
-  local_evidence_gate_hidden: 64
   init: 'none'
 
 
 WidModel:
   n_writer: 339
   in_dim: 256
   # Must match the W+B teacher exported by configs/wid_iam.yml.
   pool: 'attentive_stats'
   attention_dim: 128
   init: 'none'

```

## tests/test_compact_style_context.py

```diff
--- before/tests/test_compact_style_context.py
+++ after/tests/test_compact_style_context.py
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

## tests/test_luna_review.py

```diff
--- before/tests/test_luna_review.py
+++ after/tests/test_luna_review.py
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

## tests/test_reference_local_evidence.py

```diff
--- before/tests/test_reference_local_evidence.py
+++ after/tests/test_reference_local_evidence.py
@@ -0,0 +1,274 @@
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
