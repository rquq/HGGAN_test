# Review package: task-5
Uncommitted task changes; HEAD and index are untouched.

## ../main/networks/utils.py

```diff
--- before/../main/networks/utils.py
+++ after/../main/networks/utils.py
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

## ../main/networks/loss.py

```diff
--- before/../main/networks/loss.py
+++ after/../main/networks/loss.py
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

## ../main/networks/module.py

```diff
--- before/../main/networks/module.py
+++ after/../main/networks/module.py
@@ -284,390 +284,306 @@
 
 
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
-        style_for_probe = style_tokens
+        style_for_probe = style_tokens[:, :1]
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
+            feat_len, map_lengths = None, None
+
+        global_mu, global_logvar = self._encode_global_style(feat, feat_len)
+        # Optional returned maps preserve their original channels and selected
+        # order even when local retrieval is skipped by objective-only calls.
+        masked_all_feats = []
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
+
+        if not global_only and self.num_local_queries:
+            style_keys, style_values, key_padding_mask = self._build_local_attention_inputs(
+                selected_all_feats, map_lengths,
             )
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
-        masked_all_feats = []
-        for feature_index, proj_layer, feature in zip(self.feature_indices, self.proj_layers, selected_all_feats):
-            width_mask = (
-                torch.arange(feature.size(-1), device=img.device)[None, :]
-                < map_lengths[feature_index][:, None]
-            ) if map_lengths is not None else None
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

## ../main/networks/fusion.py

```diff
--- before/../main/networks/fusion.py
+++ after/../main/networks/fusion.py
@@ -1,212 +1,216 @@
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
 
 from .utils import ensure_dim3
 
 
 def _mask_tokens(sequence, mask):
     if mask is None:
         return sequence
     return sequence.masked_fill(~mask.unsqueeze(-1), 0.0)
 
 
-class StyleConditionedSelfAttention(nn.Module):
-    """Mix target positions, conditioned by the global writer style."""
-
-    def __init__(self, d_model, style_dim, nhead=4, attn_dim=128,
+class ContentSelfAttention(nn.Module):
+    """Mix target positions without reference-style conditioning."""
+
+    def __init__(self, d_model, nhead=4, attn_dim=128,
                  max_seq_len=32):
         super().__init__()
         if nhead < 1 or attn_dim < 1 or attn_dim % nhead:
             raise ValueError('attn_dim must be positive and divisible by nhead')
         if max_seq_len < 1:
             raise ValueError('max_seq_len must be positive')
         self.nhead = nhead
         self.head_dim = attn_dim // nhead
         self.max_seq_len = max_seq_len
         self.content_norm = nn.LayerNorm(d_model, elementwise_affine=False)
-        self.style_norm = nn.LayerNorm(style_dim, elementwise_affine=False)
-        self.style_mod = nn.Linear(style_dim, d_model * 2)
         self.qkv = nn.Linear(d_model, attn_dim * 3, bias=False)
         self.attn_out = nn.Linear(attn_dim, d_model, bias=False)
         self.relative_position_bias = nn.Parameter(
             torch.zeros(nhead, max_seq_len * 2 - 1)
         )
 
     def reset_stability_parameters(self):
-        nn.init.normal_(self.style_mod.weight, 0.0, 0.01)
-        nn.init.zeros_(self.style_mod.bias)
         nn.init.xavier_uniform_(self.qkv.weight)
         nn.init.xavier_uniform_(self.attn_out.weight)
         nn.init.zeros_(self.relative_position_bias)
 
-    def forward(self, content_seq, global_style, mask=None):
+    def forward(self, content_seq, mask=None):
         batch_size, length, _ = content_seq.shape
-        shift, scale = self.style_mod(self.style_norm(global_style)).chunk(2, dim=-1)
         conditioned = self.content_norm(content_seq)
-        conditioned = conditioned * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
         qkv = self.qkv(conditioned).view(
             batch_size, length, 3, self.nhead, self.head_dim
         ).permute(2, 0, 3, 1, 4)
         query, key, value = qkv.unbind(0)
         positions = torch.arange(length, device=content_seq.device)
         relative = (positions[:, None] - positions[None, :]).clamp(
             -self.max_seq_len + 1, self.max_seq_len - 1
         ) + self.max_seq_len - 1
         bias = self.relative_position_bias[:, relative].unsqueeze(0).to(query.dtype)
         if mask is not None:
             bias = bias.expand(batch_size, -1, -1, -1).masked_fill(
                 ~mask[:, None, None, :], float('-inf')
             )
         attended = F.scaled_dot_product_attention(
             query, key, value, attn_mask=bias, dropout_p=0.0, is_causal=False
         )
         attended = attended.transpose(1, 2).reshape(batch_size, length, -1)
         return _mask_tokens(self.attn_out(attended), mask)
 
 
-class ReferenceStyleAttention(nn.Module):
-    """Retrieve local reference evidence independently in each attention head.
-
-    Target identity and neighbours are already represented in content queries.
-    Keys match normalized local tokens. Values carry local deviations from
-    the normalized global style, so a global fallback adds no local correction.
+class SpatialStyleRefinement(nn.Module):
+    """Refine early spatial G features with local evidence relative to global.
+
+    Every valid spatial location is a query; local slots are the only keys.
+    A direct learned residual preserves the feature path without a strength cap.
     """
 
-    def __init__(self, d_model, style_dim, nhead=4, attn_dim=128):
+    def __init__(self, d_model, style_dim, nhead=4, attn_dim=128,
+                 residual_init=0.25):
         super().__init__()
         if nhead < 1 or attn_dim < 1 or attn_dim % nhead:
             raise ValueError('attn_dim must be positive and divisible by nhead')
+        if d_model < 1 or style_dim < 1 or residual_init <= 0:
+            raise ValueError('feature/style dimensions and residual_init must be positive')
+        self.d_model = d_model
+        self.residual_init = float(residual_init)
         self.nhead = nhead
         self.head_dim = attn_dim // nhead
-        self.content_norm = nn.LayerNorm(d_model, elementwise_affine=False)
+        self.feature_norm = nn.LayerNorm(d_model, elementwise_affine=False)
         self.style_norm = nn.LayerNorm(style_dim, elementwise_affine=False)
         self.q_proj = nn.Linear(d_model, attn_dim, bias=False)
         self.k_proj = nn.Linear(style_dim, attn_dim, bias=False)
         self.v_proj = nn.Linear(style_dim, attn_dim, bias=False)
         self.out_proj = nn.Linear(attn_dim, d_model, bias=False)
+        self.reference_scale = nn.Parameter(torch.full((d_model,), self.residual_init))
+        self.reset_stability_parameters()
 
     def reset_stability_parameters(self):
         for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
             nn.init.xavier_uniform_(layer.weight)
-
-    def forward(self, content_seq, local_style_seq, global_style, mask=None):
-        batch_size, length, _ = content_seq.shape
+        with torch.no_grad():
+            self.reference_scale.fill_(self.residual_init)
+
+    def forward(self, feature, style_seq, x_lens=None):
+        if feature.ndim != 4 or feature.size(1) != self.d_model:
+            raise ValueError('spatial style refinement requires BCHW features with matching channels')
+        style_seq = ensure_dim3(style_seq)
+        if style_seq.size(0) != feature.size(0) or style_seq.size(1) < 2:
+            raise ValueError('spatial refinement requires one global and at least one local token per image')
+        batch_size, channels, height, width = feature.shape
+        spatial_seq = feature.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
+        mask = None
+        if x_lens is not None:
+            lengths = x_lens.to(feature.device).long().clamp(1, width)
+            width_mask = torch.arange(width, device=feature.device)[None, :] < lengths[:, None]
+            mask = width_mask[:, None, :].expand(-1, height, -1).reshape(batch_size, height * width)
+        spatial_seq = _mask_tokens(spatial_seq, mask)
+        length = spatial_seq.size(1)
+        local_style_seq = style_seq[:, 1:]
         style = self.style_norm(local_style_seq)
-        # Subtract in the same space as the existing value projection. Do not
-        # normalize the difference again: weak evidence must remain weak.
-        value_style = style - self.style_norm(global_style).unsqueeze(1)
-        query = self.q_proj(self.content_norm(content_seq)).view(
+        value_style = local_style_seq
+        query = self.q_proj(self.feature_norm(spatial_seq)).view(
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
-        return _mask_tokens(self.out_proj(attended), mask)
+        refined = _mask_tokens(
+            spatial_seq + self.reference_scale * self.out_proj(attended), mask
+        )
+        return refined.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
 
 
 class LocalGatedFeedForward(nn.Module):
-    """Joint content/style transformation with neighbouring-position context."""
+    """Content transformation with neighbouring-position context."""
 
     def __init__(self, d_model, hidden_dim):
         super().__init__()
         self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
         self.expand = nn.Linear(d_model, hidden_dim * 2)
         self.depthwise = nn.Conv1d(
             hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim
         )
         self.project = nn.Linear(hidden_dim, d_model)
 
     def reset_stability_parameters(self):
         nn.init.xavier_uniform_(self.expand.weight)
         nn.init.zeros_(self.expand.bias)
         nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='linear')
         nn.init.zeros_(self.depthwise.bias)
         nn.init.xavier_uniform_(self.project.weight)
         nn.init.zeros_(self.project.bias)
 
     def forward(self, content_seq, mask=None):
         value, gate = self.expand(self.norm(content_seq)).chunk(2, dim=-1)
         # Clear affine biases before width mixing so extra batch padding cannot
         # leak into the final valid characters.
         value = _mask_tokens(value, mask)
         value = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
         transformed = self.project(value * F.silu(gate))
         return _mask_tokens(transformed, mask)
 
 
-class StyleContentAttentionFusion(nn.Module):
-    """Target context -> reference retrieval -> joint local mixing.
+class ContentContext(nn.Module):
+    """Target self-attention and local mixing before constructing the G seed.
 
     Each pre-normalized sublayer has one direct learnable residual scale.
     Nonzero initialization enables immediate learning, without sigmoid
     ceilings, nested output gates, forced-uniform attention or RMS caps.
     The same topology serves every supported image height.
     """
 
-    def __init__(self, d_model, style_dim, nhead=4, attn_dim=128,
+    def __init__(self, d_model, nhead=4, attn_dim=128,
                  ffn_dim=None, max_seq_len=32, residual_init=0.25):
         super().__init__()
         if residual_init <= 0:
             raise ValueError('residual_init must be positive')
         hidden_dim = ffn_dim if ffn_dim is not None else d_model * 3
         if hidden_dim < 1:
             raise ValueError('ffn_dim must be positive')
         self.residual_init = float(residual_init)
-        self.content_context = StyleConditionedSelfAttention(
-            d_model, style_dim, nhead=nhead, attn_dim=attn_dim,
+        self.content_attention = ContentSelfAttention(
+            d_model, nhead=nhead, attn_dim=attn_dim,
             max_seq_len=max_seq_len,
-        )
-        self.reference_attention = ReferenceStyleAttention(
-            d_model, style_dim, nhead=nhead, attn_dim=attn_dim,
         )
         self.local_mixer = LocalGatedFeedForward(d_model, hidden_dim)
         self.context_scale = nn.Parameter(torch.full((d_model,), self.residual_init))
-        self.reference_scale = nn.Parameter(torch.full((d_model,), self.residual_init))
         self.local_scale = nn.Parameter(torch.full((d_model,), self.residual_init))
         self.reset_stability_parameters()
 
     @property
     def residual_scales(self):
-        return torch.stack((self.context_scale, self.reference_scale, self.local_scale))
-
-    def reset_stability_parameters(self):
-        self.content_context.reset_stability_parameters()
-        self.reference_attention.reset_stability_parameters()
+        return torch.stack((self.context_scale, self.local_scale))
+
+    def reset_stability_parameters(self):
+        self.content_attention.reset_stability_parameters()
         self.local_mixer.reset_stability_parameters()
         with torch.no_grad():
-            for scale in (self.context_scale, self.reference_scale, self.local_scale):
+            for scale in (self.context_scale, self.local_scale):
                 scale.fill_(self.residual_init)
 
-    def forward(self, content_seq, style_seq, y_lens=None):
-        style_seq = ensure_dim3(style_seq)
-        if style_seq.size(1) < 2:
-            raise ValueError('fusion requires one global token and at least one local token')
+    def forward(self, content_seq, y_lens=None):
         if content_seq.size(1) < 1:
             raise ValueError('fusion requires at least one target position')
         mask = None
         if y_lens is not None:
             lengths = y_lens.to(content_seq.device).long().clamp(1, content_seq.size(1))
             positions = torch.arange(content_seq.size(1), device=content_seq.device)
             mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
         content_seq = _mask_tokens(content_seq, mask)
         content_seq = _mask_tokens(
-            content_seq + self.context_scale * self.content_context(
-                content_seq, style_seq[:, 0], mask=mask
-            ), mask,
-        )
-        content_seq = _mask_tokens(
-            content_seq + self.reference_scale * self.reference_attention(
-                content_seq, style_seq[:, 1:], style_seq[:, 0], mask=mask
+            content_seq + self.context_scale * self.content_attention(
+                content_seq, mask=mask
             ), mask,
         )
         return _mask_tokens(
             content_seq + self.local_scale * self.local_mixer(content_seq, mask=mask), mask
         )

```

## ../main/networks/model.py

```diff
--- before/../main/networks/model.py
+++ after/../main/networks/model.py
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
@@ -2516,41 +2512,41 @@
                     g_style = (
                         lambda_info * info_loss
                         + getattr(self.opt.training, 'lambda_style_cycle', 1.0)
                           * style_cycle_loss
                         + getattr(self.opt.training, 'lambda_content_adv', 0.02)
                           * content_adv_loss
                     )
                     g_context = context_weight * ctx_loss
                     g_kl = float(getattr(self.opt.training, 'lambda_kl', 0.1)) * kl_loss
                     g_loss = (
                         g_adv + g_ctc + g_writer + g_recn
                         + g_style + g_context + g_kl
                     )
 
                     g_loss.backward()
                     torch.nn.utils.clip_grad_norm_(
                         chain(self.models.G.parameters(), self.models.E.parameters()),
                         getattr(self.opt.training, 'grad_clip', 5.0),
                     )
                     generator = self.unwrap_model(self.models.G)
-                    fusion_scales = generator.style_content_mix.residual_scales.detach()
+                    fusion_scales = generator.fusion_scales.detach()
                     self.averager_meters.update_many({
                         'g_total': g_loss,
                         'g_adv': g_adv,
                         'g_adv_global': adv_loss,
                         'g_adv_patch': weighted_adv_loss_patch,
                         'g_ctc': g_ctc,
                         'g_ctc_rand': fake_ctc_loss_rand,
                         'g_ctc_style': fake_ctc_loss_style,
                         'g_writer': g_writer,
                         'g_recn': g_recn,
                         'g_style': g_style,
                         'g_info': info_loss,
                         'g_style_cycle': style_cycle_loss,
                         'g_content_adv': content_adv_loss,
                         'g_context': g_context,
                         'g_kl': g_kl,
                         'fusion_strength': fusion_scales.mean(),
                         'fusion_scale_min': fusion_scales.min(),
                         'fusion_scale_max': fusion_scales.max(),
                     })

```

## ../main/networks/BigGAN_networks.py

```diff
--- before/../main/networks/BigGAN_networks.py
+++ after/../main/networks/BigGAN_networks.py
@@ -1,30 +1,30 @@
 # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 # SPDX-License-Identifier: MIT
 import functools
 
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
 
 from . import BigGAN_layers as layers
-from .fusion import StyleContentAttentionFusion
+from .fusion import ContentContext, SpatialStyleRefinement
 from networks.utils import init_weights, _len2mask
 
 # Architectures for G
 # Attention is passed in in the format '32_64' to mean applying an attention
 # block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
 def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
     arch = {}
 
     # Keep one architecture at every supported resolution: four conditioned
     # stages and the same channel schedule. Resolution controls geometry only.
     # At 32px the image reaches its target after stage three, so stage four is
     # a learned same-resolution refinement. At 64px that stage performs the
     # final 2x upsample. There is no dataset- or metric-specific capacity path.
     arch[32] = {'in_channels': [ch * item for item in [8, 4, 2, 1]],
                 'out_channels': [ch * item for item in [4, 2, 1, 1]],
                 'upsample': [(2, 1), (2, 2), (2, 2), None],
                 'resolution': [8, 16, 32, 32],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(2, 6)}}
 
@@ -135,43 +135,48 @@
                                                   eps=self.SN_eps)
         else:
             self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
             self.which_linear = nn.Linear
 
         if self.bn_linear=='SN':
             bn_linear = functools.partial(self.which_linear, bias=False)
         else:
             bn_linear = nn.Linear
 
         self.which_bn = functools.partial(layers.ccbn,
                                           which_linear=bn_linear,
                                           cross_replica=self.cross_replica,
                                           mybn=self.mybn,
                                           input_size=self.z_chunk_size,
                                           norm_style=self.norm_style,
                                           eps=self.BN_eps)
 
         self.filter_linear = self.which_linear(self.embed_dim,
                                         self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
-        self.style_content_mix = StyleContentAttentionFusion(
-            self.embed_dim, self.style_dim, nhead=fusion_nhead,
+        self.content_context = ContentContext(
+            self.embed_dim, nhead=fusion_nhead,
             attn_dim=fusion_attn_dim, ffn_dim=fusion_ffn_dim,
+            residual_init=fusion_residual_init,
+        )
+        self.spatial_style = SpatialStyleRefinement(
+            self.arch['out_channels'][0], self.style_dim,
+            nhead=fusion_nhead, attn_dim=fusion_attn_dim,
             residual_init=fusion_residual_init,
         )
 
         self.bssp = BlockSpecificStyleProjection(style_dim=self.style_dim, num_blocks=len(self.arch['in_channels']), style_chunk_size=self.z_chunk_size, which_linear=self.which_linear)
 
         # self.blocks is a doubly-nested list of modules, the outer loop intended
         # to be over blocks at a given resolution (resblocks and/or self-attention)
         # while the inner loop is over a given block
         self.blocks = []
         for index in range(len(self.arch['out_channels'])):
             upsample_scale = self.arch['upsample'][index]
             self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                            out_channels=self.arch['out_channels'][index],
                                            which_conv1=self.which_conv,
                                            which_conv2=self.which_conv,
                                            which_bn=self.which_bn,
                                            activation=self.activation,
                                            upsample=(functools.partial(
                                                F.interpolate,
                                                scale_factor=upsample_scale,
@@ -179,125 +184,133 @@
 
             if self.arch['attention'][self.arch['resolution'][index]]:
                 self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
 
         # Turn self.blocks into a ModuleList so that it's all properly registered.
         self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
 
         # output layer: batchnorm-relu-conv.
         # Consider using a non-spectral conv here
         self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                     cross_replica=self.cross_replica,
                                                     mybn=self.mybn),
                                           self.activation,
                                           self.which_conv(self.arch['out_channels'][-1], input_nc))
 
         # Initialize weights. Optionally skip init for testing.
         if self.init != 'none':
             init_weights(self, self.init)
         # General initialization touches fusion Linear weights; restore its
         # identity-like nonzero residual handoffs afterwards.
-        self.style_content_mix.reset_stability_parameters()
+        self.content_context.reset_stability_parameters()
+        self.spatial_style.reset_stability_parameters()
 
     def forward(self, z, y, y_lens):
         # Distribution is a reusable sampler container, not an activation
         # type. Strip its Tensor subclass at the model boundary so its custom
         # .to() method cannot propagate through G and downstream networks.
         if type(z) is not torch.Tensor:
             z = z.as_subclass(torch.Tensor)
 
-        # Only the explicit global token may bypass character-level fusion.
-        # Local tokens must travel through the aligned fusion path.
+        # Global style conditions every GBlock. Local evidence is retrieved
+        # only after the first block has formed an early spatial representation.
         ys = self.bssp(z[:, 0])
 
         content = self.text_embedding(y).float().to(y.device)
-        y_mixed = self.style_content_mix(
-            content, z, y_lens=y_lens
-        )
+        y_mixed = self.content_context(content, y_lens=y_lens)
         token_positions = torch.arange(y.size(1), device=y.device).unsqueeze(0)
         valid_tokens = (token_positions < y_lens.unsqueeze(1)).unsqueeze(-1)
         y_mixed = y_mixed * valid_tokens.to(y_mixed.dtype)
         h = self.filter_linear(y_mixed) * valid_tokens.to(y_mixed.dtype)
 
         # Reshape - when y is not a single class value but rather an array of classes, the reshape is needed to create
         # a separate vertical patch for each input.
         h = h.view(h.size(0), h.shape[1] * self.bottom_width, self.bottom_height, -1)
         h = h.permute(0, 3, 2, 1)
 
         # Loop over blocks
         len_scale = 1
         x_lens = y_lens * self.bottom_width
         for index, blocklist in enumerate(self.blocks):
             stage_input_lens = (x_lens * len_scale).clamp(
                 min=1, max=h.size(-1)
             )
             upsample_scale = self.arch['upsample'][index]
             width_scale = upsample_scale[1] if upsample_scale is not None else 1
             stage_output_lens = (stage_input_lens * width_scale).clamp(
                 min=1
             )
             # Second inner loop in case block has multiple layers
             for block in blocklist:
                 if isinstance(block, layers.Attention):
                     h = block(h, x_len=stage_output_lens)
                 else:
                     h = block(
                         h, y=ys[index], x_lens=stage_input_lens,
                         out_x_lens=stage_output_lens,
                     )
+            if index == 0:
+                h = self.spatial_style(h, z, x_lens=stage_output_lens)
             # Conditional normalization and convolutions can populate padded
             # columns. Clear them after each stage so those activations cannot
             # feed back across the valid word boundary in a later convolution.
             valid_width = stage_output_lens.clamp(max=h.size(-1))
             stage_mask = (
                 torch.arange(h.size(-1), device=h.device)[None, :]
                 < valid_width[:, None]
             )
             h = h * stage_mask[:, None, None, :].to(h.dtype)
             if upsample_scale is not None:
                 len_scale *= upsample_scale[1]
 
         # The final conditional normalization can also inject values into the
         # right padded area. Mask before the last convolution so it cannot
         # perturb valid edge pixels; mask its final tail as before below.
         output_lens = torch.div(
             y_lens * h.size(-2), 2, rounding_mode='trunc'
         ).clamp(min=1)
         output = self.output_layer[0](h, valid_lens=output_lens)
         output = self.output_layer[1](output)
         output_lens = output_lens.clamp(max=output.size(-1))
         output_mask = _len2mask(
             output_lens.int(), output.size(-1), torch.float32
         ).to(output.device).unsqueeze(1).unsqueeze(1)
         output = torch.tanh(
             self.output_layer[2](output * output_mask.to(output.dtype))
         )
 
         # Mask blanks
         if not self.training:
             mask = output_mask.detach()
             output = output * mask + (mask - 1)
 
         return output
 
+    @property
+    def fusion_scales(self):
+        return torch.cat((
+            self.content_context.residual_scales.flatten(),
+            self.spatial_style.reference_scale,
+        ))
+
     def fusion_strength(self):
-        return self.style_content_mix.residual_scales.mean()
+        return self.fusion_scales.mean()
 
     def _info_attention(self):
         attn_index = -1
         for index in range(len(self.arch['out_channels'])):
             if self.arch['attention'][self.arch['resolution'][index]]:
                 attn_index = index
 
         if attn_index == -1:
             return []
 
         attn_layer = self.blocks[attn_index][-1]
         out = []
         if hasattr(attn_layer, 'attn1') and hasattr(attn_layer, 'attn2'):
             for l in [attn_layer.attn1, attn_layer.attn2]:
                 out.append({'out': getattr(l, '_vis_out', None), 'gamma': l.gamma.item()})
         elif hasattr(attn_layer, 'gamma'):
             out.append({'gamma': attn_layer.gamma.item()})
         return out
 
 

```

## ../main/configs/gan_iam_32.yml

```diff
--- before/../main/configs/gan_iam_32.yml
+++ after/../main/configs/gan_iam_32.yml
@@ -219,34 +219,31 @@
 
 
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
   # All three backbone maps keep old 32px checkpoints shape-compatible.  A
   # future speed ablation may shorten this list to select shallow maps only.
   feature_scales: [1, 2, 4]
 
 
 WidModel:
   n_writer: 339
   in_dim: 256
   pool: 'attentive_stats'
   attention_dim: 128
   init: 'none'

```

## ../main/configs/gan_iam_64.yml

```diff
--- before/../main/configs/gan_iam_64.yml
+++ after/../main/configs/gan_iam_64.yml
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

## ../main/configs/gan_iam_local.yml

```diff
--- before/../main/configs/gan_iam_local.yml
+++ after/../main/configs/gan_iam_local.yml
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
