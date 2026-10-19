### Task 4: Integrate global objectives and every random-prior producer

**Files:**
- Modify: `networks/model.py` training, sampling, unguided validation and manual interpolation/random generation paths.
- Modify: `networks/module.py`, existing `StyleEncoder.predict_content` to select token zero before the existing GRL/probe.
- Modify: ignored `tests/test_local_style_contract.py`, `tests/check_training_background.py`, `tests/check_spatial_training.py`.
- Create: ignored `tests/test_global_only_integration.py`.

**Interfaces:**
- Consume Task 1's helpers and Task 2's wrapped E `forward(..., global_only=True)`.
- Keep full reference encoding for generation; random-fake and transfer-fake re-encodings use `self.models.E(...)`, never an unwrapped custom forward. Transfer re-encoding still returns features for the existing writer/context losses.
- `predict_content(style_tokens, reverse=True)` returns `(B,1,C)` from token zero. GRL remains on the global code; content-probe parameters remain in the existing optimizer.

- [ ] **Write failing integration assertions:** a forwarding wrapper records both fake re-encoding calls with `global_only=True`; local MHA/projection hooks must stay silent, while B is called exactly once per fake re-encoding. Returned transfer B features are the ones used for writer/context. Prior-producer tests exercise train D/G, `sample_images`, unguided `image_generator`, `eval_interp` and `eval_rand` with centered locals and unchanged globals.
- [ ] **Run:** `/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/test_global_only_integration.py`. Expected: failures for omitted `global_only` and uncentered producer paths.
- [ ] **Replace the two all-slot L1 calls** with Task 1's global L1 helper using the original `encoder.num_style_tokens`, not the global-only result's size `1`. Use Task 1's BCE helper for global probe logits. Keep global KL `/N`, loss names, `lambda_*`, CTC/writer/reconstruction/context code and logging weights unchanged.
- [ ] **Center priors at producer boundaries:** both training `.z.sample_()` results, sampling `eval_z`, unguided `torch.randn`, interpolation endpoints and `eval_rand` outputs. Leave sampler metadata/RNG ownership in the original Distribution objects; pass transformed base tensors downstream. Reference E output is already centered. Do not alter rank-2 noise in external baseline tools.
- [ ] **Extend checkpoint smoke assertions:** optimizer ownership and actual updates for E `style_queries`, projections, MHA and `local_projection`; E/G full-state and EMA equivalence; optimizer moments/steps; scheduler state and post-load LR exactness; post-load OCR input backward. Legacy GAN strict loading must raise missing/unexpected-key errors rather than initialize silently.
- [ ] **Run CPU regression files** from Tasks 1–4, then the existing relevant padding/objective/runtime unit files identified in the current tree. Exclude old checkpoint-analysis scripts from regression counts; record every actual command/result.
- [ ] **Run bounded CUDA proofs sequentially after checking `nvidia-smi`:** `/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/check_spatial_training.py 32`, then `/home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/check_spatial_training.py 64`. Use memory fraction `0.3`, two CPU threads, batch `2`, 16 real samples, 8 D/P and 4 G/E updates, R1 interval `1`, sampling and checkpoint roundtrip; never run both GPU proofs concurrently. Expected: all finite FP32, no R1/SDPA/cuDNN backward failure, correct steps and valid tails.
- [ ] **Review boundary:** root reviews objective gradients, wrapped-forward routing, all random call sites and fresh logs. If memory is unavailable, record that limitation rather than declaring GPU validation passed. Do not commit.

