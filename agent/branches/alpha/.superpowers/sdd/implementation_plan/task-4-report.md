# Task 4 verification report

Status: integration present and verified; ready for root review. No product source was edited during this verification handoff. Task 5 synchronization remains separate.

## Scope and contract

The saved Task 4 changes in `networks/model.py` and `networks/module.py` cover wrapped global-only fake re-encoding, global content probing, and centered random-prior producer paths. The ignored integration tests cover both fake re-encoding routes, bypass of local attention/projection in global-only mode, transfer B-feature use, and centering through training, sampling, unguided generation, interpolation, and random evaluation paths. Existing global objective scaling and VAE behavior are covered by the CPU regressions. The bounded runtime checks exercise optimizer ownership/updates for the new E/G parameters, full E/G and EMA state restoration, optimizer moments/steps, scheduler state and LR, strict legacy GAN rejection, and OCR input backpropagation after sampling/loading.

The prior implementation worker reported a baseline RED run with 11 routing failures plus the optional-feature allocation regression. I did not reconstruct the baseline modules in this final verification pass; the current source and assertions were left unchanged.

## CPU regressions

Command:

```text
PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v /home/quq/miniconda3/envs/HiGANplus/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Result: 71 tests passed, exit 0. `unittest` reports 6.374 seconds; outer wall time was 22.03 seconds and peak host RSS was 982,864 KiB. Discovery includes the Task 4 integration tests and the earlier Task 1–3 regression files. Full output: `test_artifacts/local_style_simplification_14_10/task4_cpu_unittest.log`.

## Sequential CUDA runtime proofs

The 32px proof completed before checking GPU availability and starting 64px. Both runs used `WANDB_MODE=disabled`; the checker sets the CUDA process memory fraction to 0.3, two PyTorch CPU threads, batch size 2, 16 real samples, 8 D/P updates, 4 G/E updates, and R1 interval 1. GPU checks were serialized; no full training or metric evaluation was run.

| Command | Result | Runtime resources |
|---|---|---|
| `WANDB_MODE=disabled /home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/check_spatial_training.py 32` | Exit 0. Actual bounded train loop and sampling passed; all losses/gradients finite FP32; optimizer steps and E/G parameter updates passed; short-reference/long-target tail passed; E/G + EMA + optimizer/scheduler/progress roundtrip passed; strict legacy GAN rejection passed; OCR input backward passed. | 39.22s wall; peak host RSS 2,571,444 KiB; CUDA allocated/reserved peaks 472.1/514 MiB. |
| `WANDB_MODE=disabled /home/quq/miniconda3/envs/HiGANplus/bin/python -B tests/check_spatial_training.py 64` | Same checks passed at 64px, exit 0. | 36.51s wall; peak host RSS 3,090,624 KiB; CUDA allocated/reserved peaks 802.4/886 MiB. |

Full outputs: `test_artifacts/local_style_simplification_14_10/task4_cuda32.log` and `test_artifacts/local_style_simplification_14_10/task4_cuda64.log`.

## Archived E strict-load boundary

After the GPU checks, the complete E state from `/home/quq/machineLearning/HTG/chkpoints/14_10_2026/best_fid_2.4463_alpha.pth` was loaded on CPU and passed directly to `StyleEncoder.load_state_dict(..., strict=True)`. The expected strict `RuntimeError` was observed: missing `local_projection.weight`; unexpected `local_attention_gate_logits`, `local_evidence_gate.*`, and `style_cross_attn` bias keys. No partial load or conversion was attempted. This confirms the old E state is not silently accepted by the new architecture; fresh GAN training remains required.

The check itself exited 0 after validating the expected missing/unexpected keys. Wall time was 10.87 seconds and peak host RSS was 943,952 KiB. Full output: `test_artifacts/local_style_simplification_14_10/task4_archived_e_strict_cpu.log`.

## Hygiene and limits

`git diff --check` passed. All four execution logs and this report are confirmed ignored (`test_artifacts/` and `.superpowers/sdd/`). Existing uncommitted source/config changes were preserved; nothing was staged or committed.

These are CPU regression and bounded smoke proofs, not a full training run, metric evaluation, quality comparison, or evidence of long-run optimization stability. Root review of objective gradients, route behavior, prior call sites, and logs remains the final boundary.
