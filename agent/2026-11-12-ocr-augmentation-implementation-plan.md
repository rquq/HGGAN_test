# Paired IAM OCR Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Generate downloadable, paired IAM synthetic datasets with SpiS-GAN, FW-GAN and HiGAN+, then train/evaluate independent TrOCR augmentation arms on two Kaggle T4 GPUs.

**Architecture:** Standalone `ocr_benchmark` branch/folder; a shared immutable manifest feeds isolated native generator subprocesses and a common IAM export. Saved datasets feed the same Hugging Face OCR trainer. A small notebook/queue runs independent jobs per GPU and resumes prior Kaggle outputs.

**Tech Stack:** Python 3.10+, PyTorch 2.7.1/cu118, Transformers 5.9.0, Accelerate 1.13.0, W&B 0.27.0, HDF5, NumPy, Pillow, PyYAML; external source trees pinned independently.

**Spec:** `/home/quq/machineLearning/HTG/HGGAN_test/agent/2026-11-12-ocr-augmentation-benchmark-design.md` (approved November 12, amended below).

## Global constraints

- Only active adapters: `spis_gan`, `fw_gan`, `higanplus`. Do not build own-branch adapters yet.
- User approved one IAM32 comparison, explicitly labeling HiGAN+ native64→export32 conversion. Config must permit this conversion explicitly and report it. Keep optional native64 configuration, without claiming native64 SpiS/FW weights exist.
- Default X=5000 real, N=25000 synthetic, N bounded to [1,25000], fixed paired manifest/ref/text/seeds; real-only control.
- Separate real validation and untouched full IAM test; exclude exact input+label fingerprint groups from training when present in validation/test. No OCR/appearance quality filtering.
- FP32 for both GAN and TrOCR. One job/T4, microbatch2×accumulation16=nominal effective32 for OCR. Version1 fixes the pinned OCR base and FP32; optional FP16/alternate OCR bases are deferred, as documented in the design and run guide.
- Same pretrained-only TrOCR-small-stage1 revision, update budget10000, LR5e-5, weight_decay.01, warmup.05, max_grad_norm1, eval/save500, seed123456. Greedy max_new_tokens64; literal CER, token-edit WER, explicit max-length-normalized NED.
- Relative config paths anchor to the benchmark project root, never launch cwd. `load_config` resolves data/output/generator-config paths and validates before work. Source/checkpoint paths in adapter configs anchor to the same project root.
- Resume binds content hashes and semantic settings, not absolute machine paths. Reject incompatible manifests/checkpoints/sources/training budgets. Atomic files/shards, no partial completion claims.
- No existing GAN code edits, no heavy local training/full evaluation without explicit launch. Test/source artifacts ignored; docs in shared agent. Commit locally per task, root publishes after integration.

## Review focus

- Interrupted shard/export and changed batch size must not duplicate/reseed samples.
- Missing or wrong checkpoint/source geometry must fail strict loading, not run random weights.
- Validation/test shared constant images must not leak through selected real/reference pool; full test remains unchanged.
- Checkpoint resume must retain optimizer/LR/RNG and best checkpoint across session pause, including pause before first validation.
- Kaggle read-only resume input and process failure/clean pause must terminate monitoring correctly and preserve downloadable datasets.

## Stable cross-task interfaces

- `pipeline.config.load_config(path) -> dict`: keys `output_dir`, `dataset` (`train`,`test`,`validation`,`height`,`real_count`,`validation_fraction`), `split_seed`, `generation` (`count`,`seed`,`max_text_length`,`alphabet`,`shard_size`,`models`: name→YAML path), `ocr`, `wandb`. Resolved paths are strings; semantic config can be JSON serialized.
- `pipeline.common`: `read_json(path)`, `atomic_json(path,obj)`, `sha256_file(path)`, `digest_json(obj)`, `bind_identity(directory,payload)` returning identity SHA or rejecting mismatch.
- `pipeline.data.IAMCorpus(path)`: `.texts`, `.writers`, `.widths`, `.height`, `len(corpus)`, `.image(index)` returns source white-paper uint8 H×W; `.close()` and context-manager support.
- `pipeline.data.write_iam(path, records, height, metadata=None)`: records iterable of `(uint8 H×W image, str transcript, int writer)`, streaming atomic export; `validate_iam(path, expected_count=None)` returns summary.
- `pipeline.manifest.prepare(cfg) -> Path`: creates `<output>/prepared/{identity.json,split.json,manifest.jsonl,metadata.json,real_train.hdf5,real_val.hdf5}`. Returned path is prepared directory. Split JSON holds exact original indices, excluded counts, fingerprints. Cases fields: `sample_id` integer0..N-1, `reference_row` original row, `reference_pool_index` row in exported real_train, `reference_text`, `reference_sha256`, `writer`, `target_text`, `seed`. Metadata stores manifest SHA, dataset hashes and split/eligible lexicon counts.
- `pipeline.manifest.read_cases(path) -> list[dict]`: exact ordered manifest, reject malformed IDs/count against metadata when consumed.
- `pipeline.generation.generate(cfg, model_name, device='cpu', time_budget_seconds=None) -> dict`: consumes prepared files and model YAML, produces `<output>/datasets/<model_name>/{synthetic.hdf5,metadata.json,preview.png,dataset.zip,status.json}`. Shards in same directory, resumable. Status `complete` or `paused`. `dataset.zip` includes final HDF5, metadata, manifest and preview; prepared real pool/splits separately downloadable. Call adapters one sample at a time initially to preserve native reference geometry and row-stable randomness.
- Adapter `generators.<name>.load_adapter(model_cfg, device)` returns object with `.metadata` and `.generate(reference_uint8, reference_text, target_text, seed) -> uint8 native-height white-paper image`. Only G/E/B modules. Generic generation worker applies declared height conversion preserving aspect ratio.
- `pipeline.ocr.train(cfg, arm, seed, device='cpu', time_budget_seconds=None) -> dict`; `evaluate(cfg, arm, seed, device='cpu') -> dict`. Arm `real_only` or generator name. Runs `<output>/ocr/<arm>/seed_<seed>/`; persist identity, history CSV, best and last trainer checkpoint, W&B ID, status, final test metrics/predictions.
- `pipeline.metrics.RecognitionMetrics`: `.update(target,prediction)` and `.compute()` return raw counts, fractions and `cer_percent`,`ned_percent`,`wer_percent`,`word_accuracy_percent`.
- `pipeline.runner` owns subprocess job queue, no importing multiple GAN repositories in notebook process. `benchmark.py` CLI exposes `prepare`, `generate --model`, `train --arm --seed`, `evaluate`, `run --stage generation|ocr|all --devices 0 1`, `report`, `preflight`. Common `--config` required; workers use explicit device and optional time budget.

### Task 1: Matched IAM data, manifest and metrics

**Files:** `pipeline/{__init__,config,common,data,manifest,metrics}.py`, `configs/benchmark_{32,64}.yml`. Tests in ignored `tests/test_data_manifest.py`.

- [x] Write and run failing fixture tests: deterministic writer holdout/paired cases; fixed count/ref≠target; duplicate separation; literal repeated letters/case/punctuation; exact CER/WER/NED literals; invalid N/count; export variable widths/polarity and reject missing arrays; changed seed/input resume rejection. Validate full test membership unchanged.
- [x] Implement interfaces above using small atomic HDF5 writes and semantic/hash identities; no large corpus copies to GPU. Export original real_train/real_val and manifest using only selected training texts/references.
- [x] Test interrupted export/metadata, absolute-root portability and immutable reprepare; run whole current test suite with ML Python. Commit scoped source/config only. Record RED/GREEN and limitations in task report.

### Task 2: Three generator-only adapters and downloadable datasets

**Files:** `generators/{__init__,external,spis_gan,fw_gan,higanplus}.py`, `configs/generators/{spis_gan,fw_gan,higanplus}.yml`, `pipeline/generation.py`; ignored `tests/test_generation.py`.

- [x] Test-first deterministic shard/export resume using bounded fake adapter at test injection seam; wrong count/seed/checkpoint/source/config/cache identities reject; interrupted shards ignored/rebuilt; completed bundle reopen and exact labels/IDs.
- [x] Implement real adapters by reusing inspected generator contracts, not full GAN constructors. Pin source URL/revision and explicit model config sections; include optional effective-source patch/hash provenance. Strict load exact G/E/B; native noise32 per-record seed for SpiS/FW, deterministic mean for HiGAN+. Never fall back to random checkpoints.
- [x] Export uint8 with correct native polarity, valid widths, no batch-padding tail. Explicit HiGAN64→32 conversion and label, no thresholding/cleanup. Minimum supported encoder width is model geometry requirement, not symbol-specific handling.
- [x] Test tiny CPU inference on all three actual available checkpoints, visually inspect preview, bounded images only. Standalone .py entrypoints accept config/model YAML. Commit scoped files; record untested CUDA/full25k boundaries.

### Task 3: TrOCR trainer, independent evaluation and W&B/resume

**Files:** `pipeline/{ocr,tracking}.py`, `requirements.txt`; ignored `tests/test_ocr.py`.

- [x] Before code, pin stage1 model full revision from official source; inspect installed Transformers signatures/API. Tests first for labels-pad→-100, no truncation, image conversion, immutable identity and actual tiny-model trainer update/save/resume including LR/RNG, first-save/no-validation pause.
- [x] Implement full-parameter Seq2SeqTrainer pipeline against current pinned libraries, same schedule/effective batch/processor across arms, real-only and mixed rows, streaming aggregate metrics without full logits. Real validation selection only; separate final full real test. Keep greedy generation length-limit diagnostics and corpus counts.
- [x] Time-budget callback saves safe state and cleanly pauses before session limit. Preserve independently validation-selected best as well as resumable last. Do not assume best exists if interrupted early. Check identity before resume and ignore paths-only relocation.
- [x] W&B online/offline/disabled, starts before model load, per-arm/seed/stage IDs, clean finish/failure and no secrets. Trainer state/optimizer etc authoritative; CSV survives resume. Test disabled/offline and native tiny TrOCR architecture, no heavy full OCR training on laptop. Commit scoped sources and pinned requirements.

### Task 4: Kaggle orchestration, preflight, report and notebook

**Files:** `benchmark.py`, `pipeline/{runner,report,preflight}.py`, `kaggle_ocr_benchmark.ipynb`; ignored `tests/test_runner_notebook.py`.

- [x] Test-first CLI subprocesses on temp fixtures: successful/paused/failing jobs terminate monitoring, unique GPU ownership, no accidental training by prepare/generate; prior read-only output import is explicit, incomplete/conflicting state rejects.
- [x] Implement sequential stages with two independent GPU slots, budget shared across queue; stop starting jobs at deadline, preserve clean paused status. Stream per-job logs without endless notebook loop. Final RESULTS table/CSV/JSON compares real-only and completed GAN arms; incomplete rows clearly marked.
- [x] Notebook cells: paths/settings (generator-only or full pipeline), pinned install, explicit model source clones/checkpoint search, current release IAM download32/64, optional previous-output import, prepare, generate, train/eval, report/download. Generation-only mode must stop after exporting all three datasets; output directory exposed in one setting.
- [x] Pin external checkouts under ignored sources; never alter user's sibling repos. Checkpoint paths user/Kaggle input supplied, no checkpoint binaries committed. Restore best+last and datasets from prior save version; API keys via Secrets/env only.
- [x] Parse notebook and execute parameterized bounded cells/CLI end-to-end tiny fixtures. Review source and Kaggle Linux path/config behavior, pinned dependency dry-run/actual import where possible. Commit scoped source/notebook.

## Final integration gate

- [x] Independent whole-branch review, fix important findings and rerun full benchmark-specific suite.
- [x] Bounded all-three native generator smokes; actual tiny TrOCR update→checkpoint→resume→evaluate; export ZIP reopen; Python compile; notebook validate; git diff checks and ignored artifact inventory.
- [x] Write dated user run guide, implementation report and explicit limitations in shared agent. Update design/checklist with user amendments and actual verification.
- [x] Normal push new `ocr_benchmark` branch and scoped `agent` docs; verify both remote SHA values freshly. Main/dev/alpha/classic source trees unchanged.

## Release verification — 2026-11-14

Source release: `f2f8498cabfac095bcc4d5e7586f1b9133a54030` on `ocr_benchmark`.
Independent root test gate:137 passed,12 known dependency warnings,52.79s.
Whole-branch review plus one scoped correction wave: approved, no remaining
Critical/Important findings. Notebook schema/plain-cell compilation and Python
compilation/diff checks passed. Full native Kaggle installation, physical2xT4
execution, full25k generation and completed OCR quality runs remain unmeasured.
Source and the five shared documents are published separately with normal
non-forced pushes; fresh remote SHA checks are the release-completion gate.
