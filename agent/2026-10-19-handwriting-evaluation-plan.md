# Handwriting Evaluation and Local Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. User approved implementation in this session; execute natively without delegated agents.

**Goal:** Provide a controlled MAIN-local versus ALPHA-global experiment and bounded independent OCR evaluation with exactly CER/WER.

**Architecture:** Preserve official distribution metrics. Reuse the existing validation ledger and EMA inference; isolate manifest generation and external OCR in a small metric module usable from validation and a standalone checkpoint CLI. OCR never supplies training gradients.

**Tech Stack:** PyTorch FP32, NumPy, HDF5, Pillow, existing edit distance, Transformers TrOCR-small.

**Spec:** 2026-10-19-handwriting-evaluation-design.md, amended by user: ONLY CER/WER, cadence chosen from measured cost.

## Global Constraints

- MAIN and ALPHA only; no GPU training launch, no automatic Git commits/pushes.
- Preserve FID/KID/HWD, best-FID saving, datasets, teachers, notebooks' existing launch/monitor behavior.
- CPU load and bounded streaming inference; FP32; diagnostic batch <=4.
- Global-only retains divisor 8; fresh GAN retrain required, teacher retrain not required.
- No NED/accuracy/new teacher-score series. Independent CER/WER cannot silently use the training teacher.

## Review Focus

Completion record: all five tasks below are implemented. Eleven focused CPU tests, four real CPU GAN one-update checks (MAIN/ALPHA at 32/64px), strict saved-checkpoint inference, source parity and notebook syntax checks passed. Final independent OCR runs used the same 320-case manifest: MAIN 42.1s, ALPHA 27.3s (cached real calibration), on RTX 4060. Native Kaggle and full GAN CUDA training remain unverified. See OCR_EVALUATION.md in this folder for commands and cadence. The unchecked items below preserve the original task checklist, not outstanding work.

- Repeated letters and blank-separated repeats: literal labels decode without CTC collapse.
- One-token encoder and G: no local module instantiated/executed; global gradients remain live.
- Missing external evaluator: clear disabled state, never a fabricated zero or training-teacher fallback.
- Evaluation side effects: RNG, modes, EMA selection restored; bounded samples and word lengths.
- Vocabulary/split provenance: fixed manifest, original writer IDs, explicit exposure and missing-cell reporting.

### Task 1: Transcript decoding regression

Files: lib/alphabet.py and networks/model.py in both branches; ignored focused tests.
- [ ] Reproduce `letter -> leter` with existing decoder.
- [ ] Add `decode_labels` literal conversion, preserving old `decode` CTC API.
- [ ] Use literal conversion for OCR targets/already-collapsed predictions in GAN and pretrain validation.
- [ ] Test repeated letters/digits, padding, perfect CER/WER and one CTC collapse.

### Task 2: Global-only architecture

Files: networks/BigGAN_networks.py, networks/module.py, networks/model.py, normal/local GAN YAMLs in both branches.
- [ ] Add `use_local_style` generator flag default true; omit local refinement module when false.
- [ ] Omit encoder local parameters when num_local_queries=0 while retaining returned contextual feature maps.
- [ ] Keep MAIN current four-local defaults; mirror common code to ALPHA and set global-only with divisor 8.
- [ ] Test forward/backward at heights 32 and 64, zero/local params, shapes and strict control-checkpoint load.

### Task 3: Independent bounded OCR evaluator

Files: new metric/ocr_eval.py and tools/evaluate_ocr.py, mirrored to both branches.
- [ ] Implement deterministic dataset/text-exposure manifest and small paired-reference probe.
- [ ] Load TrOCR-small independently, greedy decoding, real calibration and only CER/WER; stream batches and export example rows and identity metadata.
- [ ] Standalone loads checkpoint G/E/B only, supports bounded CPU/CUDA, explicit evaluator preparation/download and manifest reuse.
- [ ] Test real/CTC text policies, four cells, unavailable OOV, inference flags/RNG preservation, missing weights and batch bounds.
- [ ] Benchmark tiny real-image and checkpoint subsets conservatively; record size, time, memory, real-domain limitations.

### Task 4: Training and notebook integration

Files: networks/model.py, GAN YAMLs, minimal notebook evaluator-preparation cell if required.
- [ ] Add cadence from cost evidence; one bounded check at checkpoint-session startup or first evaluation if too expensive for epochs.
- [ ] Add only independent CER/WER using existing columns; keep metric identity/revision and detailed subset outputs separate from full benchmark scores.
- [ ] Missing weights never break an unrelated GAN run; clear startup note and empty values.
- [ ] Verify RNG/modes, startup resume row merging and no stale score reuse; preserve two-job launch/stop behavior.

### Task 5: Final gates

- [ ] Run syntax, focused regression suite and bounded strict-load inference at both heights where feasible.
- [ ] Check branch diff parity except experimental configs and no test artifacts staged/tracked.
- [ ] Document commands, measured evaluator cost, quality/provenance caveats, and unverified native Kaggle/full training.
