# Checkpoint audit — DEV and alpha, 30 September 2026

## Executive assessment

Both saved checkpoints are numerically intact, and their embedded histories show substantial metric improvement. **alpha wins the saved FID/KID/HWD comparison but is markedly less reliable with low-information style references.** DEV is the safer checkpoint for arbitrary-reference generation, not the better aggregate benchmark checkpoint.

The repeated-glyph failure was reproduced from punctuation references across many writers, and also from a badly blurred whole-word reference. Controlled inference interventions locate it in **global style conditioning**, which reaches both content fusion and GBlock modulation. Separately, **both models have almost reference-independent local-style posterior means**, saturated local-evidence gates, and near-unit local VAE standard deviations. This is strong evidence of local posterior collapse. A separate generator padding-boundary inconsistency was also reproduced.

No model source, configuration, pretrained teacher, GAN checkpoint, or dataset was changed. All interventions were temporary, in-memory diagnostics. Sampling used CPU FP32 with two PyTorch threads; no GPU training or full benchmark evaluation was launched.

## 1. Files and metric histories

Exactly two checkpoint files were present:

| Checkpoint | Epoch | FID ↓ | KID ↓, existing ×100 scale | HWD ↓ |
|---|---:|---:|---:|---:|
| `last_fid_3.0053_dev_branch.pth` | 84 | 3.00527 | 0.090237 | 0.100960 |
| `best_fid_2.6892_alpha.pth` | 73 | 2.68924 | 0.067321 | 0.093945 |

RC is lower by 10.5% FID, 25.4% KID, and 6.95% HWD relative to the saved DEV checkpoint. These are saved results, not newly recomputed scores. KID here is the code's `mean(MMD²) × 100`, not an unscaled KID number.

Both checkpoints contain complete, consecutive epoch histories. Their serialized CSVs agree with their structured `eval_history` tables. Disabled metrics remain empty; I did not invent CER/WER values for those epochs.

Full exports:

- [DEV: all 84 epochs](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/dev_metrics.csv)
- [alpha: all 73 epochs](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/alpha_metrics.csv)
- [Metadata, saved configs, schedulers, integrity checks](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/checkpoint_audit.json)

### Trajectory

| Branch | Epoch | FID | KID | HWD |
|---|---:|---:|---:|---:|
| DEV | 36 | 5.85845 | 0.20284 | 0.17241 |
| DEV | 50 | 4.81859 | 0.17342 | 0.14305 |
| DEV | 60 | 4.19886 | 0.14928 | 0.14103 |
| DEV | 70 | 3.60483 | 0.12370 | 0.12567 |
| DEV | 74 | 3.11335 | 0.08522 | **0.09601** |
| DEV | 76 | 3.16112 | **0.08483** | 0.10239 |
| DEV | 84 | **3.00527** | 0.09024 | 0.10096 |
| RC | 36 | 4.03853 | 0.11426 | 0.15611 |
| RC | 50 | 3.36191 | 0.09622 | 0.10462 |
| RC | 60 | 3.02119 | 0.07962 | 0.10859 |
| RC | 70 | 2.78417 | 0.07629 | 0.10064 |
| RC | 73 | **2.68924** | **0.06732** | 0.09394 |

DEV's best HWD was epoch 74; best KID was epoch 76. RC's best HWD was epoch 53, 0.086912, rather than its best-FID epoch. A metric row in a history does not mean that epoch's weights are recoverable from the current file.

DEV's last-ten-epoch mean was FID 3.18950 / KID 0.09948 / HWD 0.10651. RC's was 2.93031 / 0.08380 / 0.10013. The RC advantage is not limited to a single exceptionally good final row, but this is still one trajectory per experiment, not replicated evidence of a universally superior schedule.

### Training-budget caveat

**Both of these checkpoints are 32px models**, batch size 8. They are not a 32px-versus-64px comparison. Active `networks` and `lib` source files currently match between DEV and `alpha`; DEV additionally has an unused backup directory. The texture-refinement head is absent from these models.

| Setting/state | DEV | RC |
|---|---:|---:|
| `num_critic_train` | 4 | 2 |
| Actual saved G/E optimizer steps | 174,825 | 303,863 |
| Actual D/P optimizer steps | 699,300 | 607,725 |
| Initial G/E, D, P LR | 0.0002 | 0.0002 |
| Saved current LR | 0.000030 | 0.0000458333 |
| Full-LR period | through epoch 36 | through epoch 36 |
| Linear decay duration / floor | 48 epochs / 15% | 48 epochs / 15% |

RC has approximately **1.74× as many G/E updates**, despite fewer epochs. Epoch counts alone are not an equal optimization budget. The saved resume paths also differ; the optimizer counters establish accumulated update counts, not complete provenance of every earlier run.

Current checkout configs extend the run limit from 84 to 100 epochs, without extending the decay schedule. This means additional late epochs at the existing LR floor, not a fresh full-LR learning phase. The histories do not demonstrate that the decay is responsible for the reference-handling failure.

## 2. Checkpoint and load integrity

Passed checks:

- All floating-point tensors in G, G_EMA, E, E_EMA, B, W, R, D, and P are finite.
- Saved G, D, and P optimizer tensors are finite.
- Actual G/E/B state-key sets match instantiated current models before strict loading. The encoder's compatibility fallback did not silently fabricate missing gate weights.
- G/G_EMA custom `stored_var` buffers are positive and finite. Minimum values are about 0.00136–0.00145, not zero/negative variances.
- Frozen B, W, and R tensors are byte-identical across the two checkpoints.
- EMA step counters agree with G/E optimizer steps.
- Both raw and EMA inference reproduce RC's bad-reference failure. Mixing raw/EMA G and E does not remove it.

This rules out several simple explanations: corrupted tensors, a randomly initialized missing encoder module, different frozen teachers between branches, or an EMA-only mishap. It does **not** certify an entire future Kaggle training run or verify GPU-specific kernels; this was an inference/state audit.

Details: [extra integrity checks](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/extra_integrity.json), [raw versus EMA RC](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/raw_vs_ema_alpha.png), [raw versus EMA DEV](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/raw_vs_ema_dev.png).

## 3. Special-reference tests: the observed failure is real

### Matched test design

Real references came from `iam/iam_test_32.hdf5`. Each short reference was paired with a longer real word from the **same writer**. Both checkpoints generated exactly `handwriting`, using their own EMA encoder and generator. Display conversion used fixed contrast and normal black-ink/white-paper polarity; images were enlarged 2× with nearest-neighbor interpolation, not sharpened or contrast-enhanced.

After an initial 16-case exploration, a writer-spread sweep used:

- 24 punctuation references from 24 different writers: eight periods, eight commas, six double quotes, one dash, one apostrophe.
- Ten digit references from ten different writers, covering the available eligible writers in this test split.

The frozen recognizer is identical for both models. Its greedy CER is a **diagnostic for this fixed target and deliberately selected subset**, not the official benchmark CER. “Severe” below means CER > 50%, not a judgment of style fidelity. OCR confuses some plausible `w`/`ur` shapes, so exact-word correctness is less informative here than the very large failure gap.

| Reference/test variant | DEV diagnostic CER | RC diagnostic CER | Severe failures DEV / RC |
|---|---:|---:|---:|
| Punctuation reference → `handwriting` | 9.09% | **78.03%** | 0/24 / **22/24** |
| Longer same-writer reference → `handwriting` | 7.58% | 4.92% | 0/24 / 0/24 |
| Short reference, replace only global token with longer same-writer token | 7.58% | 4.92% | 0/24 / 0/24 |
| Digit reference → `handwriting` | 9.09% | 11.82% | 0/10 / 0/10 |

**It is not accurate to say all digits fail.** The severe issue in these tests is predominantly punctuation/low-information reference handling. RC often generates a repeated, nearly content-independent glyph pattern across the word. It is not simply lower-resolution handwriting.

Galleries: [writers 1–12](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/writer_spread_1.png), [writers 13–24](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/writer_spread_2.png), [digit references](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/writer_spread_3.png). Numeric results: [all stress rows](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/extended_stress_test.csv), [summary](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/extended_stress_summary.csv).

### Causal intervention: global versus local style

In the initial 16-case suite, replacing only the seven local tokens with those from the longer reference changed output by only approximately 0.000005 intensity levels out of 255 in representative DEV cases. It did not fix RC. Replacing **only the global token** produced the readable long-reference result.

This is stronger evidence than merely correlating latent norms with bad images: it changes one conditioning component while keeping the requested text, G, and remaining tokens fixed. It demonstrates where the harmful reference dependence enters; it does not prove the replacement preserves the exact original reference's desired style.

The global token reaches two routes:

1. `StyleConditionedSelfAttention` in the fusion stage.
2. Block-specific global projections and GBlock conditional normalization.

Replacing the global token in fusion alone restores much of the character structure, but leaves degraded strokes. Replacing it only in GBlock conditioning still gives repeated/incorrect characters. Replacing it in both routes restores the readable output. Therefore this is not exclusively a texture-renderer or final-convolution problem.

See [global/local intervention board](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/conditioning_interventions.png), [RC generator-route probes](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/generator_routes_alpha.png), [numeric route probes](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/generator_route_probes.csv).

### Large global vectors are a warning, not the whole explanation

For the 24-writer punctuation sweep, average **pre-cap** global latent norms were approximately 11.18 in DEV and 34.91 in RC. Specific initial RC cases reached 48.85 for a period and 63.29 for a dash; longer same-writer references were much smaller. The code clips both models' global means to norm 9.5.

DEV can still work at the same 9.5 cap at which RC fails. Thus magnitude alone is not the cause: vector direction and the learned sensitivity of G/fusion matter. A temporary radius-3 intervention improved several RC failures, whereas radius 6 did not, but altered the style and was not a reliable or principled repair.

The encoder feeds unnormalized global context through the same MLP that receives normalized local attention output. This is a plausible scale-imbalance mechanism. Moreover, the KL penalty receives the already-clipped mean; beyond the hard cap it does not penalize further radial growth of the raw mean. These are design weaknesses supported by code inspection, not proof of the full training history that produced the collapse.

Source: [StyleEncoder global/local processing and cap](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/module.py), [generator routes](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/BigGAN_networks.py), [fusion](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/fusion.py).

## 4. Important shared problem: local posterior collapse

Across the initial short/long reference pairs:

| Quantity | DEV | RC |
|---|---:|---:|
| Across-reference RMS variation, global token | 0.8121 | 0.9380 |
| Across-reference RMS variation, local tokens | **0.00001256** | **0.00001145** |

The larger writer-spread sweep confirms the pattern rather than contradicting it:

- Local evidence gates are essentially 1 for both good and bad references: minimum about 0.9999907 in DEV and 0.99999988 in RC.
- Local VAE standard deviations are about **0.9958–0.9959 DEV**, **0.99825–0.99845 RC**.
- Local posterior means are small and almost fixed across reference images.
- Swapping local tokens has almost no effect. Replacing locals with random noise also does not repair the bad-global pattern.

Together, almost input-independent local means and variance near the unit prior are strong evidence of **local posterior collapse**. The seven slots retain different fixed identities, but distinct slots are not the same thing as meaningful, reference-dependent local stroke descriptors.

Stage hooks show local attention outputs with large RMS values (~147 DEV, ~299 RC), dominated by a nearly shared direction. After output normalization and the shared style MLP, reference-dependent variation becomes tiny. The reliability gate then passes these almost constant descriptors with essentially full confidence. This does not mean the entire model has collapsed: the global latent still varies considerably and clearly carries useful style.

Likely contributors needing controlled training experiments: all-token KL pressure, strong global bypasses, the shared global/local MLP scale mismatch, and objectives that can be satisfied while ignoring local evidence. The style-cycle target is produced by the same encoder, so a constant local code can satisfy its local portion; fixed slot anchors do not solve that degeneracy. This is not sufficient evidence to select one exact loss weight as the sole cause.

Evidence: [local/global summary](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/conditioning_summary.json), [stage-by-stage reference variation](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/encoder_stage_variation.json), [extended posterior statistics](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/extended_stress_test.csv).

## 5. A separate generator boundary inconsistency

Reference-side padding is effectively invariant in the tested encoder: adding 64 right-padding pixels with the same valid reference length changes output only at tiny floating-point levels, roughly 0.00001–0.00003/255 in the initial checks.

**Target-side padding is not invariant.** Keeping the exact same style and valid target text while adding zero/padding target slots changes valid output pixels:

| Target | DEV mean pixel change /255 | RC mean pixel change /255 |
|---|---:|---:|
| `.`: minimal canvas versus 11 slots | 8.56 | 16.32 |
| `/`: minimal canvas versus 11 slots | 10.11 | 20.60 |
| `1`: minimal canvas versus 11 slots | 10.06 | 17.24 |
| `handwriting`: 11 versus 20 slots | 0.33 | 0.46 |

For the long word, the first half is essentially unchanged; the issue is concentrated near the right boundary. For a one-character output, that boundary region is a large proportion of the entire image.

The generator zeroes padded content tokens initially, but convolutions/conditional offsets can populate invalid feature positions and feed them back into neighboring valid positions. The final eval mask only hides invalid output pixels; it cannot reverse changes already made inside the word. This explains potential batch-composition dependence near short-word boundaries, not the repeated-glyph failure that persists at a fixed target width.

This should be fixed as a **valid-length boundary correctness issue**, not by masking legitimate background within a word. It is distinct from the appearance masking you asked to disable.

Evidence: [padding numbers](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/target_padding_probes.csv), [RC comparison](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/target_padding_alpha.png), [DEV comparison](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/target_padding_dev.png).

## 6. Background artifacts and generated punctuation

The background problem remains visible without any texture-refinement head. DEV sometimes produces character-aligned gray bands or ragged paper boundaries; RC can produce stray lower-edge marks or uneven paper while otherwise drawing legible text.

See [DEV background gallery](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/backgrounds_dev.png) and [RC background gallery](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/backgrounds_alpha.png). These compare the real reference, generated own text, and generated new text at fixed contrast. References were selected for substantial gray border content, so this is a stress gallery, not a random quality estimate.

One selected reference, **test index 4787, label `which`**, is already a severely blurred dark blob in the dataset. RC generates the same repeated-glyph failure from it as from punctuation; a readable same-writer donor rescues the output. DEV remains substantially more legible. This is useful evidence for a **visual-evidence robustness problem**, not a rule that should special-case `.` or `/`.

Some real references also have irregular crop boundaries. Therefore “make all generated backgrounds perfectly rectangular” is not always faithful to the reference distribution. The undesirable defects are extra letter-aligned bands, disconnected specks, and inconsistent paper appearance, not every nonrectangular border.

The collapsed local path and global/content conditioning entanglement are credible contributors to poor texture transfer, but this audit does not quantify how much each increases KID. Fixing global reference handling did change background appearance in the interventions; a cleaner-looking image is not automatically a lower-KID image.

Generating punctuation is a separate weakness from using punctuation as a reference. With ordinary word references, requests for `/`, `.`, `,`, and `-` sometimes yield elongated smears or the wrong symbol, while digits are substantially more recognizable. The test set and training set contain no standalone `/` in the inspected labels; standalone punctuation examples themselves are often stretched/full-height-looking. A slash is supported by the alphabet, but support in an alphabet is not evidence of adequate training coverage.

See [DEV requested-symbol gallery](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/target_symbols_dev.png), [RC requested-symbol gallery](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/target_symbols_alpha.png), [diagnostic OCR transcriptions](/home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/target_symbol_ocr.csv). The dataset was not altered or filtered.

## 7. Why good benchmark scores coexist with these failures

The saved configuration uses `use_rand_corpus: false`. In style-guided validation:

1. E receives the real test image.
2. G receives that image's own transcription.
3. Output is resized to that sample's original width.
4. Distribution metrics compare all resulting images with real images.

Consequently, a period reference is normally asked to generate a period—not `handwriting`. The cross-content failure demonstrated here is not directly exercised by the reported metric protocol. CER/WER were disabled as well.

This is a coverage limitation, not evidence that the stored scores are fake. Preserve the benchmark protocol for fair comparisons, and add a **separate, clearly labeled cross-content stress evaluation**, ideally on a fixed development split for model selection rather than repeatedly tuning on the benchmark test examples used for this audit.

FID/KID preprocessing does not segment out paper within the valid word. It retains that appearance, replaces/pads external canvas regions, crops/pads width to four times image height, and resizes for Inception. Thus background rendering can affect the features and scores. For these 32px data the visible FID/KID window is 128 pixels; 1,567 of 25,766 real test samples are wider than that, so not every far-right defect is seen by those metrics. This should be documented, not silently changed if it is the agreed benchmark protocol.

Sources: [validation generation](/home/quq/machineLearning/HTG/HGGAN_test/dev/networks/model.py), [FID/KID preprocessing and scaling](/home/quq/machineLearning/HTG/HGGAN_test/dev/metric/val_metrics.py).

## 8. What I would do next

### Keep both checkpoints

- Keep RC as the stronger saved aggregate benchmark result.
- Keep DEV as the more robust reference-conditioned model.
- Do not overwrite either with an inference-only norm clamp or background cleanup. That would change the model behavior and style without repairing the learned representation.

### Priority 1 — repair the shared style representation

Normalize global/local encoder inputs consistently and remove the scale mismatch before increasing model size. For the next controlled architecture experiment, make local descriptors deterministic rather than forcing all seven through a stochastic unit-Gaussian bottleneck; retain a stochastic global latent if random-style generation is needed. The local path must measurably respond to visual evidence. Reconsider/remove the evidence gate if it remains saturated; do not keep adding fixed anchors and call that anti-collapse.

This is a proposed training change, not a validated performance gain. It changes the training objective and should be tested in a controlled new run. It does not require changing benchmark data or hardcoding symbol IDs/resolution.

### Priority 2 — correct generator valid-boundary behavior

Make the same text/style produce the same valid image regardless of other samples' padding widths, with explicit regression checks at both 32 and 64px. Keep valid-length attention/loss/eval semantics. Do not erase legitimate gray paper to make this test pass.

### Priority 3 — protect cross-content readability

The current style-transfer CTC coefficient decays from 0.5 to 0.08. Before adding new texture losses, test whether maintaining a stronger fixed cross-content content constraint prevents the semantic collapse while retaining style. Do not simply make OCR dominate every output: evaluate style preservation and ordinary-reference quality alongside the stress cases. The actual `rare_lexicon` is all training transcriptions, not a rarity-balanced set; its 15% sampling option should not be mistaken for uniform rare-symbol coverage.

### Schedule decision

The present evidence favors critic ratio 2 for aggregate scores per dataset epoch, but it does not establish that ratio 2 caused the reference failure or that ratio 4 prevents it generally. RC received substantially more G/E updates. Do not change LR, critic ratio, encoder, and texture objectives simultaneously and attribute the result to one of them.

Extending DEV to 100 epochs at the current floor may modestly improve distribution metrics, but is not a convincing repair for an almost constant local posterior. RC might also improve its aggregate score further while retaining the bad-reference behavior. I would prioritize the representation/boundary fixes and the separate stress evaluation over another blind epoch extension.

### What not to do

- No `if reference == '.'`/digit-length fallback.
- No resolution-specific exception to make 32px look better.
- No background removal or benchmark preprocessing changes just to lower KID.
- No claim that lowering the global norm at inference preserves writer style.
- No claim that one favorable metric row proves robust generalization.

## 9. Reproduction and limitations

Scripts in this folder: `audit_checkpoints.py`, `probe_conditioning.py`, `extended_probes.py`, `probe_generator_routes.py`, `probe_padding.py`. They read checkpoints and source from the local workspace and write only diagnostic outputs here. They are outside the DEV and RC Git worktrees; both worktrees remained clean.

Run with the existing HiGANplus environment and a two-thread CPU cap, e.g.:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/quq/miniconda3/envs/HiGANplus/bin/python /home/quq/machineLearning/HTG/chkpoints/30_9_2026/analysis/extended_probes.py
```

The audit uses current matching architecture code and checkpoint config snapshots. It does not reconstruct every historical training commit, rerun FID/KID/HWD, establish statistical significance across seeds, test all characters/styles, or prove that a proposed repair improves the benchmark. Its strongest results are the repeated failure reproduction, matched global-token interventions, local posterior measurements, and padding-width counterexample.
