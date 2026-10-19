# Independent OCR implementation checks — 2026-10-19

Exactly CER and WER were added. Both are fractions; percentages below multiply by 100. This is a bounded transfer diagnostic, not downstream recognizer training or a published benchmark reproduction.

## Existing best checkpoints, same final protocol-v2 manifest

| Checkpoint | Cases | CER | WER | Total time |
|---|---:|---:|---:|---:|
| MAIN best_fid_2.5769, epoch 77 | 320 | 25.97% | 55.31% | 42.1 s |
| ALPHA best_fid_2.7509, epoch 84 | 320 | 19.13% | 55.00% | 27.3 s |
| Matched real reference images | 320 | 22.21% | 46.56% | cached after first run |

The second evaluation reused real-image predictions. These unequal runtimes are not an architectural speed comparison. Peak total allocated CUDA memory: MAIN 300.5 MiB, ALPHA 295.9 MiB, in standalone inference on RTX 4060 laptop. Normal GAN training retains other models/optimizer states, so this is not a complete training VRAM measurement.

Evaluator: microsoft/trocr-small-handwritten, pinned revision b4648cfa171985a6745f37ddd637e98c0da958ac, 61,596,672 parameters, 245,933,041-byte weights, FP32 and greedy decoding. The encoder pooler is unused by this decoder; missing-pooler initialization warnings do not indicate missing decoder weights.

TrOCR has substantial real-word-domain error here, especially punctuation/narrow crops. MAIN's short-reference corpus CER is dominated by a recognizer insertion loop; inspect predictions and images before attributing the whole gap to the GAN. These scores are relative diagnostics, not strong absolute legibility evidence or proof that ALPHA is better overall.

**Neither checkpoint tests the newly configured global-only ablation.** They retain their saved local-token architectures and different training histories. New MAIN/control versus ALPHA/global-only requires fresh GAN training with unchanged pretrained teachers.

Outputs: /home/quq/machineLearning/HTG/reports/checkpoints_19_10_2026/main_ocr_final and /home/quq/machineLearning/HTG/reports/checkpoints_19_10_2026/alpha_ocr_final each contain metrics.json, metrics.csv, predictions.csv and samples.png. The shared manifest is main_ocr_final/manifest.json. Earlier ocr_320/smoke outputs are preserved but superseded by these final protocol-v2 outputs for OOV classification.

## Runtime choice

- Normal 32px and 64px configs: 320 fixed cases, OCR batch 2, each epoch.
- Local bug-test config: 40 cases, batch 1, first eligible validation only per process.
- Low-memory CPU fallback: once per process. Missing cached weights/dependencies or optional CUDA OCR OOM leave scores empty with a message, never a teacher fallback or fake zero.
- New notebook preparation cell downloads the pinned independent model to cache before training. No pretrain replacement or automatic GAN job launch.

## Verification

- Eleven focused CPU regression tests passed, including transcript repetitions, CTC single collapse, manifests/OOV, split-local writer IDs, RNG/mode restoration, cache identity and missing/OOM paths.
- Real CPU GAN one-update smoke passed for MAIN32, MAIN64, ALPHA32 and ALPHA64, with existing teachers, real data, optimizers and EMA.
- Both existing best checkpoints loaded strictly and completed bounded CUDA inference.
- The real MAIN32 training-validation hook passed a six-case CPU integration check: only CER/WER returned, exact modes/RNG restored, second once-per-session call skipped. Its fresh one-update random GAN outputs are a plumbing test, not quality evidence.
- Common source parity, Python AST/notebook syntax and git diff checks passed; temporary tests remain ignored.
- Full native Kaggle/T4 training, final ablation quality and long-run stability are not verified by these tests.
