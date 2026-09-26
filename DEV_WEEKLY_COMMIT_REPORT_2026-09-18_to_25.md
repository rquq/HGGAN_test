# DEV branch: weekly change report

**Review window:** 18–25 September 2026 (Asia/Bangkok)  
**Branch reviewed:** `dev`  
**HEAD at review:** `1bbc082` (`origin/dev`)  
**Commits in window:** four, from 19 to 24 September

This report summarizes changes recorded in DEV's Git history during the window, then calls out what those commits do and do not demonstrate. The active model and config diffs were read directly; checkpoint metrics below come from the saved evaluation histories under `chkpoints/23_9_2026`.

## Executive summary

The week focused on four themes:

1. Making the generator's 32px and 64px paths more structurally consistent and generalizing the local-style reliability mechanism instead of special-casing single-character references.
2. Making the texture-refinement module independently switchable for an ablation.
3. Replacing some temporary or legacy repository material with a broader `.gitignore` and removing archived backups.
4. Trying a magnitude-only frequency-distribution loss in place of contextual feature matching, then restoring bounded contextual matching while retaining frequency loss as an optional experiment.

The commits do not demonstrate that the texture module or frequency loss improves KID. The available texture-on and texture-off checkpoints began from different GAN checkpoints. Their results are descriptive, not a controlled estimate of the module's effect.

## Commit-by-commit record

| Commit | Date | Main area |
|---|---|---|
| [`187d2a3`](https://github.com/rquq/HGGAN_test/commit/187d2a3af9817e2c4904a11f9e61e0d7d873c6c1) | Sep 19 | Shared multiresolution generator geometry, learned local-style evidence gate, generalized short-reference handling, texture toggle, lexicon |
| [`745e904`](https://github.com/rquq/HGGAN_test/commit/745e904321a4b689452abfe56447011278c19f65) | Sep 19 | Ignore rules and removal of archived implementation backups |
| [`eaa8060`](https://github.com/rquq/HGGAN_test/commit/eaa8060eda81ae0e5684a97cdb2feb78a2ca1616) | Sep 23 | Initial spectral-distribution loss substitution |
| [`1bbc082`](https://github.com/rquq/HGGAN_test/commit/1bbc0822e04c42a722eb2584714f1beef4cc4eae) | Sep 24 | Contextual-loss restoration, masking/patch-input changes, matched batch/schedule settings |

### 1. `187d2a3` — shared generator geometry and evidence-based local style

**Generator geometry.** The 32px generator was expanded from a three-stage path to the same four-stage channel schedule used at 64px. At 32px, the last block refines the feature map without upsampling; at 64px, it performs the last 2× upsample. This gives the two input heights a shared block topology while letting their geometry differ. It is a G-architecture change, not proof of better image quality.

**Style encoder and OCR stride.** A shared `_input_stride_for_height` derives initial stride from input height. The recognizer's CTC length scale and the writer/style backbone's width reduction follow that same geometry rule. This replaces duplicated 32-vs-64 stride selection and keeps feature lengths aligned with the CNN output.

**Single-character references.** The earlier heuristic estimated the number of characters from image width and used that estimate to reduce local style for short references. The commit removed that content/length-specific rule and introduced a learned local-evidence gate. Its inputs are global/local feature descriptors and feature variation; it does not inspect the character ID or branch on punctuation, word length, or image-resolution label. The intent is to fall back smoothly toward global writer style when local evidence is unreliable. That intent is generalized; whether it improves rare punctuation transfer still needs measured ablation.

**Texture refinement toggle.** `GenModel.texture_refinement_enabled` was added. The module is still constructed, but disabling the setting freezes its parameters and skips its output residual in the generator. DEV configs set it off; `random_crop_recog` uses it as the on condition. This makes the comparison operationally possible without changing the main GBlock stack.

**Lexicon.** Single-character entries were allowed by changing the lexicon filter from “length at least two” to “nonempty”. This broadens sampling to one-character words where present in the lexicon.

### 2. `745e904` — repository cleanup

The commit expanded `.gitignore` to cover Python/tool caches, local environments and secrets, training outputs, datasets, generated artifacts, backups, and IDE files. It also deleted about 5,600 lines of archived StarBlock, legacy fusion, and pre-main-discriminator copies under `networks/backups/`.

The deleted files were historical backups, not the active G, fusion, or discriminator imported by the current training model. The commit also revised the existing Vietnamese improvement report. One practical consequence of the new ignore rules is that newly created paths named `tests/` and `docs/` may be ignored; check `git check-ignore -v <path>` if a future source test or document does not appear in Git status.

### 3. `eaa8060` — frequency-distribution loss experiment

The training loop stopped requesting reference/generated backbone feature maps for contextual matching. Instead, it computed a `SpectralDistributionLoss` between the real image and its same-text reconstruction. The objective compares averaged log magnitudes of 2D FFTs over image patches. The three GAN configs assigned it weight `4.0`.

This was a meaningful change in **both loss type and supervision target**: contextual loss had compared reference-style and generated-style-transfer feature maps; the spectral objective compared real and reconstructed image patches. It was not a like-for-like replacement. The magnitude-only spectrum omits phase/location information, so equal-looking frequency statistics do not ensure strokes are positioned correctly. The commit's implementation is a custom magnitude loss, not the published Focal Frequency Loss.

### 4. `1bbc082` — restore bounded contextual matching and remove patch corruption

**Contextual matching.** The style-transfer branch again compares reference and generated feature maps, using reference features already cached for the batch and generated features already computed by the style-cycle path. Contextual loss is bounded by pooling valid feature crops to at most 256 spatial positions per map. At the default weight (`lambda_ctx: 0.1`), this gives direct reference-conditioned appearance supervision without the unbounded pairwise feature comparison. Frequency loss remains in the code as an optional setting; the default config now uses `lambda_frequency: 0.0`.

**Patch discriminator inputs.** Stripe masking was removed from sampled StrokePatchD crops. This matters because the old function accepted a requested 2–5% mask ratio but did not use it; test probes showed it erased around 12.7% of 16×16 patches and 6.5% of 32×32 patches. Real patch inputs were also simplified to unwarped real crops, matching the unwarped fake-patch path. Global D keeps its real/fake DiffAug path.

**Batch and writer schedule.** The full 32px config changed from batch 16 to batch 8, matching the 64px config. The 32px-only late writer-loss decay was removed so the writer identity weight stays at 0.5 as in 64px. The local config keeps its deliberately small batch for local smoke runs.

**StrokePatchD spatial detail.** Each patch block now downsamples only if halving both spatial axes leaves at least four cells. Consequently, the score maps for 16×16 and 32×32 crops are both at least 4×4, instead of 2×2 and 4×4. The rule follows crop geometry and does not add trainable parameters.

## Texture-refinement checkpoints and KID

The two saved 32px runs in `/home/quq/machineLearning/HTG/chkpoints/23_9_2026` show:

| Saved run | Texture setting | Start checkpoint in config | Epochs recorded | Last FID / KID / HWD | Mean KID, epochs 60–74 |
|---|---:|---|---:|---:|---:|
| `last_fid_3.4563_random_crop.pth` | On | `last_fid_3.2284.pth` | 84 | 3.4563 / 0.12619 / 0.10867 | 0.12362 |
| `best_fid_3.4892_dev.pth` | Off | `last_fid_3.7043.pth` | 74 | 3.4892 / 0.12485 / 0.10591 | 0.14506 |

The enabled run's mean KID over this shared epoch window is lower, and its final HWD is close to the disabled run. But the config snapshots differ in the GAN checkpoint loaded (`last_fid_3.2284.pth` versus `last_fid_3.7043.pth`) as well as the texture toggle. The saved runs therefore **do not isolate the toggle's effect**. They are consistent with the texture path helping, doing little, or being outweighed by the different starting state; the metrics cannot distinguish those explanations.

The enabled checkpoint's learned detail gain is about 0.0394, compared with its initial 0.03 and maximum 0.15. This shows the detail residual was used and adjusted, but the gain value alone says nothing about KID. The disabled branch remains at its initialization value because that path is frozen.

**Assessment:** I would not claim the texture generator is improving KID from these runs. Keep it as a research toggle if you want to measure it. For a valid on/off test, load the *same* GAN checkpoint, use the same data order, seed and remaining config, and compare multiple runs or at least repeat the paired continuation. Compare KID trajectories on the exact same evaluator protocol, not just the best-FID checkpoint names.

## What this week does not establish

- The shared G topology, local-evidence gate, contextual-loss return, or texture residual has not been proven to improve KID by the commit diff itself.
- The two saved texture checkpoints are not an apples-to-apples toggle test because their starting GAN checkpoints differ.
- A passing forward/backward smoke test does not show that an architectural change improves generated handwriting.
- The four-commit window changed both model behavior and optimizer/data schedules; a metric change across a whole run cannot automatically be attributed to one component.

## Files worth reading alongside the history

- Active generator, global discriminator, and StrokePatchD: [`networks/BigGAN_networks.py`](networks/BigGAN_networks.py)
- Style encoder, writer backbone, and OCR stride: [`networks/module.py`](networks/module.py)
- Loss implementations: [`networks/loss.py`](networks/loss.py)
- Training paths and loss wiring: [`networks/model.py`](networks/model.py)
- 32px/64px/local GAN settings: [`configs/gan_iam_32.yml`](configs/gan_iam_32.yml), [`configs/gan_iam_64.yml`](configs/gan_iam_64.yml), [`configs/gan_iam_local.yml`](configs/gan_iam_local.yml)
- Existing change report revised during this window: [`BAO_CAO_CAI_TIEN_DEV_01-10_09_2026.md`](BAO_CAO_CAI_TIEN_DEV_01-10_09_2026.md)
