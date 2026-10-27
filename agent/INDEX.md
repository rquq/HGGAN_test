# Project documentation index

Collected on 2026-10-20 from the four local branch checkouts, project reports and locally available Git history. Historical documents retain their original claims and dates; they are not rewritten as descriptions of the current architecture.

## Start here

- [2026-10-27 MAIN/DEV cleanup and checkpoint continuity](2026-10-27-main-dev-cleanup-port.md)
- [2026-10-27 ALPHA cleanup implementation and numerical limits](2026-10-27-alpha-cleanup-implementation.md)
- [2026-10-27 ALPHA refactor review](2026-10-27-alpha-refactor-review.md)
- [2026-10-24 shared 64D reference memory and objective balance — current DEV/MAIN settings](2026-10-24-shared-memory-balance.md)
- [2026-10-24 MAIN gain-only spatial modulation experiment](2026-10-24-main-gain-only-modulation.md)
- [2026-10-24 DEV memory dimensions, redundancy and gradient audit](analysis/2026-10-24-dev-memory-capacity/REPORT.md)
- Local-only 2026-10-24 ALPHA/DEV sample and checkpoint analysis: `analysis/2026-10-24-checkpoints/REPORT.md` (generated sample panels stay outside Git).
- [2026-10-23 DEV shared-backbone reference memory — current design](2026-10-23-dev-shared-backbone-reference-memory.md)
- [Superseded two-view experiment and correction history](2026-10-23-dev-dual-reference-memory.md)
- [Change documentation workflow](WORKFLOW.md)

- [2026-10-23 MAIN/ALPHA checkpoint audit and local-token collapse probes](analysis/2026-10-23-checkpoints/REPORT.md)

- [2026-10-22 independent CER/WER comparison: MAIN, ALPHA and three baselines](evaluation/2026-10-22-baseline-ocr/REPORT.md)

- [Current independent OCR evaluation guide](OCR_EVALUATION.md)
- [OCR implementation measurements and verification](OCR_IMPLEMENTATION_RESULTS.md)
- [2026-10-19 experiment design](2026-10-19-handwriting-evaluation-design.md)
- [2026-10-19 implementation plan](2026-10-19-handwriting-evaluation-plan.md)
- [Latest checkpoint-analysis report](reports/checkpoints_19_10_2026/REPORT.md)
- [Detailed ALPHA versus HiGAN+ report](reports/alpha_vs_higanplus_2026_10_15/alpha_vs_higanplus_vi.md)
- [Expanded Draw.io architecture with comparison panel](reports/alpha_vs_higanplus_2026_10_15/drawio_5_panels_expanded/alpha_training_architecture.drawio)

## Collection and preservation

200 existing documentation/support files were moved. Three additional unique documents were recovered from Git history. Seventy identical historical copies point to existing archive files, with every branch/revision association retained in [relocation_manifest.json](relocation_manifest.json).

The follow-up collection also moved 101 checkpoint-analysis documents/support assets from September 14 and 30. ALPHA naming changes are recorded in [ALPHA_MIGRATION.md](ALPHA_MIGRATION.md); original relocation provenance intentionally retains historical source names.

### Additional checkpoint analyses

- [September 14 checkpoint report](reports/checkpoints_14_9_2026/analysis/REPORT.md)
- [Late-KID patch notes](reports/checkpoints_14_9_2026/analysis/LATE_KID_PATCH.md)
- [September 30 checkpoint report](reports/checkpoints_30_9_2026/analysis/REPORT.md)
- [September 30 review](reports/checkpoints_30_9_2026/analysis/LUNA_REVIEW.md)
- [September 30 fix implementation](reports/checkpoints_30_9_2026/analysis/FIX_IMPLEMENTATION.md)

All 239 runtime Python/YAML files checked before/after documentation relocation were identical. The subsequent ALPHA rename changes only a branch-label lookup in each training checkout, not model computation. Tracked reports have been removed from the model branches and preserved in this shared agent archive.

Preserved: branch-specific versions, LaTeX inputs, rendered diagrams, report builders, review notes and supporting report assets. Data, checkpoints, training notebooks, run logs/results, credentials, third-party repositories and operational agent instructions were not relocated. Two companion-report hyperlinks were repaired; original and archived checksums are recorded in link_updates.json.

CLASSIC_OPTIMIZED has no standalone documentation in its current checkout. Its available historical document paths resolve to shared archived copies; consult the manifest rather than treating inherited DEV documents as evidence of CLASSIC contributions.

## Documentation by source

### DEV

- [BAO_CAO_CAI_TIEN_DEV_01-10_09_2026.md](branches/dev/BAO_CAO_CAI_TIEN_DEV_01-10_09_2026.md)
- [DEV_WEEKLY_COMMIT_REPORT_2026-09-18_to_25.md](branches/dev/DEV_WEEKLY_COMMIT_REPORT_2026-09-18_to_25.md)
- [DEV_WEEKLY_SCORE_IMPROVEMENT_2026-09-24_to_30.md](branches/dev/DEV_WEEKLY_SCORE_IMPROVEMENT_2026-09-24_to_30.md)
- [docs/diagrams/dev_model_loss_map.html](branches/dev/docs/diagrams/dev_model_loss_map.html)
- [docs/diagrams/dev_model_overview.html](branches/dev/docs/diagrams/dev_model_overview.html)
- [docs/diagrams/dev_three_training_flows.html](branches/dev/docs/diagrams/dev_three_training_flows.html)
- [docs/diagrams/latex/README.md](branches/dev/docs/diagrams/latex/README.md)
- [docs/diagrams/latex/dev_diagram_style.tex](branches/dev/docs/diagrams/latex/dev_diagram_style.tex)
- [docs/diagrams/latex/dev_full_architecture.pdf](branches/dev/docs/diagrams/latex/dev_full_architecture.pdf)
- [docs/diagrams/latex/dev_full_architecture.tex](branches/dev/docs/diagrams/latex/dev_full_architecture.tex)
- [docs/diagrams/latex/dev_fusion.pdf](branches/dev/docs/diagrams/latex/dev_fusion.pdf)
- [docs/diagrams/latex/dev_fusion.tex](branches/dev/docs/diagrams/latex/dev_fusion.tex)
- [docs/diagrams/latex/dev_strokepatchd.pdf](branches/dev/docs/diagrams/latex/dev_strokepatchd.pdf)
- [docs/diagrams/latex/dev_strokepatchd.tex](branches/dev/docs/diagrams/latex/dev_strokepatchd.tex)
- [docs/diagrams/latex/dev_strokepatchd_mechanism.pdf](branches/dev/docs/diagrams/latex/dev_strokepatchd_mechanism.pdf)
- [docs/diagrams/latex/dev_strokepatchd_mechanism.tex](branches/dev/docs/diagrams/latex/dev_strokepatchd_mechanism.tex)
- [docs/diagrams/latex/dev_style_encoder.pdf](branches/dev/docs/diagrams/latex/dev_style_encoder.pdf)
- [docs/diagrams/latex/dev_style_encoder.tex](branches/dev/docs/diagrams/latex/dev_style_encoder.tex)
- [networks/backups/dev_6955ee7/README.md](branches/dev/networks/backups/dev_6955ee7/README.md)
- [networks/backups/pre_main_discriminator_20260730/README.md](branches/dev/networks/backups/pre_main_discriminator_20260730/README.md)

### MAIN

- [BAO_CAO_CAI_TIEN_DEV_01-10_09_2026.md](branches/main/BAO_CAO_CAI_TIEN_DEV_01-10_09_2026.md)
- [docs/style_encoder_context.md](branches/main/docs/style_encoder_context.md)

### ALPHA

- [.superpowers/sdd/implementation_plan/constraints.md](branches/alpha/.superpowers/sdd/implementation_plan/constraints.md)
- [.superpowers/sdd/implementation_plan/final-fix-review-package.md](branches/alpha/.superpowers/sdd/implementation_plan/final-fix-review-package.md)
- [.superpowers/sdd/implementation_plan/final-review-package.md](branches/alpha/.superpowers/sdd/implementation_plan/final-review-package.md)
- [.superpowers/sdd/implementation_plan/progress.md](branches/alpha/.superpowers/sdd/implementation_plan/progress.md)
- [.superpowers/sdd/implementation_plan/task-1-brief.md](branches/alpha/.superpowers/sdd/implementation_plan/task-1-brief.md)
- [.superpowers/sdd/implementation_plan/task-1-report.md](branches/alpha/.superpowers/sdd/implementation_plan/task-1-report.md)
- [.superpowers/sdd/implementation_plan/task-1-review-package.md](branches/alpha/.superpowers/sdd/implementation_plan/task-1-review-package.md)
- [.superpowers/sdd/implementation_plan/task-2-brief.md](branches/alpha/.superpowers/sdd/implementation_plan/task-2-brief.md)
- [.superpowers/sdd/implementation_plan/task-2-report.md](branches/alpha/.superpowers/sdd/implementation_plan/task-2-report.md)
- [.superpowers/sdd/implementation_plan/task-2-review-package.md](branches/alpha/.superpowers/sdd/implementation_plan/task-2-review-package.md)
- [.superpowers/sdd/implementation_plan/task-3-brief.md](branches/alpha/.superpowers/sdd/implementation_plan/task-3-brief.md)
- [.superpowers/sdd/implementation_plan/task-3-report.md](branches/alpha/.superpowers/sdd/implementation_plan/task-3-report.md)
- [.superpowers/sdd/implementation_plan/task-3-review-package.md](branches/alpha/.superpowers/sdd/implementation_plan/task-3-review-package.md)
- [.superpowers/sdd/implementation_plan/task-4-brief.md](branches/alpha/.superpowers/sdd/implementation_plan/task-4-brief.md)
- [.superpowers/sdd/implementation_plan/task-4-report.md](branches/alpha/.superpowers/sdd/implementation_plan/task-4-report.md)
- [.superpowers/sdd/implementation_plan/task-4-review-package.md](branches/alpha/.superpowers/sdd/implementation_plan/task-4-review-package.md)
- [.superpowers/sdd/implementation_plan/task-5-brief.md](branches/alpha/.superpowers/sdd/implementation_plan/task-5-brief.md)
- [.superpowers/sdd/implementation_plan/task-5-report.md](branches/alpha/.superpowers/sdd/implementation_plan/task-5-report.md)
- [.superpowers/sdd/implementation_plan/task-5-review-package.md](branches/alpha/.superpowers/sdd/implementation_plan/task-5-review-package.md)
- [BAO_CAO_CAI_TIEN_DEV_01-10_09_2026.md](branches/alpha/BAO_CAO_CAI_TIEN_DEV_01-10_09_2026.md)
- [docs/style_encoder_context.md](branches/alpha/docs/style_encoder_context.md)
- [networks/backups/dev_6955ee7/README.md](branches/alpha/networks/backups/dev_6955ee7/README.md)
- [networks/backups/pre_main_discriminator_20260730/README.md](branches/alpha/networks/backups/pre_main_discriminator_20260730/README.md)

### Shared reports

- [DEV_VS_ALPHA_REPORT.md](shared/DEV_VS_ALPHA_REPORT.md)
- [REPORT_DEV_MAIN_ALPHA_VI.md](shared/REPORT_DEV_MAIN_ALPHA_VI.md)
- [main-htg-test-specia-degits.docx](shared/main-htg-test-specia-degits.docx)
- [main-htg-test-specia-degits.html](shared/main-htg-test-specia-degits.html)
- [main-htg-test-specia-degits.md](shared/main-htg-test-specia-degits.md)
- [main_design_vs_higanplus_vi.html](shared/main_design_vs_higanplus_vi.html)
- [main_vs_dev_generator_report.tex](shared/main_vs_dev_generator_report.tex)
- [main_vs_dev_generator_report_vi.tex](shared/main_vs_dev_generator_report_vi.tex)
- [alpha_design_vs_higanplus_vi.html](shared/alpha_design_vs_higanplus_vi.html)
- [spis-gan-test-special-degits.docx](shared/spis-gan-test-special-degits.docx)
- [spis-gan-test-special-degits.html](shared/spis-gan-test-special-degits.html)
- [spis-gan-test-special-degits.md](shared/spis-gan-test-special-degits.md)

### Recovered unique history

- [main/findings.md](history/main/findings.md)
- [main/research-log.md](history/main/research-log.md)
- [main/to_human/kaggle_optimization_report.html](history/main/to_human/kaggle_optimization_report.html)

### ALPHA versus HiGAN+ report and diagrams

- [drawio/README.md](reports/alpha_vs_higanplus_2026_10_15/drawio/README.md)
- [drawio/preview.html](reports/alpha_vs_higanplus_2026_10_15/drawio/preview.html)
- [drawio/alpha_training_architecture.drawio](reports/alpha_vs_higanplus_2026_10_15/drawio/alpha_training_architecture.drawio)
- [drawio_5_panels/README.md](reports/alpha_vs_higanplus_2026_10_15/drawio_5_panels/README.md)
- [drawio_5_panels/preview.html](reports/alpha_vs_higanplus_2026_10_15/drawio_5_panels/preview.html)
- [drawio_5_panels/alpha_training_architecture.drawio](reports/alpha_vs_higanplus_2026_10_15/drawio_5_panels/alpha_training_architecture.drawio)
- [drawio_5_panels_expanded/README.md](reports/alpha_vs_higanplus_2026_10_15/drawio_5_panels_expanded/README.md)
- [drawio_5_panels_expanded/preview.html](reports/alpha_vs_higanplus_2026_10_15/drawio_5_panels_expanded/preview.html)
- [drawio_5_panels_expanded/alpha_training_architecture.drawio](reports/alpha_vs_higanplus_2026_10_15/drawio_5_panels_expanded/alpha_training_architecture.drawio)
- [figures/01_overview.pdf](reports/alpha_vs_higanplus_2026_10_15/figures/01_overview.pdf)
- [figures/02_style_encoder.pdf](reports/alpha_vs_higanplus_2026_10_15/figures/02_style_encoder.pdf)
- [figures/03_fusion_generator.pdf](reports/alpha_vs_higanplus_2026_10_15/figures/03_fusion_generator.pdf)
- [figures/04_strokepatchd.pdf](reports/alpha_vs_higanplus_2026_10_15/figures/04_strokepatchd.pdf)
- [figures/05_random_flow.pdf](reports/alpha_vs_higanplus_2026_10_15/figures/05_random_flow.pdf)
- [figures/06_transfer_flow.pdf](reports/alpha_vs_higanplus_2026_10_15/figures/06_transfer_flow.pdf)
- [figures/07_reconstruction_flow.pdf](reports/alpha_vs_higanplus_2026_10_15/figures/07_reconstruction_flow.pdf)
- [alpha_vs_higanplus_vi.html](reports/alpha_vs_higanplus_2026_10_15/alpha_vs_higanplus_vi.html)
- [alpha_vs_higanplus_vi.md](reports/alpha_vs_higanplus_2026_10_15/alpha_vs_higanplus_vi.md)
- [alpha_vs_higanplus_vi.pdf](reports/alpha_vs_higanplus_2026_10_15/alpha_vs_higanplus_vi.pdf)
