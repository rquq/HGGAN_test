# OCR W&B history after checkpoint restore — 2026-11-20

The Kaggle excerpt shows actual TrOCR updates around 5,850–6,150, while
W&B's internal history cursors are already 10,195 and 10,924. The code
resumes stable W&B IDs online and passed the optimizer update as the
SDK's explicit `step=`. W&B drops lower-step rows, so missing history
charts do not imply that training or checkpointing stopped. Cloud
history can be ahead when the restored checkpoint is older, but the
latest cloud history/restore receipt was not available for inspection.
The locally available November 18 archive uses the earlier 10,000-update
recipe and does not establish the state of this 15,008-update run.

The tracking wrapper now lets W&B advance its own history index and logs
the exact TrOCR update as `ocr_update`. Train and validation metrics use
that field as the x-axis. No dashboard history is replaced, deleted or
backfilled. The same online IDs and checkpoint recipes still resume.
The worker prints the mode and online run URL once; offline/disabled
modes do not publish charts live. Console capture for OCR remains off:
Kaggle still shows progress, W&B renders metric charts and final test
summary, and W&B's Logs tab is intentionally not populated by OCR.

Verification: 33 focused CPU tests passed. A mock resumed run with a
history cursor at 10,195 dropped all three lower-step rows before the
patch and accepts all three afterward. Pinned W&B 0.27.0 offline also
discarded the old explicit-step payload at 6,097, then persisted patched
`(_step, ocr_update)` pairs `(10196, 6097)`, `(10197, 6100)` and
`(10198, 6100)`. The local `.wandb` records were read back to confirm
the values. No private dashboard was modified or live Kaggle job run.

On the next launch, keep the same batch multiplier and restore the
normal saved output. The notebook already fetches the benchmark branch
before starting workers. An already-running process will not pick up
the patch. Missing old log points may still exist in local checkpoint
history CSV; this fix does not re-upload them.
