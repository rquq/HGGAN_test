# Notebook previous-best carry-forward — 2026-11-15

Authoritative files are `notebooks/one_for_all_kaggle_HTG.ipynb` and its local
counterpart. The workspace-root notebooks are symlinks, not separate copies.
The Kaggle helper inspected has SHA256
`fd25d94943993a1bbeabfe538c4b89f91da59e59838a187fdec1bed19c5bc1be`.

Status: diagnosis and exact-function fixture reproduction complete; a source
repair has been offered for approval. No notebook/model edits are claimed here.

## Confirmed causes

1. **One-shot30-second startup window.** `copy_baseline_best_to_run_folder`
   polls at seconds0…29, then gives up permanently. A status at29 copies;
   status at30 or31 is missed. The monitor never retries. Slow cold imports can
   cross that boundary; W&B/model loading occur after startup status and do not
   explain a missing receipt in MAIN/DEV.
2. **Filename-dependent metric inference.** MAIN now selects KID. The helper
   recognizes `_kid_`, but a materialized generic `last.pth` defaults to FID.
   It can skip a real `best_kid_*` or carry a FID sibling instead. Explicit
   `best_kid.pth` also fails the direct-best check. A preserved alias symlink
   works because the helper resolves the real filename; materialization is
   therefore a real reason behavior can differ between exports.
3. **Best must be in the inspected source directory.** Nested Kaggle paths are
   not inherently broken: a last checkpoint plus its best sibling works at any
   depth. A best in a different nested folder is not found. Multiple matching
   bests are deliberately rejected, rather than guessed from newest/minimum.
4. **Some outcomes are silent or only conditionally safe.** Existing outputs are
   preserved silently. A writer racing between existence-check and replace can
   be overwritten; the fixture demonstrates the race, not an observed Kaggle
   event. A stale same-branch receipt is accepted, but normal launch deletes it
   first, so stale reuse needs overlapping/manual activity to explain it.

The active helper uses the exact status-reported run directory. It does **not**
pick the newest run; that old helper is only commented in the local notebook.
ALPHA publishes `checkpoint_dir`, while MAIN/DEV/classic currently fall back to
`ckpts`. Their current configs all use that name, so custom-output mismatch is
only a conditional risk, not a proven present failure.

## Narrow repair proposal

Keep existing notebook names/settings and avoid a new training/checkpoint system.
Resolve the intended source best explicitly, preserve metric names, and do not
pretend a last checkpoint is best. Handle generic/explicit best aliases and KID
as well as FID. Use actual source provenance instead of arbitrary global search.

For a notebook-only repair, retain a pending carry in the existing monitor and
retry after exact startup status arrives, including the final status check before
monitor completion. Validate branch/job/PID or launch identity; never target an
unrelated/newest run. Copy only into that run, with no-clobber publication, and
print a concise copied/preserved/missing/ambiguous outcome. Do not replace the
30-second window with another arbitrary longer deadline.

A synchronous model-startup copy can remove the race entirely, but it touches
every launcher/model entrypoint; it is not automatically the smallest change.
Keep linking/carry in the notebook unless that broader change is approved.

## Proof and limits

The exact existing notebook function passed19 fixture assertions documenting
working paths and the failures above. Reproducer, result JSON and detailed
mechanism report are under
`agent/analysis/2026-11-15-notebook-carry/`. Compilation passed. These are local
byte fixtures, not a live Kaggle Save Version test or a source-fix verification.
