# ALPHA publication repair — 2026-11-03

## Observed failure and cause

The supplied GitHub screenshot showed ALPHA's latest commit as
`90225a2b1ba5380104729053814454b291ae24b0`, dated October 20.
Live `git ls-remote` confirmed this was not a UI caching problem.

The approved local implementation was committed as
`0c435bdc41b29119b53d64f47960716f299b5cd9`, but had not been pushed.
Local ALPHA and remote ALPHA had 115 and 78 unique commits respectively,
with no common merge base. Prior publication stopped at this divergence.
A local commit did not make the implementation available to a Kaggle clone.

The active Kaggle notebook clones the requested remote branch, or fetches
and resets an existing checkout to `origin/<branch>`. Consequently, a fresh
ALPHA launch through that step before this repair would use the old remote
global-only experiment, not the approved local classic-baseline/PatchD code.
No live Kaggle process or its checked-out SHA was available here; the code
used by a specific already-running job cannot be proved from the screenshot.

## Repair without changing model code

1. Fetched and inspected remote ALPHA and the divergent histories.
2. Preserved local safety refs for both previous tips.
3. Created a history-only `ours` merge, retaining the complete approved local
   source tree and keeping the old remote history as the second parent.
4. Verified that both previous tips are ancestors and that the merge tree is
   exactly the approved local tree:
   `383685323d4e53465f788a98ca0a296b52b48935`.
5. Pushed the merge with an ordinary fast-forward push; no force-push or
   published-history rewrite was used.

Publication merge:
`03ff873364dffb63b904ddd17a1a4755025d4022`.
Parents:
`0c435bdc41b29119b53d64f47960716f299b5cd9` and
`90225a2b1ba5380104729053814454b291ae24b0`.

## Persistent publication requirement

Added `AGENTS.md` to all four model branches. It requires completing a
commit and push after user-approved edits, unless explicitly instructed
otherwise, and comparing local HEAD with a fresh remote SHA before reporting
completion. Publication failure must be reported as incomplete. It also
requires preserving unrelated edits/history and excluding generated
artifacts, datasets, checkpoint binaries, and credentials.

Verified model-branch remote tips after publication:

| Branch | Verified remote SHA |
| --- | --- |
| alpha | `26936b0e4156f5e8ff3e8cd6910b1e8b13a2adfa` |
| main | `c17ab2510e0289ca62168140bf2c129451d79ce0` |
| dev | `c9c17fde0cef625e3099fb37683f8ff504023c8d` |
| classic_optimized | `83010dd274160d00bc27667b0a411d564bcb4a6c` |

Each matched its local HEAD in a fresh remote query. ALPHA's only content
change relative to the approved implementation is the new `AGENTS.md`;
the other three branches also received only that instruction file in this
repair. No model, loss, data, optimizer, or configuration behavior changed.

## Limits and exclusions

- Already-running Kaggle processes do not acquire new source after a push.
  The checkout must be refreshed before launching the intended experiment.
- Older results remain results for the code actually executed. They cannot
  be relabeled as tests of the newly published implementation.
- This repair verifies Git publication and source identity, not training
  correctness, convergence, GPU fit, or improved metrics. No training run was
  launched for this publication-only task.
- The outer master checkout's unrelated deletions and ALPHA's three
  untracked pretrained binaries were not staged or published.
- This report is published separately on the documentation `agent` branch;
  it does not import other pre-existing unpublished documentation/notebook
  changes from the outer working directory.
