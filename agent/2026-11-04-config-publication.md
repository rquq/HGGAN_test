# Configuration publication — 2026-11-04

The user requested committing and force-pushing the current branch changes.
Only the existing configuration edits were committed; no new model changes
were introduced during this task.

| Branch | Published commit | Change |
| --- | --- | --- |
| MAIN | `d0b812c75f0f7bdc898a2ee483a954c1a58b8501` | Remove a stale OCR evaluation comment from the 32px GAN config |
| ALPHA | `c406eb37eef9a7ca1bfa3fd0aa664f4bcbfb497a` | Set 32px/64px training batch size to 8, eval batch size to 8, and sample interval to 500 |
| DEV | `c9c17fde0cef625e3099fb37683f8ff504023c8d` | No new changes; already current |
| CLASSIC_OPTIMIZED | `83010dd274160d00bc27667b0a411d564bcb4a6c` | No new changes; already current |

All four pushes used an explicit `--force-with-lease` against the freshly
queried remote SHA. Each old remote tip was retained in a local safety ref.
All old remote tips were ancestors of their respective local heads, so no
remote history was actually rewritten or discarded. A fresh `ls-remote`
query confirmed that every published model-branch head matched local HEAD.

The three changed YAML files parsed successfully and the staged diffs passed
`git diff --check`. This was a publication task, not a training or GPU-memory
test; no claim about batch-size fit or model quality is made.

ALPHA's untracked pretrained binaries, generated artifacts, and the outer
master checkout's unrelated deletions were not staged. No empty commits
were created on unchanged branches.
