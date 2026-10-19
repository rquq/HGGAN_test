# Agent documentation

All new agent-authored project documentation belongs in this folder.
Keep generated test artifacts, checkpoints and metric output files separate and ignored by Git.
The model repositories are nested checkouts; this shared folder is outside their branch working trees.

Start with [INDEX.md](INDEX.md) for reports, architecture diagrams and recovered history.
The relocation_manifest.json records original paths, SHA-256 hashes and Git revisions.
Documents describe their original dated experiments, not necessarily the current model.

The shared archive is versioned on the agent Git branch; model branches remain main, dev, alpha and classic_optimized.
Both notebooks are versioned beside this folder under notebooks/.
Original HTG-level notebook paths link to those canonical notebook files.
