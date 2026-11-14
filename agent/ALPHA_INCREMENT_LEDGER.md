# ALPHA increment ledger

Updated: 2026-11-15. This is the maintained index; detailed historical evidence
and controls are in [the full November15 record](2026-11-15-alpha-increment-ledger.md).
Append each new increment here and link its dated report. Source approval and
tested wiring are not trained quality results.

## Active Classic-derived experiment

| Stage | Source commit | Change / evidence |
|---|---|---|
| C0 | `0c435bd` | Approved Classic-derived rebuild with overlapping StrokePatchD, compatible teachers/native32 adaptation and runtime utilities; bundled starting point, not a pure crop ablation |
| Publication | `03ff873` | History repair; same C0 tree, no model increment |
| Workflow | `26936b0` | Verified publication requirement, no model increment |
| C1 | `c406eb3` | Batch8/sample interval500; training-recipe change, not architecture |
| C2 | `0532c68` | Metric/log parity and epoch sample axes; no new model |
| C3 | `84fbab2` | Loading/export/sampling/resume bookkeeping fixes; no new model |
| C4 | `14f6eea` | Target-only ContentContext before the original style/text seed; fresh GAN needed, teachers reusable |
| Recordkeeping | `c70f441` | Adds the branch instruction requiring this maintained index; documentation only, no tensors, objectives or optimizer behavior change |
| C5-BSSP32 | Proposed only | Four nonlinear stage-specific style MLPs, keeping the32D style and existing mixed seed; not implemented or trained |

The supplied current ALPHA checkpoint at epoch69 records FID6.271029,
KID(×100)0.265770, HWD0.118055. It is compatible with the current model but
does not establish a paired C3→C4 gain. No current contribution gets invented
quality improvements from an unrelated or resumed run.

## Retired lineage

The October global-only ablation, gain-only reference-memory reset, Gemini
cleanup and automatic GAN-OCR variants are historical experiments, not
cumulative components of today's C0–C4 model. Their SHAs, superseded designs,
publication history and evidence limitations are recorded separately in the
full November15 ledger. Do not resurrect memory or reuse those scores as a
current increment without an explicitly named new experiment.

## Entry contract

Each future implementation records: date; base/result source SHA; motivation;
exact delta; unchanged data/teachers/recipe/evaluator controls; whether fresh
GAN or teacher training is needed; measured verification and limitations;
normal push/fresh remote check; and any matched trained result when available.
Unknown results remain unknown. Workflow/utility fixes get their own label.
