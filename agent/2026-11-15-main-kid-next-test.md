# MAIN KID: next controlled test — 2026-11-15

No MAIN architecture/config change was requested after the user's clarification;
this is a proposal, not an implemented or measured quality improvement.

The current best saved KID×100 is0.049210. Finish the existing80-epoch schedule
without a simultaneous LR/loss/architecture rewrite. Keep the test-selected and
same-transcript evaluation limitations from the checkpoint report explicit.

My first small experiment would increase the existing SWD directions32→64,
holding samples-per-word64, lambda_distribution0.1, every other objective,
architecture, teachers, inputs, seeds and G-update budget fixed. More directions
can reduce per-update projection-estimation noise; this is a hypothesis, not
evidence of a lower KID. Current directions already refresh each update, so
there is no permanent32-direction blind subspace to claim as a bug. Increasing
quantile resolution or changing loss weight simultaneously would confound it.

The existing SW objective operates on frozen writer-backbone features, not the
Inception features used for KID. Texture-distribution matching is supported by
[Heitz et al., CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Heitz_A_Sliced_Wasserstein_Loss_for_Neural_Texture_Synthesis_CVPR_2021_paper.html),
but that paper does not establish a KID gain for this handwriting model or prove
that more projections help here. Do not train directly against test Inception
features to chase the benchmark number.

Compare matched training budgets, not different epochs per Kaggle session.
Evaluate full unchanged streams with the same KID subsets/seeds and examine HWD
plus a fixed arbitrary-text reference panel. An apparent tiny KID gain without
repeats may be estimator/training variation. A new prospective validation split
should select future experiments; do not repeatedly optimize against final test.

If this does nothing, the next meaningful candidate is multiscale image criticism
or less content-specific fixed feature supervision, as a separately named
experiment—not another attention/memory branch or more stacked auxiliary losses.
