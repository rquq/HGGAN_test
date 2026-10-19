# Autoresearch Log (DEV Branch)

## Experiment Trajectory & Decision Timeline

### 2026-08-30: Checkpoint Analysis & Kaggle GPU Optimization (Iteration 0)
* **Anchor**: Checkpoint `best_fid_3.0260_main.pth` (Epoch 56, FID: 3.0260, KID: 0.0633, HWD: 0.1001).
* **Throughput Diagnosis**:
  * Kaggle running 8 epochs / 12 hrs (~90 mins/epoch) due to:
    1. Small `batch_size: 8` causing 4,352 iterations/epoch with poor GPU Tensor Core saturation.
    2. Dual critic passes (`num_critic_train: 2`) adding 8,704 D-passes per epoch.
    3. Full 25k evaluation at `eval_batch_size: 8` consuming 18 mins per epoch.
* **Actions Taken**:
  * **H1 (Throughput Acceleration)**: Set training `batch_size: 16`, `num_critic_train: 1` (resumed phase), validation `batch_size: 32`. Projected speedup: 90 mins -> 26 mins/epoch (~27 epochs/12h).
  * **H2 (KID Optimization)**: Set `min_lr_ratio: 0.15` and `lambda_patch_adv: 0.45` for refined stroke settling.
  * **H3 (Anti-Elongation)**: Implemented breadth-aware slot contraction and safe norm bounding in `StyleEncoder`.
  * **H4 (Rare Characters)**: Implemented 25% scheduled rare-character synthesis in `idx_to_words`.
