#!/usr/bin/env python3
"""Read-only, CPU-only metadata/evaluation-history extraction for local GAN checkpoints."""

import csv
import gc
import io
from pathlib import Path

import torch


ROOT = Path('/home/quq/machineLearning/HTG')
CHECKPOINTS = [
    'chkpoints/14_10_2026/best_fid_2.4463_alpha.pth',
    'chkpoints/14_10_2026/best_fid_2.5052_alpha.pth',
    'chkpoints/14_10_2026/last_fid_2.8220_alpha.pth',
    'chkpoints/14_10_2026/best_fid_2.7598_main.pth',
    'chkpoints/14_10_2026/last_fid_2.8652_main.pth',
    'chkpoints/7_10_2026/best_fid_3.2614_alpha.pth',
    'chkpoints/7_10_2026/last_fid_3.3180_alpha.pth',
    'chkpoints/7_10_2026/best_fid_2.5464_main.pth',
    'chkpoints/7_10_2026/last_fid_2.8561_main.pth',
]


def scalar_dict(values):
    if not isinstance(values, dict):
        return {}
    return {
        str(k): v for k, v in values.items()
        if v is None or isinstance(v, (str, int, float, bool))
    }


def flatten(value, prefix=''):
    flattened = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f'{prefix}.{key}' if prefix else str(key)
            flattened.update(flatten(item, path))
    elif isinstance(value, (list, tuple)):
        flattened[prefix] = f'list[{len(value)}]'
    elif value is None or isinstance(value, (str, int, float, bool)):
        flattened[prefix] = value
    return flattened


def metric_summary(rows):
    parsed = []
    for row in rows:
        try:
            parsed.append({
                key: float(row[key]) if row.get(key, '') not in ('', None) else None
                for key in ('epoch', 'fid', 'kid', 'hwd')
            })
        except (TypeError, ValueError, KeyError):
            continue
    fid_rows = [row for row in parsed if row.get('fid') is not None]
    if not fid_rows:
        return len(rows), None, None
    best = min(fid_rows, key=lambda row: row['fid'])
    return len(rows), best, parsed[-1] if parsed else None


def main():
    torch.set_num_threads(1)
    print(f'torch={torch.__version__}; cuda_available={torch.cuda.is_available()} (not used)')
    for relative in CHECKPOINTS:
        path = ROOT / relative
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        print(f'\n=== {relative} ===')
        print(f'root_keys={sorted(checkpoint.keys())}')
        for key in ('checkpoint_format_version', 'Epoch', 'iter_count', 'fid',
                    'best_fid', 'dataset', 'seed', 'code_version', 'git_sha',
                    'commit', 'run_id', 'config'):
            if key in checkpoint:
                value = checkpoint[key]
                print(f'{key}={value!r}')

        config = checkpoint.get('config_snapshot')
        print(f'config_snapshot_keys={sorted(config.keys()) if isinstance(config, dict) else type(config).__name__}')
        if isinstance(config, dict):
            important = {
                key: value for key, value in flatten(config).items()
                if any(token in key.lower() for token in (
                    'height', 'width', 'crop', 'critic', 'seed', 'batch_size',
                    'epoch', 'lr', 'rare_word', 'lambda_', 'pretrained_ckpt',
                    'dataset', 'polarity', 'split', 'class', 'ema_beta',
                    'd_aug', 'vae_mode', 'char_width',
                ))
            }
            print('config_relevant=' + repr(important))

        manifest = checkpoint.get('model_manifest')
        if isinstance(manifest, dict):
            print('model_manifest=' + repr(manifest))

        history = checkpoint.get('eval_history')
        if isinstance(history, (list, tuple)):
            n_rows, best, last = metric_summary(history)
            print(f'embedded_history_rows={n_rows}; min_fid_row={best}; final_row={last}')
        csv_text = checkpoint.get('eval_metrics_csv')
        if isinstance(csv_text, str) and csv_text:
            csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
            n_rows, best, last = metric_summary(csv_rows)
            print(f'embedded_csv_rows={n_rows}; min_fid_row={best}; final_row={last}')
            artifact = ROOT / 'HGGAN_test/alpha/test_artifacts' / (
                'checkpoint_analysis_14_10' if '14_10_2026' in relative
                else 'checkpoint_analysis_7_10'
            ) / (Path(relative).stem + '.csv')
            if artifact.exists():
                artifact_rows = list(csv.DictReader(artifact.open(newline='', encoding='utf-8')))
                print(f'artifact_csv={artifact.relative_to(ROOT)}; rows={len(artifact_rows)}; '
                      f'rows_exact={csv_rows == artifact_rows}')
        else:
            print('embedded_csv=missing')

        # State tensor shape/size signatures only; no modules are instantiated and
        # no forward pass or other inference is performed.
        for key, value in checkpoint.items():
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                total = sum(t.numel() for t in value.values())
                samples = [(name, tuple(t.shape)) for name, t in list(value.items())[:3]]
                if key in ('G', 'G_EMA'):
                    print(f'state[{key}]: tensors={len(value)}; numel={total}; sample_shapes={samples}')

        del checkpoint
        gc.collect()


if __name__ == '__main__':
    main()
