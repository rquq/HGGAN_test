import os
import argparse
import random
import numpy as np
import torch
from munch import Munch
from lib.utils import yaml2config
from networks import get_model


def format_metrics(metrics: dict) -> str:
    lines = [
        "",
        "=" * 54,
        "               EVALUATION METRICS SUMMARY",
        "=" * 54,
    ]

    cleaned = {}
    for key, val in metrics.items():
        if isinstance(val, (np.generic, np.ndarray)):
            cleaned[key] = float(val)
        elif isinstance(val, tuple) and len(val) == 2:
            cleaned[key] = (float(val[0]), float(val[1]))
        else:
            cleaned[key] = val

    metric_names = [
        ('fid', 'FID (Frechet Inception Distance)'),
        ('kid', 'KID (Kernel Inception Distance)'),
        ('is_gen', 'IS (Inception Score - Gen)'),
        ('is_org', 'IS (Inception Score - Real)'),
        ('hwd', 'HWD (Handwriting Distance)'),
        ('cmmd', 'CMMD (CLIP MMD)'),
        ('wier', 'WIER (Writer ID Error Rate)'),
        ('cer', 'CER (Character Error Rate)'),
        ('wer', 'WER (Word Error Rate)'),
        ('psnr', 'PSNR'),
        ('mssim', 'MS-SSIM'),
    ]

    printed_keys = set()
    for key, label in metric_names:
        if key in cleaned:
            val = cleaned[key]
            printed_keys.add(key)
            if isinstance(val, float):
                lines.append(f"  {label:<34}: {val:.4f}")
            elif isinstance(val, tuple) and len(val) == 2:
                lines.append(f"  {label:<34}: mean={val[0]:.4f}, std={val[1]:.4f}")
            else:
                lines.append(f"  {label:<34}: {val}")

    for key, val in cleaned.items():
        if key not in printed_keys:
            if isinstance(val, float):
                lines.append(f"  {key.upper():<34}: {val:.4f}")
            else:
                lines.append(f"  {key.upper():<34}: {val}")

    lines.append("=" * 54)
    lines.append("")
    return "\n".join(lines)


def parse_bool(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ('true', '1', 'yes', 't', 'y'):
        return True
    elif s in ('false', '0', 'no', 'f', 'n'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{val}'.")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation test script")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/gan_iam.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--ckpt",
        nargs="?",
        type=str,
        default=None,
        help="Checkpoint for evaluation (overrides config if set)",
    )

    parser.add_argument(
        "--split",
        nargs="?",
        type=str,
        default=None,
        help="Dataset split for evaluation (overrides config if set)",
    )

    parser.add_argument(
        "--guided",
        dest='guided',
        default=None,
        type=parse_bool,
        help="Guided mode flag (overrides config if set)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (e.g. cuda:0 or cpu)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for evaluation reproducibility",
    )

    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Enable evaluation of all standard metrics (FID, KID, IS, HWD, CMMD)",
    )

    args = parser.parse_args()
    config_path = args.config if getattr(args, 'config', None) is not None else "configs/gan_iam.yml"
    cfg = yaml2config(config_path)
    infer_cfg = getattr(cfg, 'inference', cfg)

    # Resolution order: CLI flag > config YAML (inference block or root) > fallback default
    device = args.device if args.device is not None else getattr(infer_cfg, 'device', getattr(cfg, 'device', 'cuda:0'))
    ckpt = args.ckpt if args.ckpt is not None else getattr(infer_cfg, 'ckpt', getattr(cfg, 'ckpt', './pretrained/HiGAN+.pth'))
    cfg.valid = getattr(cfg, 'valid', None) or Munch()

    split = args.split if args.split is not None else getattr(cfg.valid, 'dset_split', getattr(infer_cfg, 'split', getattr(cfg, 'split', 'test')))
    guided = args.guided if args.guided is not None else getattr(infer_cfg, 'guided', getattr(cfg, 'guided', True))

    if args.seed is not None:
        seed = args.seed
    elif hasattr(cfg, 'seed') and cfg.seed is not None:
        seed = cfg.seed
    else:
        seed = random.randint(0, 10000)

    seed_everything(seed)

    cfg.device = device
    cfg.seed = seed
    cfg.valid.dset_split = split
    cfg.guided = guided

    if args.all_metrics:
        cfg.valid.validate_fid = True
        cfg.valid.validate_kid = True
        cfg.valid.validate_hwd = True
        cfg.valid.validate_cmmd = True
        cfg.valid.validate_cer = True
        cfg.valid.validate_wer = True
        cfg.valid.validate_is_gen = True
        cfg.valid.validate_is_org = True
        cfg.valid.validate_psnr = True
        cfg.valid.validate_mssim = True
        cfg.valid.validate_wier = True

    print("=" * 60)
    print("EVALUATION TEST CONFIGURATION")
    print(f" - Config File : {config_path}")
    print(f" - Checkpoint  : {ckpt}")
    print(f" - Split       : {split}")
    print(f" - Device      : {device}")
    print(f" - Guided Mode : {guided}")
    print(f" - Seed        : {seed}")
    print("=" * 60)

    model = get_model(cfg.model)(cfg)
    if not os.path.exists(ckpt):
        print(f"[Warning] Specified checkpoint path does not exist: {ckpt}")
    model.load(ckpt, device)
    model.set_mode('eval')
    val_results = model.validate(guided, test_stage=True)
    print(format_metrics(val_results))
    print("Raw metrics dict:")
    print(val_results)