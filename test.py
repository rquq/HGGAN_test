import os
import argparse
import random
from munch import Munch
from lib.utils import yaml2config
from networks import get_model

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

    cfg.device = device
    cfg.seed = seed
    cfg.valid.dset_split = split
    cfg.guided = guided

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
    print("Evaluation Results:")
    for k, v in val_results.items():
        if isinstance(v, float):
            print(f"  {k:10s}: {v:.4f}")
        else:
            print(f"  {k:10s}: {v}")