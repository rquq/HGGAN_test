import argparse
import os
import torch
from lib.utils import yaml2config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract deployable weights from checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gan_iam.yml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to input checkpoint (overrides config if set)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Path to output deploy checkpoint (overrides config if set)",
    )

    args = parser.parse_args()

    # Load config if available
    cfg = None
    infer_cfg = None
    if os.path.exists(args.config):
        cfg = yaml2config(args.config)
        infer_cfg = getattr(cfg, 'inference', cfg)

    ckpt = args.ckpt if args.ckpt is not None else (
        getattr(infer_cfg, 'ckpt', getattr(cfg, 'ckpt', './pretrained/HiGAN+.pth')) if cfg else './pretrained/HiGAN+.pth'
    )
    dst = args.dst if args.dst is not None else (
        getattr(infer_cfg, 'deploy_dst', getattr(cfg, 'deploy_dst', './pretrained/deploy_HiGAN+.pth')) if cfg else './pretrained/deploy_HiGAN+.pth'
    )

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Source checkpoint not found: {ckpt}")

    state_dict = torch.load(ckpt, map_location='cpu', weights_only=False)
    new_state_dict = {}
    for key in ['Generator', 'StyleEncoder', 'StyleBackbone']:
        if key in state_dict:
            new_state_dict[key] = state_dict[key]

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(new_state_dict, dst)
    print('Saved deploy checkpoint to ->', dst)
