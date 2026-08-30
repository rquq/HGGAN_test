import argparse
import os
import torch
from lib.utils import yaml2config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract deployable weights from checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gan_iam_64.yml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to input training checkpoint (e.g., ckpts/best_fid.pth or runs/.../ckpts/last.pth)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Path to output deploy checkpoint (e.g., pretrained/deploy.pth)",
    )
    parser.add_argument(
        "--use_raw",
        action="store_true",
        help="Force using raw weights instead of EMA weights if both are present",
    )

    args = parser.parse_args()

    # Load config if available
    cfg = None
    infer_cfg = None
    if os.path.exists(args.config):
        cfg = yaml2config(args.config)
        infer_cfg = getattr(cfg, 'inference', cfg)

    ckpt = args.ckpt if args.ckpt is not None else (
        getattr(infer_cfg, 'ckpt', getattr(cfg, 'ckpt', './pretrained/best_fid.pth')) if cfg else './pretrained/best_fid.pth'
    )
    dst = args.dst if args.dst is not None else (
        getattr(infer_cfg, 'deploy_dst', getattr(cfg, 'deploy_dst', './pretrained/deploy.pth')) if cfg else './pretrained/deploy.pth'
    )

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Source checkpoint not found: {ckpt}")

    state_dict = torch.load(ckpt, map_location='cpu', weights_only=False)
    new_state_dict = {}
    extracted_keys = []

    for key in ['Generator', 'StyleEncoder', 'StyleBackbone']:
        ema_key = f"ema_{key}"
        if not args.use_raw and ema_key in state_dict:
            new_state_dict[key] = state_dict[ema_key]
            extracted_keys.append(f"{key} (from {ema_key})")
        elif key in state_dict:
            new_state_dict[key] = state_dict[key]
            extracted_keys.append(key)
        else:
            print(f"[Warning] Key '{key}' not found in source checkpoint {ckpt}")

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(new_state_dict, dst)

    orig_size_mb = os.path.getsize(ckpt) / (1024 * 1024)
    dst_size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"Deployment Checkpoint Summary:")
    print(f" - Source Checkpoint : {ckpt} ({orig_size_mb:.2f} MB)")
    print(f" - Saved Checkpoint  : {dst} ({dst_size_mb:.2f} MB)")
    print(f" - Extracted Modules : {extracted_keys}")
