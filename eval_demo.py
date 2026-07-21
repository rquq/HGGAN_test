import argparse
import sys
from lib.utils import yaml2config
from networks import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation demo script")
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
        help="checkpoint for evaluation (overrides config if set)",
    )

    parser.add_argument(
        "--mode",
        nargs="?",
        type=str,
        default=None,
        choices=["rand", "style", "text", "interp"],
        help="mode: [rand] [style] [text] [interp] (overrides config if set)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (e.g. cuda:0 or cpu)",
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)
    infer_cfg = getattr(cfg, 'inference', cfg)

    # Resolution order: CLI flag > config YAML (inference block or root) > fallback default
    device = args.device if args.device is not None else getattr(infer_cfg, 'device', getattr(cfg, 'device', 'cuda:0'))
    ckpt = args.ckpt if args.ckpt is not None else getattr(infer_cfg, 'ckpt', getattr(cfg, 'ckpt', './pretrained/HiGAN+.pth'))
    mode = args.mode if args.mode is not None else getattr(infer_cfg, 'mode', getattr(cfg, 'mode', 'text'))

    cfg.device = device
    cfg.ckpt = ckpt
    cfg.mode = mode

    model = get_model(cfg.model)(cfg)
    model.load(ckpt, device)
    model.set_mode('eval')

    if mode == 'style':
        model.eval_style()
    elif mode == 'rand':
        model.eval_rand()
    elif mode == 'interp':
        model.eval_interp()
    elif mode == 'text':
        model.eval_text()
    else:
        print(f"Unsupported mode: {mode} | Choose from [rand, style, text, interp]")
        sys.exit(1)