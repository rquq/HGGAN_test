import os
# The texture refiner alternates between a very large G-backward graph and
# a smaller D graph.  Keep the native caching allocator so it retains its
# high-water blocks instead of repeatedly returning/remapping them; this makes
# the physical GPU-memory trace stable without changing model computation.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'backend:native')

from datetime import datetime
import argparse
import traceback

import random
import numpy as np
import torch
import torch.distributed as dist

from lib.utils import yaml2config, init_wandb_run, write_wandb_log
from networks import get_model


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    # Variable word widths make cuDNN autotuning allocate many shape-specific
    # workspaces, so use the predictable heuristic instead.
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/gan_iam_64.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)

    run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4] + '-' + str(run_id))
    ckpt_dir = os.path.join(logdir, getattr(getattr(cfg, 'training', {}), 'ckpt_dir', 'ckpts'))
    os.makedirs(ckpt_dir, exist_ok=True)

    local_rank = getattr(cfg, 'local_rank', -1)
    if local_rank == -1 and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

    # Check if config specifies multi-gpu
    if local_rank == -1:
        if hasattr(cfg, "gpus") and cfg.gpus:
            if isinstance(cfg.gpus, (list, tuple)) and len(cfg.gpus) > 1:
                if "LOCAL_RANK" in os.environ:
                    local_rank = int(os.environ["LOCAL_RANK"])
            elif isinstance(cfg.gpus, int) and cfg.gpus > 1:
                if "LOCAL_RANK" in os.environ:
                    local_rank = int(os.environ["LOCAL_RANK"])
        elif getattr(cfg, "multi_gpu", False) or getattr(cfg, "distributed", False):
            if "LOCAL_RANK" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])

    cfg.local_rank = local_rank

    if local_rank > -1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        seed = getattr(cfg, "seed", 123456)
        if seed is not None:
            seed_everything(seed + local_rank)

        run_id_tensor = torch.tensor([ord(c) for c in run_id], dtype=torch.long, device=local_rank)
        dist.broadcast(run_id_tensor, src=0)
        run_id = "".join([chr(c) for c in run_id_tensor.cpu().tolist()])
    else:
        if hasattr(cfg, "seed") and cfg.seed is not None:
            seed_everything(cfg.seed)

    # Initialize W&B before model/logger construction so startup summaries and
    # checkpoint loading appear in the run's Logs tab.
    wandb_run = init_wandb_run(cfg)
    write_wandb_log(f'[Startup] branch=main config={args.config} logdir={logdir}')
    model = None
    try:
        model = get_model(cfg.model)(cfg, logdir)
        model.train()
    except KeyboardInterrupt:
        print("\n[Notice] Training interrupted by user.")
        write_wandb_log('[Notice] Training interrupted by user.')
        if model is not None and hasattr(model, 'save') and getattr(model, 'local_rank', 0) <= 0:
            print("[Notice] Saving emergency checkpoint (tag='interrupted')...")
            try:
                model.save('interrupted')
                print("[Notice] Emergency checkpoint saved successfully.")
            except Exception as e:
                print(f"[Warning] Failed to save interrupted checkpoint: {e}")
    except Exception:
        traceback.print_exc()
        write_wandb_log(traceback.format_exc())
        raise
    finally:
        if wandb_run is not None:
            try:
                import wandb
                if wandb.run is not None:
                    write_wandb_log('[Shutdown] Finishing W&B run.')
                    wandb.finish()
            except Exception as e:
                print(f"[Warning] WandB shutdown failed: {e}")
