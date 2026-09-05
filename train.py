import os

from datetime import datetime
import argparse
import traceback
import time

import random
import numpy as np
import torch
import torch.distributed as dist

from lib.utils import (
    yaml2config, init_wandb_run, write_wandb_log,
    update_job_status, write_results_table,
)
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

    if hasattr(cfg, 'img_height') and cfg.img_height:
        from lib.path_config import set_img_height
        set_img_height(cfg.img_height)

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
    job_started_at = time.time()
    status_path = os.environ.get('HGGAN_STATUS_PATH', '')
    branch_name = os.path.basename(os.path.abspath(os.getcwd()))
    job_name = os.environ.get('HGGAN_JOB_NAME', branch_name)
    job_status = 'running'
    error_text = ''
    update_job_status(
        status_path, job_status,
        job=job_name,
        branch=branch_name,
        config=args.config,
        model=getattr(cfg, 'model', 'unknown'),
        logdir=os.path.abspath(logdir),
        pid=os.getpid(),
        started_at=job_started_at,
    )

    wandb_run = init_wandb_run(cfg)
    write_wandb_log(
        f'[Startup] job={job_name} branch={branch_name} '
        f'model={getattr(cfg, "model", "unknown")} config={args.config} '
        f'logdir={logdir}'
    )
    model = None
    try:
        model = get_model(cfg.model)(cfg, logdir)
        model.train()
    except KeyboardInterrupt:
        job_status = 'interrupted'
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
        job_status = 'failed'
        error_text = traceback.format_exc()
        traceback.print_exc()
        write_wandb_log(error_text)
        raise
    finally:
        results_path = ''
        result_rows = {}
        try:
            results_path, result_rows = write_results_table(
                logdir, cfg, model=model, status=job_status,
                started_at=job_started_at, error=error_text,
                metadata={'job': job_name, 'branch': branch_name, 'config': args.config},
            )
        except Exception as results_error:
            result_rows = {'results_error': str(results_error)}
            print(f'[Warning] Could not write RESULTS table: {results_error}')
            write_wandb_log(f'[Warning] Could not write RESULTS table: {results_error}')
        update_job_status(
            status_path, job_status,
            job=job_name,
            branch=branch_name,
            config=args.config,
            model=getattr(cfg, 'model', 'unknown'),
            logdir=os.path.abspath(logdir),
            pid=os.getpid(),
            results_path=os.path.abspath(results_path) if results_path else '',
            metrics=result_rows,
            error=error_text.strip().splitlines()[-1] if error_text else '',
        )
        if wandb_run is not None:
            try:
                import wandb
                if wandb.run is not None:
                    write_wandb_log('[Shutdown] Finishing W&B run.')
                    wandb.finish()
            except Exception as e:
                print(f"[Warning] WandB shutdown failed: {e}")
