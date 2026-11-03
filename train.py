import os
from datetime import datetime
import argparse
import random
import time
import traceback

import numpy as np
import torch

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
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HiGAN+ configuration')
    parser.add_argument('--config', nargs='?', type=str,
                        default='configs/gan_iam.yml')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    cfg = yaml2config(args.config)

    from lib.path_config import set_data_height, set_img_height
    model_height = int(getattr(cfg, 'img_height', 64))
    if model_height not in (32, 64):
        raise ValueError('ALPHA supports native image heights 32 and 64')
    set_img_height(model_height)
    set_data_height(int(getattr(cfg, 'source_height', model_height)))

    run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M-%S') + f'-{os.getpid()}'
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] + '-' + run_id)
    ckpt_dir = os.path.join(logdir, getattr(cfg.training, 'ckpt_dir', 'ckpts'))
    os.makedirs(ckpt_dir, exist_ok=True)

    local_rank = args.local_rank
    if local_rank == -1 and 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    cfg.local_rank = local_rank
    if local_rank > -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')

    seed_everything(int(getattr(cfg, 'seed', 123456)) + max(local_rank, 0))

    started_at = time.time()
    status_path = os.environ.get('HGGAN_STATUS_PATH', '')
    branch = os.path.basename(os.path.abspath(os.getcwd()))
    job = os.environ.get('HGGAN_JOB_NAME', branch)
    job_status = 'running'
    error_text = ''
    update_job_status(status_path, job_status, job=job, branch=branch,
                      config=args.config, model=getattr(cfg, 'model', 'unknown'),
                      logdir=os.path.abspath(logdir), checkpoint_dir=os.path.abspath(ckpt_dir), pid=os.getpid(), started_at=started_at)

    wandb_run = init_wandb_run(cfg)
    write_wandb_log('[Startup] job={} branch={} model={} config={} logdir={}'.format(
        job, branch, getattr(cfg, 'model', 'unknown'), args.config, logdir
    ))
    model = None
    try:
        model = get_model(cfg.model)(cfg, logdir)
        model.train()
        job_status = 'completed'
    except KeyboardInterrupt:
        job_status = 'interrupted'
        error_text = 'Training interrupted by user.'
        print(error_text)
        write_wandb_log(error_text)
    except Exception:
        job_status = 'failed'
        error_text = traceback.format_exc()
        traceback.print_exc()
        write_wandb_log(error_text)
        raise
    finally:
        results_path = ''
        results = {}
        try:
            results_path, results = write_results_table(
                logdir, cfg, model=model, status=job_status, started_at=started_at,
                error=error_text, metadata={'job': job, 'branch': branch, 'config': args.config},
            )
        except Exception as exc:
            results = {'results_error': str(exc)}
            print('Could not write RESULTS table: {}'.format(exc))
        if wandb_run is not None:
            try:
                import wandb
                if wandb.run is not None:
                    write_wandb_log('[Shutdown] Finishing W&B run.')
                    wandb.finish()
            except Exception as exc:
                print('WandB shutdown warning: {}'.format(exc))

        update_job_status(status_path, job_status, job=job, branch=branch,
                          config=args.config, model=getattr(cfg, 'model', 'unknown'),
                          logdir=os.path.abspath(logdir), checkpoint_dir=os.path.abspath(ckpt_dir), pid=os.getpid(),
                          results_path=os.path.abspath(results_path) if results_path else '',
                          metrics=results,
                          error=error_text.strip().splitlines()[-1] if error_text else '')
