import os
from datetime import datetime
import argparse

import random
import numpy as np
import torch

from lib.utils import yaml2config
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/gan_iam.yml",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
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

    local_rank = args.local_rank
    if local_rank == -1 and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    cfg.local_rank = local_rank

    if local_rank > -1:
        import torch
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        
    if hasattr(cfg, "seed") and cfg.seed is not None:
        seed_everything(cfg.seed)
    else:
        seed_everything(123456)

    model = get_model(cfg.model)(cfg, logdir)
    model.train()
