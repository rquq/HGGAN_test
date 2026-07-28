import os
from datetime import datetime
import argparse

import random
import numpy as np
import torch
import torch.distributed as dist

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
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/gan_iam.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)

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

        run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
        run_id_tensor = torch.tensor([ord(c) for c in run_id], dtype=torch.long, device=local_rank)
        dist.broadcast(run_id_tensor, src=0)
        run_id = "".join([chr(c) for c in run_id_tensor.cpu().tolist()])
    else:
        if hasattr(cfg, "seed") and cfg.seed is not None:
            seed_everything(cfg.seed)
        run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')

    logdir = os.path.join("runs", os.path.basename(args.config)[:-4] + '-' + str(run_id))

    model = get_model(cfg.model)(cfg, logdir)
    model.train()

