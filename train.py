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

    if hasattr(cfg, "seed") and cfg.seed is not None:
        seed_everything(cfg.seed)

    run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4] + '-' + str(run_id))

    model = get_model(cfg.model)(cfg, logdir)
    model.train()

