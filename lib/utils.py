import os
import glob
import json
import logging
import datetime
import numpy
from munch import Munch
import matplotlib.pyplot as plt
import torch


def get_logger(logdir):
    logger = logging.getLogger("gan")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))

    # File handler
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger


import yaml, munch
def yaml2config(yml_path):
    with open(yml_path) as fp:
        json = yaml.load(fp, Loader=yaml.FullLoader)

    def to_munch(json):
        for key, val in json.items():
            if isinstance(val, dict):
                json[key] = to_munch(val)
        return munch.Munch(json)

    cfg = to_munch(json)
    return cfg


def _result_scalar(value):
    if value is None:
        return ''
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item() if value.numel() == 1 else str(value.detach().cpu().tolist())
    elif isinstance(value, numpy.generic):
        value = value.item()
    return value if isinstance(value, (bool, int, float, str)) else str(value)


def update_job_status(status_path, status, **fields):
    """Atomically publish completion state for the notebook monitor."""
    if not status_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(status_path)) or '.', exist_ok=True)
    temporary = str(status_path) + '.tmp-{}'.format(os.getpid())
    try:
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump({'status': str(status), **fields}, handle, indent=2,
                      default=_result_scalar)
            handle.write('\n')
        os.replace(temporary, status_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def write_wandb_log(message):
    """Send text explicitly to the W&B Logs tab when a run is active."""
    try:
        import wandb
        write_logs = getattr(wandb.run, 'write_logs', None) if wandb.run else None
        if callable(write_logs):
            write_logs(str(message))
    except Exception:
        pass


def init_wandb_run(opt, project='HiGANplus'):
    """Start W&B before model construction so startup messages are retained."""
    if int(getattr(opt, 'local_rank', -1)) > 0 or bool(getattr(opt, 'no_wandb', False)):
        return None
    try:
        import wandb
        branch = os.path.basename(os.path.abspath(os.getcwd()))
        cfg_wandb = getattr(opt, 'wandb', {})
        project = getattr(cfg_wandb, 'project', project)
        key = os.environ.get('WANDB_API_KEY', '') or getattr(cfg_wandb, 'key', '')
        if key:
            wandb.login(key=key)
        run = wandb.init(
            project=project,
            name='{}_{}_src{}_{}'.format(
                branch, getattr(opt, 'model', 'model'),
                getattr(opt, 'source_height', getattr(opt, 'img_height', 'unknown')),
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
            ),
            config=dict(opt),
            resume='allow',
            settings=wandb.Settings(console='off'),
        )
        wandb.define_metric('pretrain/epoch')
        wandb.define_metric('pretrain/*', step_metric='pretrain/epoch')
        wandb.define_metric('valid/epoch')
        wandb.define_metric('valid/*', step_metric='valid/epoch')
        write_wandb_log('[WandB] startup logging is active before model construction.')
        return run
    except Exception as exc:
        print('WandB initialization skipped or failed: {}'.format(exc))
        return None


def write_results_table(logdir, opt, model=None, status='completed', started_at=None,
                        error='', metadata=None):
    """Write the compact RESULTS table consumed by the Kaggle monitor."""
    import time
    os.makedirs(logdir, exist_ok=True)
    metadata = metadata or {}
    metrics = dict(getattr(model, 'last_eval_scores', {}) or {}) if model is not None else {}
    rows = [
        ('status', status), ('job', metadata.get('job', '')),
        ('branch', metadata.get('branch', '')), ('config', metadata.get('config', '')),
        ('model', getattr(opt, 'model', 'unknown')), ('dataset', getattr(opt, 'dataset', 'unknown')),
        ('image_height', getattr(opt, 'img_height', '')),
        ('source_height', getattr(opt, 'source_height', getattr(opt, 'img_height', ''))),
        ('requested_epochs', getattr(getattr(opt, 'training', {}), 'epochs', '')),
        ('completed_epoch', getattr(model, 'completed_epoch', '')),
        ('batch_size', getattr(getattr(opt, 'training', {}), 'batch_size', '')),
        ('elapsed_seconds', round(time.time() - started_at, 2) if started_at else ''),
    ]
    for key, value in sorted(metrics.items()):
        rows.append(('last_{}'.format(str(key).lower()), value))
    for attr in ('best_cer', 'best_wrr'):
        if model is not None and hasattr(model, attr):
            rows.append((attr, getattr(model, attr)))
    ckpt_dir = os.path.join(logdir, getattr(getattr(opt, 'training', {}), 'ckpt_dir', 'ckpts'))
    candidates = [p for p in glob.glob(os.path.join(ckpt_dir, '*.pth'))
                  if not os.path.basename(p).startswith('.tmp_')]
    if candidates:
        rows.append(('latest_checkpoint', max(candidates, key=os.path.getmtime)))
    if error:
        rows.append(('error', str(error).strip().splitlines()[-1]))
    width = max([len(str(key)) for key, _ in rows] + [7])
    table = ['==================== RESULTS ====================']
    table.extend('{:<{}} | {}'.format(str(key), width, _result_scalar(value)) for key, value in rows)
    table.append('====================================================')
    text = '\n'.join(table)
    path = os.path.join(logdir, 'RESULTS.txt')
    temporary = path + '.tmp-{}'.format(os.getpid())
    with open(temporary, 'w', encoding='utf-8') as handle:
        handle.write(text + '\n')
    os.replace(temporary, path)
    print(text)
    write_wandb_log(text)
    return path, dict(rows)


from torchvision.utils import make_grid
def draw_image(tensor, nrow=8, padding=2,
           normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    try:
        grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, value_range=range, scale_each=scale_each)
    except TypeError:
        grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).cpu().numpy().astype(numpy.uint8)
    return ndarr

import cv2
def plot_heatmap(arr):
    heatmapshow = cv2.normalize(arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    return heatmapshow

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def eval(self):
        return self.avg


class AverageMeterManager(object):
    def __init__(self, keys):
        self.meters = {}
        for key in keys:
            self.meters[key] = AverageMeter()

    def reset(self, key):
        self.meters[key].reset()

    def reset_all(self):
        for key in self.meters.keys():
            self.meters[key].reset()

    def update(self, key, val, n=1):
        self.meters[key].update(val, n)

    def eval(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        res = {}
        for key in keys:
            res[key] = self.meters[key].eval()
        return res

    def eval_all(self):
        res = {}
        for key in self.meters.keys():
            res[key] = self.meters[key].eval()
        return res


def option_to_string(opt, row_blanks=20):
    def opt_to_str(opt, depth=0):
        res = ''
        for key, val in opt.items():
            if isinstance(val, Munch) or isinstance(val, dict):
                res += '-'*row_blanks + '\n' + key + '\n' + opt_to_str(val, depth + 2)
            else:
                res += '{}{}: {}\n'.format('|' + '-' * depth, key, val)
        return res

    res = '='*row_blanks + '\nRoot\n' + '-'*row_blanks + '\n' + opt_to_str(opt) + '='*row_blanks
    return res


def get_corpus(corpus_path):
    items = []
    with open(corpus_path, 'r') as f:
        for line in f.readlines():
            items.append(line.strip())
    return items


def show_image_pair(img1, img2, title1='', title2=''):
    plt.subplot(211)
    plt.imshow(img1, cmap='binary')
    plt.title(title1)
    plt.subplot(212)
    plt.imshow(img2, cmap='binary')
    plt.title(title2)
    plt.show()
