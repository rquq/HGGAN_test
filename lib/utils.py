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
    from lib.path_config import set_source_polarity
    set_source_polarity(getattr(cfg, 'dataset_source_polarity', 'white'))
    from lib.path_config import set_img_height, set_data_height
    height = int(getattr(cfg, 'img_height', 64))
    set_img_height(height)
    set_data_height(int(getattr(cfg, 'source_height', height)))
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
    """Own one run in train.py; keep startup and RESULTS in the same log."""
    cfg = getattr(opt, 'wandb', {})
    if (int(getattr(opt, 'local_rank', -1)) > 0 or bool(getattr(opt, 'no_wandb', False))
            or not getattr(cfg, 'enabled', True)):
        return None
    try:
        import wandb
        branch = os.path.basename(os.path.abspath(os.getcwd()))
        project = getattr(cfg, 'project', project)
        key = os.environ.get('WANDB_API_KEY', '') or getattr(cfg, 'key', '')
        if not key and os.environ.get('WANDB_MODE', '').lower() not in ('disabled', 'offline'):
            for candidate in ('./wandb_key.txt', '../wandb_key.txt', '../../wandb_key.txt',
                              '/kaggle/working/wandb_key.txt'):
                if os.path.isfile(candidate):
                    with open(candidate, encoding='utf-8') as handle:
                        key = handle.read().strip()
                    if key:
                        break
        if key and os.environ.get('WANDB_MODE', '').lower() not in ('disabled', 'offline'):
            wandb.login(key=key)
        try:
            from wandb.sdk.wandb_run import Run
            direct_logs = hasattr(Run, 'write_logs')
        except ImportError:
            direct_logs = False
        name = getattr(cfg, 'name', None) or '{}_{}_{}px_{}'.format(
            branch, getattr(opt, 'model', 'model'), getattr(opt, 'img_height', 'unknown'),
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        )
        run_id = getattr(cfg, 'id', None)
        run = wandb.init(
            project=project, name=name, id=run_id,
            config=opt.toDict() if hasattr(opt, 'toDict') else dict(opt),
            resume=getattr(cfg, 'resume', 'allow') if run_id else None,
            settings=wandb.Settings(console='off' if direct_logs else 'redirect'),
        )
        wandb.define_metric('train/iter')
        for pattern in ('loss/*', 'train/lr_*', 'train/weight_*'):
            wandb.define_metric(pattern, step_metric='train/iter')
        wandb.define_metric('samples/epoch')
        wandb.define_metric('samples/*', step_metric='samples/epoch')
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
    if model is not None:
        best = getattr(model, 'best_fid', getattr(model, 'restored_metadata', {}).get('best_fid'))
        if best is not None and numpy.isfinite(float(best)):
            rows.append(('best_fid', float(best)))
        history = getattr(model, 'eval_history', []) or []
        candidates = [r for r in history if r.get('fid') not in (None, '') and numpy.isfinite(float(r['fid']))]
        if candidates:
            row = min(candidates, key=lambda r: float(r['fid']))
            if best is None or float(row['fid']) == float(best):
                rows.append(('best_fid_epoch', row['epoch']))
    for attr in ('best_cer', 'best_wrr'):
        if model is not None and hasattr(model, attr):
            rows.append((attr, getattr(model, attr)))
    ckpt_dir = os.path.join(logdir, getattr(getattr(opt, 'training', {}), 'ckpt_dir', 'ckpts'))
    candidates = [p for p in glob.glob(os.path.join(ckpt_dir, '*.pth'))
                  if not os.path.basename(p).startswith('.tmp_')]
    if candidates:
        last = [p for p in candidates if os.path.basename(p).startswith('last')]
        best = [p for p in candidates if os.path.basename(p).startswith('best')]
        rows.append(('latest_checkpoint', max(last or candidates, key=os.path.getmtime)))
        if best:
            rows.append(('best_checkpoint', max(best, key=os.path.getmtime)))
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


def build_gan_log(values, iteration, epoch, lrs, g_updates, d_updates):
    """Report real objectives separately from their raw terms and multipliers."""
    log = {'train/iter': iteration, 'train/epoch': epoch,
           'train/lr_g': lrs['g'], 'train/lr_d': lrs['d'], 'train/lr_p': lrs['p']}
    if d_updates:
        for name, source in {'d_total':'d_total', 'd_global_real':'real_disc_loss',
                             'd_global_fake':'fake_disc_loss', 'd_patch_real':'real_disc_loss_patch',
                             'd_patch_fake':'fake_disc_loss_patch'}.items():
            log['loss/' + name] = values[source]
    if g_updates:
        mapping = {'g_total':'g_total', 'g_adv_global':'adv_loss', 'g_adv_patch':'adv_loss_patch',
                   'g_ctc':'g_ctc_weighted', 'g_info':'g_info_weighted', 'g_writer':'g_writer_weighted',
                   'g_reconstruction':'g_recn_weighted', 'g_contextual':'g_context_weighted', 'g_kl':'g_kl_weighted',
                   'raw_ctc_rand':'ctc_rand', 'raw_ctc_style':'ctc_style', 'raw_ctc_reconstruction':'ctc_recn',
                   'raw_ctc_total':'fake_ctc_loss', 'raw_info':'info_loss', 'raw_writer':'fake_wid_loss',
                   'raw_reconstruction':'recn_loss', 'raw_contextual':'ctx_loss', 'raw_kl':'kl_loss',
                   'raw_adv_patch':'adv_patch_raw'}
        for name, source in mapping.items():
            log['loss/' + name] = values[source]
        for name in ('ctc', 'info', 'wid', 'recn'):
            log['train/weight_' + name] = values['gp_' + name]
    return log


def format_gan_losses(values, epoch, epochs, iteration, loader_len, lrs, g_updates, d_updates, training):
    number = lambda value: format(float(value), '.6g')
    g = number(values['g_total']) if g_updates else 'not-updated'
    lines = [f'[{epoch:3d}/{epochs}] [{(iteration-1) % loader_len:5d}/{loader_len}] '
             f'G-total:{g} D-total:{number(values["d_total"])} '
             f'updates(G/D):{g_updates}/{d_updates} '
             f'lr(G/D/P):{lrs["g"]:.6g}/{lrs["d"]:.6g}/{lrs["p"]:.6g}']
    if not getattr(training, 'detailed_loss_print', True):
        return '\n'.join(lines)
    if g_updates:
        lines.append('  G weighted | ' + ' '.join(f'{name}:{number(values[key])}' for name,key in (
            ('Adv-global','adv_loss'), ('Adv-patch','adv_loss_patch'), ('CTC','g_ctc_weighted'),
            ('Info','g_info_weighted'), ('Writer','g_writer_weighted'), ('L1-rec','g_recn_weighted'),
            ('Context','g_context_weighted'), ('KL','g_kl_weighted'))))
        lines.append('  G raw      | ' + ' '.join(f'{name}:{number(values[key])}' for name,key in (
            ('Adv-patch','adv_patch_raw'), ('CTC-rand','ctc_rand'), ('CTC-style','ctc_style'),
            ('CTC-recon','ctc_recn'), ('Info','info_loss'), ('Writer','fake_wid_loss'),
            ('L1-rec','recn_loss'), ('Context','ctx_loss'), ('KL','kl_loss'))))
        lines.append('  Weights    | ' + ' '.join(f'gp-{name}:{number(values["gp_"+name])}' for name in ('ctc','info','wid','recn'))
                     + f' | fixed patch-G:{getattr(training,"lambda_patch_adv",.45):.6g}'
                       f' context:{training.lambda_ctx:.6g} KL:{training.lambda_kl:.6g}')
    else:
        lines.append('  G: no optimizer update in this reporting window.')
    lines.append('  D terms    | ' + ' '.join(f'{name}:{number(values[key])}' for name,key in (
        ('Real-global','real_disc_loss'), ('Fake-global','fake_disc_loss'),
        ('Real-patch','real_disc_loss_patch'), ('Fake-patch','fake_disc_loss_patch'))))
    return '\n'.join(lines)


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

    def update_many(self, values, n=1):
        """Transfer detached scalar tensors together, with one host sync."""
        if not values:
            return
        keys = list(values)
        scalars = torch.stack([values[key].detach().reshape(()) for key in keys])
        for key, value in zip(keys, scalars.cpu().tolist()):
            self.update(key, value, n)


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
