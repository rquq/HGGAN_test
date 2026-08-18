import os
import logging
import datetime
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from munch import Munch
from torchvision.utils import make_grid
from PIL import Image


def get_logger(logdir):
    logger = logging.getLogger("gan")
    logger.setLevel(logging.INFO)

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))

    formatter = logging.Formatter('%(message)s')

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


def yaml2config(yml_path):
    with open(yml_path, 'r', encoding='utf-8') as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)

    def to_munch(d):
        if isinstance(d, dict):
            return Munch({k: to_munch(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [to_munch(v) for v in d]
        return d

    cfg = to_munch(data)
    return cfg


def draw_image(tensor, nrow=8, padding=2,
               normalize=False, value_range=None, scale_each=False, pad_value=0):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, value_range=value_range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add(0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return ndarr


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