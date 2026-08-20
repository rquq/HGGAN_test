import os, h5py
import numpy as np
from PIL import Image
import cv2
from copy import deepcopy
import itertools
import glob

import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import Compose, Normalize, ToTensor
from lib.alphabet import strLabelConverter
from lib.path_config import data_roots, data_paths, ImgHeight, CharWidth
from lib.transforms import RandomScale, RandomClip, MildHandwritingAugment



class Hdf5Dataset(Dataset):
    def __init__(self, root, split, transforms=None, alphabet_key='all', process_style=False, normalize_wid=True):
        super(Hdf5Dataset, self).__init__()
        self.root = root
        self._load_h5py(os.path.join(self.root, split), normalize_wid)
        self.transforms = transforms
        self.org_transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])
        self.label_converter = strLabelConverter(alphabet_key)
        self.process_style = process_style

    def _load_h5py(self, file_path, normalize_wid=True):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"HDF5 dataset file path does not exist: {self.file_path}")

        with h5py.File(self.file_path, 'r') as h5f:
            self.imgs, self.lbs = h5f['imgs'][:], h5f['lbs'][:]
            self.img_seek_idxs, self.lb_seek_idxs = h5f['img_seek_idxs'][:], h5f['lb_seek_idxs'][:]
            self.img_lens, self.lb_lens = h5f['img_lens'][:], h5f['lb_lens'][:]
            if 'wids' in h5f:
                self.wids = h5f['wids'][:]
            else:
                self.wids = np.zeros((len(self.img_lens),), dtype=np.int32)
            if normalize_wid and len(self.wids) > 0:
                self.wids -= self.wids.min()

    def __getitem__(self, idx):
        data = {}
        img_seek_idx, img_len = self.img_seek_idxs[idx], self.img_lens[idx]
        lb_seek_idx, lb_len = self.lb_seek_idxs[idx], self.lb_lens[idx]
        img = self.imgs[:, img_seek_idx : img_seek_idx + img_len]
        text = ''.join(chr(ch) for ch in self.lbs[lb_seek_idx : lb_seek_idx + lb_len])
        data['text'] = text
        lb = self.label_converter.encode(text)
        wid = self.wids[idx]
        data['lb'], data['wid'] = lb, wid

        data['org_img'] = self.org_transforms(Image.fromarray(img, mode='L'))

        # style image
        if self.process_style:
            h, w = img.shape[:2]
            new_w = CharWidth * len(text)
            dim = (new_w, ImgHeight)
            if new_w < w:
                style_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            else:
                style_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
            style_img = Image.fromarray(style_img, mode='L')

        else:
            style_img = Image.fromarray(img, mode='L')

        data['style_img'] = self.org_transforms(style_img)

        if self.transforms is not None:
            data['aug_img'] = self.transforms(style_img)

        return data

    def __len__(self):
        return len(self.img_lens)

    @staticmethod
    def _recalc_len(leng, scale=CharWidth):
        tmp = leng % scale
        return leng + scale - tmp if tmp != 0 else leng

    @staticmethod
    def collect_fn(batch):
        org_imgs, org_img_lens, style_imgs, style_img_lens, aug_imgs, aug_img_lens,\
        lbs, lb_lens, wids = [], [], [], [], [], [], [], [], []

        for data in batch:
            org_img, style_img, lb, wid = data['org_img'], data['style_img'], data['lb'], data['wid']
            aug_img = data['aug_img'] if 'aug_img' in data else None
            if isinstance(org_img, torch.Tensor):
                org_img = org_img.numpy()
            if isinstance(style_img, torch.Tensor):
                style_img = style_img.numpy()
            if aug_img is not None and isinstance(aug_img, torch.Tensor):
                aug_img = aug_img.numpy()

            org_imgs.append(org_img)
            org_img_lens.append(org_img.shape[-1])
            style_imgs.append(style_img)
            style_img_lens.append(style_img.shape[-1])
            lbs.append(lb)
            lb_lens.append(len(lb))
            wids.append(wid)
            if aug_img is not None:
                aug_imgs.append(aug_img)
                aug_img_lens.append(Hdf5Dataset._recalc_len(aug_img.shape[-1]))

        bdata = {}
        bz = len(lb_lens)
        org_h = max(img.shape[-2] for img in org_imgs)
        pad_org_img_max_len = Hdf5Dataset._recalc_len(max(org_img_lens))
        pad_org_imgs = np.full((bz, 1, org_h, pad_org_img_max_len), -1.0, dtype=np.float32)
        for i, (org_img, org_img_len) in enumerate(zip(org_imgs, org_img_lens)):
            pad_org_imgs[i, 0, :org_img.shape[-2], :org_img_len] = org_img
        bdata['org_imgs'] = torch.from_numpy(pad_org_imgs)
        bdata['org_img_lens'] = torch.tensor(org_img_lens, dtype=torch.int32)

        style_h = max(img.shape[-2] for img in style_imgs)
        pad_style_img_max_len = Hdf5Dataset._recalc_len(max(style_img_lens))
        pad_style_imgs = np.full((bz, 1, style_h, pad_style_img_max_len), -1.0, dtype=np.float32)
        for i, (style_img, style_img_len) in enumerate(zip(style_imgs, style_img_lens)):
            pad_style_imgs[i, 0, :style_img.shape[-2], :style_img_len] = style_img
        bdata['style_imgs'] = torch.from_numpy(pad_style_imgs)
        bdata['style_img_lens'] = torch.tensor(style_img_lens, dtype=torch.int32)

        pad_lbs = np.zeros((bz, max(lb_lens)), dtype=np.int64)
        for i, (lb, lb_len) in enumerate(zip(lbs, lb_lens)):
            pad_lbs[i, :lb_len] = lb
        bdata['lbs'] = torch.from_numpy(pad_lbs)
        bdata['lb_lens'] = torch.tensor(lb_lens, dtype=torch.int32)
        bdata['wids'] = torch.tensor(wids, dtype=torch.long)

        if len(aug_imgs) > 0:
            aug_h = max(img.shape[-2] for img in aug_imgs)
            pad_aug_imgs = np.full((bz, 1, aug_h, max(aug_img_lens)), -1.0, dtype=np.float32)
            for i, aug_img in enumerate(aug_imgs):
                pad_aug_imgs[i, 0, :aug_img.shape[-2], :aug_img.shape[-1]] = aug_img

            bdata['aug_imgs'] = torch.from_numpy(pad_aug_imgs)
            bdata['aug_img_lens'] = torch.tensor(aug_img_lens, dtype=torch.int32)

        return bdata

    @staticmethod
    def sort_collect_fn_style(batch):
        batch = Hdf5Dataset.collect_fn(batch)

        style_img_lens = batch['style_img_lens']
        _, idx = torch.sort(style_img_lens, descending=True)

        for key, val in batch.items():
            batch[key] = val[idx]
        return batch

    @staticmethod
    def sort_collect_fn_aug(batch):
        batch = Hdf5Dataset.collect_fn(batch)

        if 'aug_img_lens' not in batch:
            return batch

        style_img_lens = batch['aug_img_lens']
        _, idx = torch.sort(style_img_lens, descending=True)

        for key, val in batch.items():
            batch[key] = val[idx]
        return batch

    @staticmethod
    def merge_batch(batch1, batch2, device):
        lbs1, lb_lens1, wids1 = batch1['lbs'], batch1['lb_lens'], batch1['wids']
        lbs2, lb_lens2, wids2 = batch2['lbs'], batch2['lb_lens'], batch2['wids']
        bz1, bz2 = lb_lens1.size(0), lb_lens2.size(0)

        mbdata = {}
        for img_key, img_len_key in [('org_imgs', 'org_img_lens'),
                                     ('style_imgs', 'style_img_lens'),
                                     ('aug_imgs', 'aug_img_lens')]:
            if img_len_key not in batch1 or img_len_key not in batch2:
                continue

            imgs1, imgs2 =  batch1[img_key], batch2[img_key]
            img_lens1, img_lens2 = batch1[img_len_key], batch2[img_len_key]
            max_img_len = max(imgs1.size(-1), imgs2.size(-1))
            pad_imgs = torch.full((bz1 + bz2, imgs1.size(1), imgs1.size(2), max_img_len), -1.0, dtype=torch.float32, device=device)
            pad_imgs[:bz1, :, :, :imgs1.size(-1)] = imgs1
            pad_imgs[bz1:, :, :, :imgs2.size(-1)] = imgs2
            merge_img_lens = torch.cat([img_lens1, img_lens2]).to(device)

            mbdata[img_key] = pad_imgs
            mbdata[img_len_key] = merge_img_lens

        max_lb_len = max(lbs1.size(-1), lbs2.size(-1))
        pad_lbs = torch.zeros((bz1 + bz2, max_lb_len), dtype=torch.long, device=device)
        pad_lbs[:bz1, :lbs1.size(-1)] = lbs1
        pad_lbs[bz1:, :lbs2.size(-1)] = lbs2
        mbdata['lbs'] = pad_lbs
        merge_lb_lens = torch.cat([lb_lens1, lb_lens2]).to(device)
        mbdata['lb_lens'] = merge_lb_lens
        merge_wids = torch.cat([wids1, wids2]).long().to(device)
        mbdata['wids'] = merge_wids
        return mbdata

    @staticmethod
    def gen_h5file(all_imgs, all_texts, all_wids, save_path):
        img_seek_idxs, img_lens = [], []
        cur_seek_idx = 0
        for img in all_imgs:
            img_seek_idxs.append(cur_seek_idx)
            img_lens.append(img.shape[-1])
            cur_seek_idx += img.shape[-1]

        lb_seek_idxs, lb_lens = [], []
        cur_seek_idx = 0
        for lb in all_texts:
            lb_seek_idxs.append(cur_seek_idx)
            lb_lens.append(len(lb))
            cur_seek_idx += len(lb)

        save_imgs = np.concatenate(all_imgs, axis=-1)
        save_texts = list(itertools.chain(*all_texts))
        save_lbs = [ord(ch) for ch in save_texts]
        with h5py.File(save_path, 'w') as h5f:
            h5f.create_dataset('imgs',
                               data=save_imgs,
                               compression='gzip',
                               compression_opts=4,
                               dtype=np.uint8)
            h5f.create_dataset('lbs',
                               data=save_lbs,
                               dtype=np.int32)
            h5f.create_dataset('img_seek_idxs',
                               data=img_seek_idxs,
                               dtype=np.int64)
            h5f.create_dataset('img_lens',
                               data=img_lens,
                               dtype=np.int32)
            h5f.create_dataset('lb_seek_idxs',
                               data=lb_seek_idxs,
                               dtype=np.int64)
            h5f.create_dataset('lb_lens',
                               data=lb_lens,
                               dtype=np.int16)
            h5f.create_dataset('wids',
                               data=all_wids,
                               dtype=np.int16)
        print('save->', save_path)


class ImageDataset(Hdf5Dataset):
    ImgHeight = 64

    def __init__(self, *args, **kwargs):
        super(ImageDataset, self).__init__(*args, **kwargs)

    def _load_h5py(self, file_path, normalize_wid=True):
        assert os.path.exists(file_path), file_path + " does not exist!"

        fileExtensions = ["jpg", "jpeg", "png", "bmp", "gif"]
        listOfFiles = []
        for extension in fileExtensions:
            listOfFiles.extend(glob.glob(os.path.join(file_path, "*." + extension)))
            listOfFiles.extend(glob.glob(os.path.join(file_path, "*." + extension.upper())))

        all_imgs = []
        all_texts = []
        for fn in listOfFiles:
            img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Read image labels
            label_text = os.path.splitext(os.path.basename(fn))[0]

            # Normalize image-height
            h, w = img.shape[:2]
            r = self.ImgHeight / float(h)
            new_w = max(int(w * r), int(self.ImgHeight / 4 * len(label_text)))
            dim = (new_w, self.ImgHeight)
            if new_w < w:
                resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            else:
                resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
            res_img = 255 - resize_img

            all_imgs.append(res_img)
            all_texts.append(label_text)

        if len(all_imgs) == 0:
            raise ValueError(f"No valid image files found in {file_path}")

        '''========prepare image dataset==========='''
        img_seek_idxs, img_lens = [], []
        cur_seek_idx = 0
        for img in all_imgs:
            img_seek_idxs.append(cur_seek_idx)
            img_lens.append(img.shape[-1])
            cur_seek_idx += img.shape[-1]

        lb_seek_idxs, lb_lens = [], []
        cur_seek_idx = 0
        for lb in all_texts:
            lb_seek_idxs.append(cur_seek_idx)
            lb_lens.append(len(lb))
            cur_seek_idx += len(lb)

        self.imgs = np.concatenate(all_imgs, axis=-1).astype(np.uint8)
        save_texts = list(itertools.chain(*all_texts))
        self.lbs = [ord(ch) for ch in save_texts]
        self.img_seek_idxs, self.lb_seek_idxs =\
            np.array(img_seek_idxs).astype(np.int64), np.array(lb_seek_idxs).astype(np.int64)
        self.img_lens, self.lb_lens = \
            np.array(img_lens).astype(np.int32), np.array(lb_lens).astype(np.int32)
        self.wids = np.zeros((len(all_imgs),)).astype(np.int32)


def get_dataset(dset_name, split, wid_aug=False, recogn_aug=False, process_style=False):
    name = dset_name.strip()
    tag = name.split('_')[0]
    alphabet_key = 'rimes_word' if tag.startswith('rimes') else 'all'

    transforms = [ToTensor(), Normalize([0.5], [0.5])]
    if recogn_aug:
        # Safe OCR-only perturbations: preserve glyph identity and writer style
        # while covering scan/ink variation seen by the GAN recognizer loss.
        transforms = [RandomScale(), MildHandwritingAugment()] + transforms
    if wid_aug:
        transforms = [RandomClip()] + transforms
    if not recogn_aug and not wid_aug:
        transforms = None
    else:
        transforms = Compose(transforms)

    if dset_name.startswith('custom'):
        dataset = ImageDataset(root=split, split='',
                               transforms=transforms,
                               alphabet_key=alphabet_key,
                               process_style=process_style)
    else:
        dataset = Hdf5Dataset(data_roots[tag],
                              data_paths[name][split],
                              transforms=transforms,
                              alphabet_key=alphabet_key,
                              process_style=process_style)
    return dataset


def get_collect_fn(sort_input=False, sort_style=True):
    if sort_input:
        if sort_style:
            return Hdf5Dataset.sort_collect_fn_style
        else:
            return Hdf5Dataset.sort_collect_fn_aug
    else:
        return Hdf5Dataset.collect_fn


def get_alphabet_from_corpus(corpus_path):
    items = []
    with open(corpus_path, 'r') as f:
        for line in f.readlines():
            items.append(line.strip())
    alphabet = ''.join(sorted(list(set(''.join(items)))))
    return alphabet


def get_max_image_width(dset):
    if hasattr(dset, 'img_lens') and dset.img_lens is not None:
        return int(dset.img_lens.max())
    max_image_width = 0
    for data in dset:
        max_image_width = max(max_image_width, data['org_img'].size(-1))
    return max_image_width
