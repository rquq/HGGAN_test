import sys
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from .hwd.scores import HWDScore

class ImageListDataset(Dataset):
    def __init__(self, imgs, authors):
        self.imgs = imgs
        self.authors = authors
        self.transform = None
        self.path = ''
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        author = self.authors[idx]
        if self.transform:
            img = self.transform(img)
        return img, author, 0


def tensor_to_pil(img_tensor, length):
    # img_tensor shape: [C, H, W] values in range [-1, 1] usually or [0, 1]
    # In HGGAN, real images are usually passed as (imgs + 1) / 2 in get_activations
    img_np = img_tensor.detach().cpu().numpy()
    if img_np.min() < 0:
        img_np = (img_np + 1) / 2
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)
    img_np = img_np[:, :, :length]
    img_np = np.transpose(img_np, (1, 2, 0))
    return Image.fromarray(img_np)


def calculate_hwd_score(data_loader, generator, n_rand_repeat, device, n_batches=None):
    if n_batches is None:
        n_batches = len(data_loader)
        
    real_imgs_list = []
    real_authors_list = []
    
    fake_imgs_list = []
    fake_authors_list = []
    
    print("Extracting images for HWD calculation...")
    
    # Process real data
    for batch in tqdm(data_loader, total=n_batches, desc='Real Images'):
        imgs = batch['org_imgs']
        lens = batch['org_img_lens']
        # authors might not be available directly in HGGAN eval loader, 
        # so we mock author ids using idx or if 'wids' exists
        wids = batch.get('wids', torch.arange(imgs.size(0)))
        
        for i in range(imgs.size(0)):
            pil_img = tensor_to_pil(imgs[i], lens[i].item())
            real_imgs_list.append(pil_img)
            real_authors_list.append(str(wids[i].item()))
            
    # Process fake data
    for batch in tqdm(generator, total=n_batches * n_rand_repeat, desc='Fake Images'):
        imgs = batch['org_imgs']
        lens = batch['org_img_lens']
        wids = batch.get('wids', torch.arange(imgs.size(0)))
        
        for i in range(imgs.size(0)):
            pil_img = tensor_to_pil(imgs[i], lens[i].item())
            fake_imgs_list.append(pil_img)
            fake_authors_list.append(str(wids[i].item()))
            
    real_dataset = ImageListDataset(real_imgs_list, real_authors_list)
    fake_dataset = ImageListDataset(fake_imgs_list, fake_authors_list)
    
    print("Computing HWD Score...")
    hwd_scorer = HWDScore(height=32)
    score = hwd_scorer(fake_dataset, real_dataset, stream=True)
    return score
