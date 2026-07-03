import numpy as np
import torch
from tqdm import tqdm
from .cmmd.distance import mmd

def preprocess_images_gpu(imgs, lens, size, device):
    batch_size = imgs.size(0)
    
    # Check min on batch level to avoid batch_size CPU-GPU syncs
    if (imgs.min() < 0).item():
        imgs = (imgs + 1) / 2
    imgs = torch.clamp(imgs, 0, 1)
    
    preprocessed = []
    for i in range(batch_size):
        img = imgs[i]
        length = lens[i].item()
        
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
            
        img = img[:, :, :length]
        
        c, h, w = img.shape
        l = min(h, w)
        top = (h - l) // 2
        left = (w - l) // 2
        img = img[:, top:top+l, left:left+l]
        
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(size, size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        preprocessed.append(img)
        
    return torch.stack(preprocessed, dim=0)

def compute_real_embeddings(data_loader, embedding_model, n_batches=None, device='cuda'):
    if n_batches is None:
        n_batches = len(data_loader)
        
    size = embedding_model.input_image_size
    real_embs_list = []
    
    for idx, batch in enumerate(tqdm(data_loader, total=n_batches, desc='CMMD Real Embeddings')):
        if idx >= n_batches:
            break
        imgs = batch['org_imgs'].to(device)
        lens = batch['org_img_lens']
        
        batch_images = preprocess_images_gpu(imgs, lens, size, device)
        embs = embedding_model.embed(batch_images)
        real_embs_list.append(embs.numpy())
        
    return np.concatenate(real_embs_list, axis=0).astype("float32")

def calculate_cmmd_score(data_loader, generator, n_rand_repeat, device, n_batches=None, real_embeddings=None, embedding_model=None):
    if n_batches is None:
        n_batches = len(data_loader)
        
    if embedding_model is None:
        from .cmmd.embedding import ClipEmbeddingModel
        embedding_model = ClipEmbeddingModel()
        
    size = embedding_model.input_image_size
    
    if real_embeddings is None:
        real_embeddings = compute_real_embeddings(data_loader, embedding_model, n_batches, device=device)
        
    print("Extracting generated image embeddings for CMMD...")
    fake_embs_list = []
    for idx, batch in enumerate(tqdm(generator, total=n_batches * n_rand_repeat, desc='CMMD Fake Embeddings')):
        if idx >= n_batches * n_rand_repeat:
            break
        imgs = batch['org_imgs'].to(device)
        lens = batch['org_img_lens']
        
        batch_images = preprocess_images_gpu(imgs, lens, size, device)
        embs = embedding_model.embed(batch_images)
        fake_embs_list.append(embs.numpy())
        
    fake_embeddings = np.concatenate(fake_embs_list, axis=0).astype("float32")
    
    print("Computing CMMD Score...")
    score = mmd(real_embeddings, fake_embeddings)
    return score.item()
