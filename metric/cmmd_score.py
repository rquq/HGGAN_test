import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from .cmmd.distance import mmd

def tensor_to_numpy_cmmd(img_tensor, length, size):
    img_np = img_tensor.detach().cpu().numpy()
    if img_np.min() < 0:
        img_np = (img_np + 1) / 2
    img_np = np.clip(img_np, 0, 1)
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)
    # slice to the actual (non-padded) length
    img_np = img_np[:, :, :length]
    
    # Convert to PIL for easy center cropping and resizing
    img_np_255 = (img_np * 255.0).astype(np.uint8)
    img_np_255 = np.transpose(img_np_255, (1, 2, 0))
    im = Image.fromarray(img_np_255)
    
    # Center crop and resize to CLIP input size
    w, h = im.size
    l = min(w, h)
    top = (h - l) // 2
    left = (w - l) // 2
    box = (left, top, left + l, top + l)
    im = im.crop(box)
    im = im.resize((size, size), resample=Image.BICUBIC)
    
    # Return as numpy array in [0, 1] range
    return np.asarray(im).astype(np.float32) / 255.0


def compute_real_embeddings(data_loader, embedding_model, n_batches=None):
    if n_batches is None:
        n_batches = len(data_loader)
        
    size = embedding_model.input_image_size
    real_embs_list = []
    
    for batch in tqdm(data_loader, total=n_batches, desc='CMMD Real Embeddings'):
        imgs = batch['org_imgs']
        lens = batch['org_img_lens']
        
        batch_images = []
        for i in range(imgs.size(0)):
            np_img = tensor_to_numpy_cmmd(imgs[i], lens[i].item(), size)
            batch_images.append(np_img)
        
        batch_images = np.stack(batch_images, axis=0)  # (batch_size, 336, 336, 3)
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
        real_embeddings = compute_real_embeddings(data_loader, embedding_model, n_batches)
        
    # Process fake data
    print("Extracting generated image embeddings for CMMD...")
    fake_embs_list = []
    for batch in tqdm(generator, total=n_batches * n_rand_repeat, desc='CMMD Fake Embeddings'):
        imgs = batch['org_imgs']
        lens = batch['org_img_lens']
        
        batch_images = []
        for i in range(imgs.size(0)):
            np_img = tensor_to_numpy_cmmd(imgs[i], lens[i].item(), size)
            batch_images.append(np_img)
            
        batch_images = np.stack(batch_images, axis=0)  # (batch_size, 336, 336, 3)
        embs = embedding_model.embed(batch_images)
        fake_embs_list.append(embs.numpy())
        
    fake_embeddings = np.concatenate(fake_embs_list, axis=0).astype("float32")
    
    print("Computing CMMD Score...")
    score = mmd(real_embeddings, fake_embeddings)
    return score.item()
