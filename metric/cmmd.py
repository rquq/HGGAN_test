# coding=utf-8
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
_CUDA_AVAILABLE = torch.cuda.is_available()
_SIGMA = 10
_SCALE = 1000

class ClipEmbeddingModel:
    def __init__(self):
        self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        if _CUDA_AVAILABLE:
            self._model = self._model.cuda()
        self.input_image_size = 336
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self._device_mean = {}
        self._device_std = {}

    @torch.no_grad()
    def embed(self, images):
        # images is a PyTorch tensor of shape (B, 3, 336, 336) in [0, 1] range on GPU
        device = images.device
        if device not in self._device_mean:
            self._device_mean[device] = self.mean.to(device)
            self._device_std[device] = self.std.to(device)
        mean = self._device_mean[device]
        std = self._device_std[device]
        normalized = (images - mean) / std
        
        image_embs = self._model(pixel_values=normalized).image_embeds
        image_embs = image_embs / torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs.cpu()


def mmd(x, y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)


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
        imgs = batch['org_imgs'].to(device, non_blocking=True)
        lens = batch['org_img_lens']
        
        batch_images = preprocess_images_gpu(imgs, lens, size, device)
        embs = embedding_model.embed(batch_images)
        real_embs_list.append(embs.numpy())
        
    return np.concatenate(real_embs_list, axis=0).astype("float32")


def calculate_cmmd_score(data_loader, generator, n_rand_repeat, device, n_batches=None, real_embeddings=None, embedding_model=None):
    if n_batches is None:
        n_batches = len(data_loader)
        
    if embedding_model is None:
        embedding_model = ClipEmbeddingModel()
        
    size = embedding_model.input_image_size
    
    if real_embeddings is None:
        real_embeddings = compute_real_embeddings(data_loader, embedding_model, n_batches, device=device)
        
    print("Extracting generated image embeddings for CMMD...")
    fake_embs_list = []
    for idx, batch in enumerate(tqdm(generator, total=n_batches * n_rand_repeat, desc='CMMD Fake Embeddings')):
        if idx >= n_batches * n_rand_repeat:
            break
        imgs = batch['org_imgs'].to(device, non_blocking=True)
        lens = batch['org_img_lens']
        
        batch_images = preprocess_images_gpu(imgs, lens, size, device)
        embs = embedding_model.embed(batch_images)
        fake_embs_list.append(embs.numpy())
        
    fake_embeddings = np.concatenate(fake_embs_list, axis=0).astype("float32")
    
    print("Computing CMMD Score...")
    score = mmd(real_embeddings, fake_embeddings)
    return score.item()
