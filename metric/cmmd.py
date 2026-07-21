# coding=utf-8
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
_CUDA_AVAILABLE = torch.cuda.is_available()
_SIGMA = 10
_SCALE = 1000

def get_clip_model_path_and_cache():
    import os
    
    # Resolve the shared cache directory inside HGGAN_test/pretrained/hf_cache
    try:
        metric_dir = os.path.dirname(os.path.abspath(__file__))
        branch_root = os.path.dirname(metric_dir)
        project_root = os.path.dirname(branch_root)
        cache_dir = os.path.join(project_root, "pretrained", "hf_cache")
        save_dir = os.path.join(project_root, "pretrained", "clip-vit-large-patch14-336")
    except Exception:
        cache_dir = "./pretrained/hf_cache"
        save_dir = "./pretrained/clip-vit-large-patch14-336"

    # 1. Check if running on Kaggle and search for an attached dataset/model with CLIP
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            if "config.json" in files:
                has_weights = any(f.endswith(".bin") or f.endswith(".safetensors") for f in files)
                if has_weights and ("clip-vit-large-patch14-336" in root.lower() or "clip" in root.lower()):
                    return root, cache_dir, save_dir, True

    # 2. Check if already present in the shared clean save_dir
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "config.json")):
        files = os.listdir(save_dir)
        has_weights = any(f.endswith(".bin") or f.endswith(".safetensors") for f in files)
        if has_weights:
            return save_dir, cache_dir, save_dir, True

    # 3. Check if already present in the shared project cache_dir
    if os.path.exists(cache_dir):
        for root, dirs, files in os.walk(cache_dir):
            if "config.json" in files:
                has_weights = any(f.endswith(".bin") or f.endswith(".safetensors") for f in files)
                if has_weights:
                    return root, cache_dir, save_dir, True
                    
    return "openai/clip-vit-large-patch14-336", cache_dir, save_dir, False


class ClipEmbeddingModel:
    def __init__(self, device=None):
        import os
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            try:
                import huggingface_hub
                huggingface_hub.login(token=hf_token, write_permission=False)
            except Exception:
                pass

        model_path, cache_dir, save_dir, is_local = get_clip_model_path_and_cache()
        
        if is_local:
            try:
                self._model = CLIPVisionModelWithProjection.from_pretrained(
                    model_path, local_files_only=True
                ).eval()
            except Exception:
                self._model = CLIPVisionModelWithProjection.from_pretrained(
                    "openai/clip-vit-large-patch14-336", cache_dir=cache_dir, local_files_only=False, token=hf_token
                ).eval()
                # Save to output for future offline use
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    self._model.save_pretrained(save_dir)
                    print(f"Saved CLIP model to outputs: {save_dir}")
                except Exception as e:
                    print(f"Warning: Could not save CLIP model to outputs: {e}")
        else:
            try:
                self._model = CLIPVisionModelWithProjection.from_pretrained(
                    "openai/clip-vit-large-patch14-336", cache_dir=cache_dir, local_files_only=True
                ).eval()
            except Exception:
                self._model = CLIPVisionModelWithProjection.from_pretrained(
                    "openai/clip-vit-large-patch14-336", cache_dir=cache_dir, local_files_only=False, token=hf_token
                ).eval()
                
            # If successfully downloaded/loaded from Hub, save a clean copy to output save_dir
            if not os.path.exists(save_dir) or not os.path.exists(os.path.join(save_dir, "config.json")):
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    self._model.save_pretrained(save_dir)
                    print(f"Saved CLIP model to outputs: {save_dir}")
                except Exception as e:
                    print(f"Warning: Could not save CLIP model to outputs: {e}")
                
        if device is not None:
            self._model = self._model.to(device)
        elif _CUDA_AVAILABLE:
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

    x_sqnorms = torch.sum(x**2, dim=-1)
    y_sqnorms = torch.sum(y**2, dim=-1)

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
        embedding_model = ClipEmbeddingModel(device)
        
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
