# coding=utf-8
from transformers import CLIPVisionModelWithProjection
import torch

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
_CUDA_AVAILABLE = torch.cuda.is_available()

class ClipEmbeddingModel:
    def __init__(self):
        self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        if _CUDA_AVAILABLE:
            self._model = self._model.cuda()
        self.input_image_size = 336
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(self, images):
        # images is a PyTorch tensor of shape (B, 3, 336, 336) in [0, 1] range on GPU
        device = images.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        normalized = (images - mean) / std
        
        image_embs = self._model(pixel_values=normalized).image_embeds
        image_embs = image_embs / torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs.cpu()
