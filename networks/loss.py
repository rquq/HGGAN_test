import torch
import torch.nn as nn
import torch.nn.functional as F


def _len2mask(length, max_len=None, dtype=torch.float32):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if length.numel() == 0:
        return torch.empty((0, 0), device=length.device, dtype=dtype or torch.float32)
    max_len = max_len or int(length.max().item())
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def tv_loss(img, img_lens):
    loss = (recn_l1_loss(img[:, :, 1:, :], img[:, :, :-1, :], img_lens) +
            recn_l1_loss(img[:, :, :, 1:], img[:, :, :, :-1], img_lens - 1)) / 2
    return loss


def recn_l1_loss(img1, img2, img_lens):
    mask = _len2mask(img_lens, img1.size(-1)).to(img1.device)
    diff_img = (img1 - img2) * mask.view(mask.size(0), 1, 1, mask.size(1))
    loss = diff_img.abs().sum() / (diff_img.size(1) * diff_img.size(2) * torch.clamp(img_lens.sum(), min=1))
    return loss


def calc_loss_perceptual(hout, hgt, img_lens):
    loss = 0
    for j in range(3):
        scale = 2 ** (3 - j)
        loss += recn_l1_loss(hout[j], hgt[j], img_lens // scale) / scale
    return loss


def KLloss(mu, logvar):
    # Handle both 2D [B, D] and 3D [B, L, D] cases
    loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    # Sum over all dimensions except batch, then mean over batch
    loss = loss.view(loss.size(0), -1).sum(dim=1)
    return torch.mean(loss)


##############################################################################
# Contextual loss (Perceptual feature matching without rigid spatial alignment)
##############################################################################
class CXLoss(nn.Module):
    def __init__(self, sigma=0.5, b=1.0, similarity="cosine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(dim=(0, 2, 3), keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = torch.clamp(features.norm(p=2, dim=1, keepdim=True), min=1e-8)
        features = features.div(norms)
        return features

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''
        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        N, C, H_T, W_T = featureT.shape
        _, _, H_I, W_I = featureI.shape

        featI_flat = featureI.view(N, C, H_I * W_I)
        featT_flat = featureT.view(N, C, H_T * W_T)

        # batched matrix multiplication: (N, P_T, C) x (N, C, P_I) -> (N, P_T, P_I)
        dist = torch.bmm(featT_flat.transpose(1, 2), featI_flat)

        raw_dist = (1. - dist) / 2.
        relative_dist = self.calc_relative_distances(raw_dist, axis=1)

        CX = self.calc_CX(relative_dist, axis=1)
        
        # Take max over spatial dimensions of Inference feature map (dim=2, which is P_I)
        CX_max = CX.max(dim=2)[0]
        CX_mean = torch.mean(CX_max, dim=1)
        CX_loss = torch.mean(-torch.log(CX_mean + 1e-5))
        return CX_loss


##############################################################################
# Token-Level Contrastive Style Loss (InfoNCE)
##############################################################################
def contrastive_style_loss(fake_styles, real_styles, temperature=0.07):
    """
    Enforces stroke and texture consistency at the latent feature level.
    Supports both 2D (B, style_dim) and 3D (B, S, style_dim) multi-token style representations.
    For 3D token representations, preserves fine-grained token diversity via max-similarity matching.
    """
    if fake_styles.dim() == 3:
        # Token-level matching for multi-token style sequence (B, S, D)
        f_s = F.normalize(fake_styles, dim=-1) # (B, S, D)
        r_s = F.normalize(real_styles, dim=-1) # (B, S, D)
        
        # Batch-wide pairwise similarity matrix across tokens
        # f_s: (B, S, D), r_s: (B, S, D) -> logits: (B, B, S, S)
        logits = torch.einsum('b s d, c t d -> b c s t', f_s, r_s) / temperature
        
        # Max-pooled token similarity between batch items
        token_logits = logits.max(dim=-1)[0].mean(dim=-1) # (B, B)
        
        labels = torch.arange(f_s.size(0), device=fake_styles.device)
        return F.cross_entropy(token_logits, labels)
    else:
        f_s = F.normalize(fake_styles, dim=-1)
        r_s = F.normalize(real_styles, dim=-1)
        logits = torch.matmul(f_s, r_s.t()) / temperature
        labels = torch.arange(f_s.size(0), device=fake_styles.device)
        return F.cross_entropy(logits, labels)
