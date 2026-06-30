import torch
import torch.nn as nn
import torch.nn.functional as F


def _len2mask(length, max_len, dtype=torch.float32):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
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
    loss = diff_img.abs().sum() / (diff_img.size(1) * diff_img.size(2) * img_lens.sum())
    return loss


def calc_loss_perceptual(hout, hgt, img_lens):
    loss = 0
    for j in range(3):
        scale = 2 ** (3 - j)
        loss += recn_l1_loss(hout[j], hgt[j], img_lens // scale) / scale
    return loss


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def KLloss(mu, logvar):
    # Handle both 2D [B, D] and 3D [B, L, D] cases
    loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    # Sum over all dimensions except batch, then mean over batch
    loss = loss.view(loss.size(0), -1).sum(dim=1)
    return torch.mean(loss)


##############################################################################
# Contextual loss
##############################################################################
class CXLoss(nn.Module):
    def __init__(self, sigma=0.5, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms + 1e-8)
        return features

    def forward(self, featureT, featureI):
        featureI, featureT = self.center_by_T(featureI, featureT)
        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        B, C, H, W = featureT.shape
        P = H * W

        # Reshape to (B, C, P)
        featI_flat = featureI.view(B, C, P)
        featT_flat = featureT.view(B, C, P)

        # Compute cosine similarity matrix of shape (B, P_T, P_I)
        dist = torch.bmm(featT_flat.transpose(1, 2), featI_flat)

        raw_dist = (1. - dist) / 2.

        epsilon = 1e-5
        div = torch.min(raw_dist, dim=1, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)

        W = torch.exp((self.b - relative_dist) / self.sigma)
        W_sum = W.sum(dim=1, keepdim=True)
        CX = W.div(W_sum + 1e-8)

        # Max over P_I (dim=2), then mean over P_T (dim=1)
        CX = torch.mean(CX.max(dim=2)[0], dim=1)
        CX = torch.mean(-torch.log(CX + 1e-5))
        return CX


##############################################################################
# Gram style loss
##############################################################################
class GramStyleLoss(nn.Module):
    def __init__(self):
        super(GramStyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def __call__(self, input_feat, target_feat, feat_len=None):
        input_gram = self.gram(input_feat, feat_len)
        target_gram = self.gram(target_feat, feat_len)
        loss = self.criterion(input_gram, target_gram)
        return loss


class GramMatrix(nn.Module):
    def forward(self, input, feat_len=None):
        a, b, c, d = input.size()

        if feat_len is not None:
            # mask for varying lengths
            mask = _len2mask(feat_len, d).view(a, 1, 1, d)
            input = input * mask

        features = input.view(a, b, c * d)
        G = torch.bmm(features, features.transpose(1, 2))

        return G.div(b * c * d)


def contrastive_style_loss(fake_styles, real_styles, temperature=0.07):
    """
    Enforces stroke and texture consistency at the latent feature level.
    fake_styles: (B, 32, style_dim) - Extracted from generated images
    real_styles: (B, 32, style_dim) - Extracted from input real images
    """
    # Mean-pool to get style vectors
    f_s = F.normalize(fake_styles.mean(dim=1), dim=-1) # (B, D)
    r_s = F.normalize(real_styles.mean(dim=1), dim=-1) # (B, D)
    
    # Compute similarity matrix
    logits = torch.matmul(f_s, r_s.t()) / temperature # (B, B)
    labels = torch.arange(f_s.size(0)).to(fake_styles.device)
    
    # InfoNCE Loss
    loss = F.cross_entropy(logits, labels)
    return loss
