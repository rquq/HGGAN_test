import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from scipy.stats import entropy
from sklearn.metrics.pairwise import polynomial_kernel
from torch.utils.data import Dataset

# Local imports
from metric.inception import InceptionV3 as ReferenceInceptionV3, fid_inception_v3, _inception_v3
from metric.hwd import HWDScore
from metric.cmmd import calculate_cmmd_score, compute_real_embeddings, ClipEmbeddingModel
from metric.mssim_psnr import calculate_mssim_psnr
from networks.utils import pad_image_lengths

# We need to wrap or subclass InceptionV3 to support masking and returning logits.
class HGGANInceptionV3(ReferenceInceptionV3):
    def __init__(self, output_blocks=[3], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        self.hggan_resize_input = resize_input
        super().__init__(output_blocks, resize_input=False, normalize_input=normalize_input, requires_grad=requires_grad, use_fid_inception=use_fid_inception)

        # We need self.last_fc for Inception Score calculation
        if use_fid_inception:
            inception_full = fid_inception_v3()
        else:
            inception_full = _inception_v3(pretrained=True)
        self.last_fc = inception_full.fc
        for param in self.last_fc.parameters():
            param.requires_grad = requires_grad

    def _len2mask(self, length, max_len, dtype=torch.float32):
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask

    def forward(self, inp, inp_len=None):
        if inp_len is None:
            # Standard FID/KID path: use parent's unmodified forward pass (preserves AdaptiveAvgPool2d)
            outp = super().forward(inp)
            return outp[-1], None

        # IS path: manual forward pass to get Mixed_7c before AdaptiveAvgPool2d
        x = inp
        if self.hggan_resize_input:
            x = F.interpolate(x, scale_factor=299 / x.size(2), mode='bilinear', align_corners=True)

        if self.normalize_input:
            x = 2 * x - 1

        # Run blocks 0, 1, 2
        for idx in range(3):
            x = self.blocks[idx](x)

        # Run block 3 up to Mixed_7c (all except final AdaptiveAvgPool2d)
        block3_layers = list(self.blocks[3].children())[:-1]
        for layer in block3_layers:
            x = layer(x)

        # Apply AvgPool2d(8, 8) to reduce height from 8 to 1
        last_feat = F.avg_pool2d(x, kernel_size=8, stride=8)
        squeezed = last_feat.squeeze(2)

        # Apply sequence length mask
        inp_mask = self._len2mask(inp_len, squeezed.size(-1))
        feat = squeezed * inp_mask.unsqueeze(1)
        feat = feat.sum(dim=-1) / (inp_len.unsqueeze(dim=1) + 1e-8)
        pooled_feat = feat.view(*feat.size(), 1, 1)

        logits = self.last_fc(pooled_feat.view(pooled_feat.size(0), -1)).softmax(dim=-1)
        return pooled_feat, logits

# Export CustomInceptionV3 as InceptionV3
InceptionV3 = HGGANInceptionV3

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
    img_np = img_tensor.detach().cpu().numpy()
    if img_np.min() < 0:
        img_np = (img_np + 1) / 2
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)
    img_np = img_np[:, :, :length]
    img_np = np.transpose(img_np, (1, 2, 0))
    return Image.fromarray(img_np)

def batch_tensor_to_pil_list(imgs_tensor, lengths_tensor):
    imgs_np = imgs_tensor.detach().cpu().numpy()
    lengths = lengths_tensor.detach().cpu().numpy()
    if imgs_np.min() < 0:
        imgs_np = (imgs_np + 1) / 2
    imgs_np = np.clip(imgs_np * 255.0, 0, 255).astype(np.uint8)
    if imgs_np.shape[1] == 1:
        imgs_np = np.repeat(imgs_np, 3, axis=1)
    pil_imgs = []
    for i in range(imgs_np.shape[0]):
        length = lengths[i]
        img_np = imgs_np[i, :, :, :length]
        img_np = np.transpose(img_np, (1, 2, 0))
        pil_imgs.append(Image.fromarray(img_np))
    return pil_imgs


# Activations computation
def get_activations(data_source, n_batches, model, dims, device, crop=False, eval_is=True):
    model.eval()
    pred_arr, pred_logits = [], []
    for idx, batch in enumerate(tqdm(data_source, total=n_batches)):
        if idx >= n_batches:
            break
        if isinstance(batch, dict):
            imgs = batch['org_imgs'].to(device, non_blocking=True)
            org_img_lens = batch['org_img_lens'].to(device, non_blocking=True)
        else:
            imgs, org_img_lens = batch[:2]
            imgs, org_img_lens = imgs.to(device, non_blocking=True), org_img_lens.to(device, non_blocking=True)

        # ----------------------------------------------------
        # 1. FID/KID preprocessing
        # ----------------------------------------------------
        if eval_is:
            imgs_is = imgs.clone()
            org_img_lens_is = org_img_lens.clone()

        # Replace background value of -1 with 1.0 starting from org_img_lens (Vectorized)
        batch_size, _, height, width = imgs.size()
        imgs_fid = imgs.clone()
        col_indices = torch.arange(width, device=device).view(1, 1, 1, width)
        padding_mask = col_indices >= org_img_lens.view(batch_size, 1, 1, 1)
        is_neg_one = (imgs_fid == -1)
        all_neg_one_in_padding = (is_neg_one | ~padding_mask).flatten(1).all(dim=1)
        replace_mask = padding_mask & all_neg_one_in_padding.view(batch_size, 1, 1, 1)
        imgs_fid.masked_fill_(replace_mask, 1.0)


        # Normalize to [0, 1]
        imgs_fid = (imgs_fid + 1) / 2

        # Repeat grayscale channels
        if imgs_fid.size(1) == 1:
            imgs_fid = imgs_fid.repeat(1, 3, 1, 1)

        # Pad or crop width to 4 * height
        target_width = 4 * height
        if width != target_width:
            if width < target_width:
                pad_width = target_width - width
                imgs_fid = torch.nn.functional.pad(imgs_fid, [0, pad_width, 0, 0], value=1.0)
            elif width > target_width:
                imgs_fid = imgs_fid[:, :, :, :target_width]

        # Resize to (299, 299) and normalize to [-1, 1]
        imgs_fid = torch.nn.functional.interpolate(
            imgs_fid, size=(299, 299), mode='bilinear', align_corners=False
        )
        imgs_fid = 2 * imgs_fid - 1

        # Run through model for FID/KID features
        orig_resize = getattr(model, 'resize_input', True)
        orig_normalize = getattr(model, 'normalize_input', True)
        model.resize_input = False
        model.normalize_input = False

        with torch.no_grad():
            pred, _ = model(imgs_fid)

        model.resize_input = orig_resize
        model.normalize_input = orig_normalize

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr.append(pred.cpu().data.numpy().reshape(pred.size(0), -1))

        # ----------------------------------------------------
        # 2. Original Baseline Preprocessing for IS Logits
        # ----------------------------------------------------
        if eval_is:
            img_lens_is = pad_image_lengths(org_img_lens_is, scale=height)
            imgs_is = (imgs_is + 1) / 2
            if imgs_is.size(1) == 1:
                imgs_is = imgs_is.repeat(1, 3, 1, 1)

            with torch.no_grad():
                if not crop:
                    _, logits = model(imgs_is, img_lens_is // height)
                else:
                    _, logits = model(imgs_is[:, :, :, :height * 2],
                                      2 * torch.ones((imgs_is.size(0),)).to(device))
            pred_logits.append(logits.cpu().data.numpy())

    pred_arr = np.concatenate(pred_arr, axis=0)
    assert pred_arr.shape[-1] == dims

    if eval_is:
        pred_logits = np.concatenate(pred_logits, axis=0)
        return pred_arr, pred_logits
    else:
        return pred_arr, None

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print('Warning: Imaginary component {} in FID sqrtm calculation. Proceeding with real part.'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(*args, **kwargs):
    # Retrieve eval_is from kwargs or defaults
    eval_is = kwargs.pop('eval_is', True)
    act, logits = get_activations(*args, **kwargs, eval_is=eval_is)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return act, mu, sigma, logits

def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True, output=sys.stdout, device=None,
                            **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    if subset_size > len(codes_g):
        subset_size = len(codes_g)
    if subset_size > len(codes_r):
        subset_size = len(codes_r)

    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(
                g, r, **kernel_args, var_at_m=m, ret_var=ret_var,
                device=device,
            )
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds

def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        X_t = torch.from_numpy(codes_g).to(device)
        Y_t = torch.from_numpy(codes_r).to(device)

        if gamma is None:
            gamma = 1.0 / X_t.shape[1]

        K_XX = (gamma * torch.matmul(X_t, X_t.T) + coef0) ** degree
        K_YY = (gamma * torch.matmul(Y_t, Y_t.T) + coef0) ** degree
        K_XY = (gamma * torch.matmul(X_t, Y_t.T) + coef0) ** degree

        K_XX_np = K_XX.cpu().numpy()
        K_YY_np = K_YY.cpu().numpy()
        K_XY_np = K_XY.cpu().numpy()
    else:
        from sklearn.metrics.pairwise import polynomial_kernel
        K_XX_np = polynomial_kernel(codes_g, degree=degree, gamma=gamma, coef0=coef0)
        K_YY_np = polynomial_kernel(codes_r, degree=degree, gamma=gamma, coef0=coef0)
        K_XY_np = polynomial_kernel(codes_g, codes_r, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX_np, K_XY_np, K_YY_np, var_at_m=var_at_m, ret_var=ret_var)

def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)

def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)
        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()
        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est

def calculate_inception_score(logits, splits=1):
    split_scores = []
    N = logits.shape[0]

    for k in range(splits):
        part = logits[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)

def calculate_fid_kid_is(cfg, data_loader, generator, n_rand_repeat, device, crop=False, real_stats=None, n_batches=None, inceptionV3_model=None):
    eval_fid = bool(getattr(cfg, 'validate_fid', False))
    eval_kid = bool(getattr(cfg, 'validate_kid', False))
    legacy_is = getattr(cfg, 'validate_is', None)
    eval_is_gen = bool(getattr(
        cfg, 'validate_is_gen', legacy_is if legacy_is is not None else False
    ))
    eval_is_org = bool(getattr(
        cfg, 'validate_is_org', legacy_is if legacy_is is not None else False
    ))
    eval_is = eval_is_gen or eval_is_org

    res = {}
    if not (eval_fid or eval_kid or eval_is):
        return res
    if inceptionV3_model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inceptionV3_model = InceptionV3([block_idx])
        inceptionV3_model.to(device)
        inceptionV3_model.eval()

    if n_batches is None:
        n_batches = len(data_loader)

    with torch.no_grad():
        act2, m2, s2, logits2 = calculate_activation_statistics(generator, n_batches * n_rand_repeat, inceptionV3_model,
                                                                cfg.dims, device, crop, eval_is=eval_is)
        if real_stats is None:
            act1, m1, s1, logits1 = calculate_activation_statistics(data_loader, n_batches, inceptionV3_model,
                                                                    cfg.dims, device, crop, eval_is=eval_is)
        else:
            act1, m1, s1, logits1 = real_stats

    if eval_fid:
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        res['fid'] = fid_value

    if eval_is_gen:
        res['is_gen'] = calculate_inception_score(logits2)
    if eval_is_org:
        res['is_org'] = calculate_inception_score(logits1)

    if eval_kid:
        ret = polynomial_mmd_averages(
                act1, act2, degree=cfg.mmd_degree, gamma=cfg.mmd_gamma,
                coef0=cfg.mmd_coef0, ret_var=cfg.mmd_var,
                n_subsets=cfg.mmd_subsets, subset_size=cfg.mmd_subset_size,
                device=device)

        if cfg.mmd_var:
            mmd2s, vars = ret
        else:
            mmd2s = ret
        kid = mmd2s.mean() * 100
        res['kid'] = kid

    return res

# Handwriting Distance (HWD) Wrapper
def calculate_hwd_score(
    data_loader, generator, n_rand_repeat, device, n_batches=None,
    real_dataset=None, real_features=None, batchsize=32,
):
    if n_batches is None:
        n_batches = len(data_loader)

    fake_imgs_list = []
    fake_authors_list = []

    if real_features is None and real_dataset is None:
        real_imgs_list = []
        real_authors_list = []
        print("Extracting images for HWD calculation...")
        for idx, batch in enumerate(tqdm(data_loader, total=n_batches, desc='Real Images')):
            if idx >= n_batches:
                break
            imgs = batch['org_imgs']
            lens = batch['org_img_lens']
            wids = batch.get('wids', torch.arange(imgs.size(0)))

            pil_imgs = batch_tensor_to_pil_list(imgs, lens)
            real_imgs_list.extend(pil_imgs)
            for i in range(imgs.size(0)):
                real_authors_list.append(str(wids[i].item()))
        real_dataset = ImageListDataset(real_imgs_list, real_authors_list)

    for idx, batch in enumerate(tqdm(generator, total=n_batches * n_rand_repeat, desc='Fake Images')):
        if idx >= n_batches * n_rand_repeat:
            break
        imgs = batch['org_imgs']
        lens = batch['org_img_lens']
        wids = batch.get('wids', torch.arange(imgs.size(0)))

        pil_imgs = batch_tensor_to_pil_list(imgs, lens)
        fake_imgs_list.extend(pil_imgs)
        for i in range(imgs.size(0)):
            fake_authors_list.append(str(wids[i].item()))

    fake_dataset = ImageListDataset(fake_imgs_list, fake_authors_list)

    print("Computing HWD Score...")
    hwd_scorer = HWDScore(batchsize=int(batchsize)).to(device)

    fake_pd = hwd_scorer.digest(fake_dataset)
    if real_features is None:
        real_pd = hwd_scorer.digest(real_dataset)
    else:
        real_pd = real_features

    score = hwd_scorer.distance(fake_pd, real_pd)
    return score
