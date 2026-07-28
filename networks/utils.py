import functools
import numpy as np
from itertools import groupby
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
from torch.optim import lr_scheduler
from networks.block import AdaptiveInstanceNorm2d, Identity, AdaptiveInstanceLayerNorm2d, InstanceLayerNorm2d
from lib.alphabet import word_capitalize
from lib.path_config import ImgHeight, CharWidth


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        if (isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)
                or isinstance(m, nn.Embedding)):
            if init_type == 'N02':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type in ['glorot', 'xavier']:
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'ortho':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    if init_type in ['N02', 'glorot', 'xavier', 'kaiming', 'ortho']:
        net.apply(init_func)  # apply the initialization function <init_func>
    return net


def get_norm_layer(norm='in', **kwargs):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm == 'bn':
        norm_layer = functools.partial(nn.BatchNorm2d)
    elif norm == 'gn':
        norm_layer = functools.partial(nn.GroupNorm)
    elif norm == 'in':
        norm_layer = functools.partial(nn.InstanceNorm2d)
    elif norm == 'adain':
        norm_layer = functools.partial(AdaptiveInstanceNorm2d)
    elif norm == 'iln':
        norm_layer = functools.partial(InstanceLayerNorm2d)
    elif norm == 'adailn':
        norm_layer = functools.partial(AdaptiveInstanceLayerNorm2d)
    elif norm == 'none':
        def norm_layer(x): return Identity()
    else:
        raise ValueError("Unsupported normalization: {}".format(norm))
    return norm_layer


def frozen_bn(model):
    def fix_bn(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    model.apply(fix_bn)


def get_linear_scheduler(optimizer, start_decay_iter, n_iters_decay):
    def lambda_rule(iter):
        lr_l = 1.0 - max(0, iter - start_decay_iter) / float(n_iters_decay + 1)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def get_scheduler(optimizer, opt, last_epoch=-1, base_lr=None):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    """
    if base_lr is None:
        base_lr = getattr(opt, 'lr', None)
    for group in optimizer.param_groups:
        if 'initial_lr' not in group or base_lr is not None:
            group['initial_lr'] = base_lr if base_lr is not None else group.get('lr', 1e-4)

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - min(max(0, (epoch - opt.start_decay_epoch) / float(opt.n_epochs_decay + 1)), 0.999)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1, last_epoch=last_epoch)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0, last_epoch=last_epoch)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented' % opt.lr_policy)
    return scheduler


def restore_scheduler_state(scheduler, optimizer, state_dict, base_lr,
                            completed_epochs):
    """Restore epoch progress while rebasing a stale checkpoint learning rate."""
    if state_dict:
        scheduler.load_state_dict(state_dict)

    completed_epochs = max(0, int(completed_epochs))
    base_lrs = [float(base_lr)] * len(optimizer.param_groups)
    if hasattr(scheduler, 'lr_lambdas'):
        last_lrs = [
            lr * scheduler.lr_lambdas[index](completed_epochs)
            for index, lr in enumerate(base_lrs)
        ]
    else:
        old_bases = state_dict.get('base_lrs', base_lrs) if state_dict else base_lrs
        old_lrs = state_dict.get('_last_lr', old_bases) if state_dict else old_bases
        last_lrs = [
            new_base * (old_lr / old_base if old_base else 1.0)
            for new_base, old_lr, old_base in zip(base_lrs, old_lrs, old_bases)
        ]

    scheduler.base_lrs = base_lrs
    scheduler.last_epoch = completed_epochs
    scheduler._last_lr = last_lrs
    if hasattr(scheduler, '_step_count'):
        scheduler._step_count = completed_epochs + 1
    for param_group, initial_lr, lr in zip(optimizer.param_groups, base_lrs, last_lrs):
        param_group['initial_lr'] = initial_lr
        param_group['lr'] = lr
    return scheduler


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


def get_init_state(deepth, batch_size, hidden_dim, device, bidirectional=False):
    """Get cell states and hidden states."""
    if bidirectional:
        deepth *= 2
        hidden_dim //= 2

    h0_encoder_bi = torch.zeros(
        deepth,
        batch_size,
        hidden_dim, requires_grad=False)
    c0_encoder_bi = torch.zeros(
        deepth,
        batch_size,
        hidden_dim, requires_grad=False)
    return h0_encoder_bi.to(device), c0_encoder_bi.to(device)


def _info(model, detail=False, ret=False):
    nParams = sum([p.nelement() for p in model.parameters()])
    mSize = nParams * 4.0 / 1024 / 1024
    res = "*%-12s  param.: %dK  Stor.: %.4fMB" % (type(model).__name__,  nParams / 1000, mSize)
    if detail:
        res += '\r\n' + str(model)
    if ret:
        return res
    else:
        print(res)


def _info_simple(model, tag=None):
    nParams = sum([p.nelement() for p in model.parameters()])
    mSize = nParams * 4.0 / 1024 / 1024
    if tag is None:
        tag = type(model).__name__
    res = "%-12s P:%6dK  S:%8.4fMB" % (tag,  nParams / 1000, mSize)
    return res


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def idx_to_words(idx, lexicon, max_word_len=0, capitalize_ratio=0.5, blank_ratio=0., sort=True):
    words = []
    for i in idx:
        word = lexicon[i]
        if np.random.random() < capitalize_ratio:
            word = word_capitalize(word)

        if len(word) > max_word_len >= 1:
            pos = np.random.randint(0, len(word) - max_word_len)
            word = word[pos: pos + max_word_len]

        words.append(word)

    if sort:
        words.sort(key=lambda x: len(x), reverse=True)
    return words


def pil_text_img(im, text, pos, color=(255, 0, 0), textSize=25):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    try:
        font = ImageFont.truetype('font/arial.ttf', textSize)
    except OSError:
        font = ImageFont.load_default()
    fillColor = color  # (255,0,0)
    position = pos  # (100,100)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def words_to_images(texts, img_h, img_w, n_channel=1):
    word_imgs = np.zeros((len(texts), img_h, img_w, 3), dtype=np.uint8)
    for i in range(len(texts)):
        word_imgs[i] = pil_text_img(word_imgs[i], texts[i], (1, 1), textSize=25)
    word_imgs_sum = word_imgs.sum(axis=-1, keepdims=True).astype(np.uint8)
    word_imgs_t = torch.from_numpy(word_imgs_sum).permute([0, 3, 1, 2]).float() / 128.0 - 1.0
    return word_imgs_t


def ctc_greedy_decoder(probs_seq, blank_index=0):
    """CTC greedy (best path) decoder.
    Path consisting of the most probable tokens are further post-processed to
    remove consecutive repetitions and all blanks.
    :param probs_seq: 2-D list of probabilities over the vocabulary for each
                      character. Each element is a list of float probabilities
                      for one character.
    :type probs_seq: list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :return: Decoding result string.
    :rtype: baseline
    """

    # argmax to get the best index for each time step
    max_index_list = list(np.array(probs_seq).argmax(axis=1))
    # remove consecutive duplicate indexes
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # remove blank indexes
    index_list = [index for index in index_list if index != blank_index]
    # convert index list to string
    return index_list


class PatchSampler(object):
    def __init__(self, patch_size=(32, 32), sample_density=2, char_size=(64, 32)):
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.sample_density = sample_density
        self.char_size = char_size
        self.max_y = self.char_size[0] - self.patch_size[0]

    def random_sample(self, fake_imgs, img_lens, ret_xy=False):
        lb_lens = img_lens // self.char_size[1]
        patches = []
        pos_xy = []
        for bid in range(fake_imgs.size(0)):
            max_x = img_lens[bid] - self.patch_size[1]
            rand_top_xy = np.random.random((lb_lens[bid] * self.sample_density, 2))
            inc_x = np.linspace(0, max_x, lb_lens[bid] * self.sample_density)
            rand_y = rand_top_xy[:, 0] * self.max_y
            rand_x = rand_top_xy[:, 1] * self.char_size[1] // 4 + inc_x
            rand_x = rand_x.clip(0, max_x)
            rand_top_xy = np.stack([rand_x, rand_y]).transpose().astype('int')
            pos_xy.append(rand_top_xy)
            for tx, ty in rand_top_xy:
                patch = fake_imgs[bid, :, ty:ty+self.patch_size[0], tx:tx+self.patch_size[1]]
                patches.append(patch)

        if ret_xy:
            return patches, pos_xy
        else:
            return patches


def extract_patches_2d(img,patch_shape,step=[1.0,1.0],batch_first=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    return patches

def extract_all_patches(org_imgs, org_img_lens, block_size=32, step=8, plot=False):
    img_h = org_imgs.size(-2)
    n_patch_row = (img_h - block_size) // step + 1
    patches = extract_patches_2d(org_imgs, (block_size, block_size), step=[step, step], batch_first=True)
    patch_lens = torch.div(org_img_lens - block_size, step, rounding_mode='trunc') + 1
    mask = _len2mask(patch_lens, patches.size(1) // n_patch_row).repeat(1, n_patch_row).bool()
    patches = patches.masked_select(mask.view(*mask.size(), 1, 1, 1))
    patches = patches.view(-1, 1, block_size, block_size)
    if plot:
        idx = np.random.randint(1, org_imgs.size(0))

        import matplotlib.pyplot as plt
        from itertools import accumulate
        from lib.utils import draw_image

        plt.subplot(211)
        plt.imshow(org_imgs[idx, 0, :, :org_img_lens[idx].cpu().detach().numpy()].cpu().detach().numpy(), cmap='binary')
        # plt.axis('off')
        plt.subplot(212)
        sum_patch_lens = list(accumulate(patch_lens.cpu().detach().numpy() * n_patch_row))
        print(sum_patch_lens)
        patch_imgs = []
        for i in range(sum_patch_lens[idx - 1], sum_patch_lens[idx]):
            patch_imgs.append(patches[i])
        img = draw_image(1 - torch.stack(patch_imgs, dim=0).repeat(1, 3, 1, 1).cpu(), nrow=patch_lens[idx],
                         normalize=True)
        plt.imshow(img, cmap='binary')
        plt.axis('off')
        plt.show()
    return patches


def rand_clip_images(imgs, img_lens, min_clip_width=64):
    device = imgs.device
    min_clip_width = max(1, int(min_clip_width))
    step = max(1, min_clip_width // 4)
    clip_imgs = []
    clip_img_lens = []

    lens_list = img_lens.tolist() if isinstance(img_lens, torch.Tensor) else [int(l) for l in img_lens]

    for i, img_len in enumerate(lens_list):
        img_len = int(img_len)
        if img_len <= min_clip_width:
            clip_imgs.append(imgs[i, :, :, :img_len])
            clip_img_lens.append(img_len)
        else:
            crop_width = int(np.random.randint(min_clip_width, img_len))
            crop_width = crop_width - crop_width % step
            crop_width = min(img_len, max(min_clip_width, crop_width))

            max_pos = img_len - crop_width
            rand_pos = int(np.random.randint(0, max_pos)) if max_pos > 0 else 0
            clip_img = imgs[i, :, :, rand_pos : rand_pos + crop_width]
            clip_imgs.append(clip_img)
            clip_img_lens.append(clip_img.size(-1))

    max_img_len = max(clip_img_lens)
    pad_imgs = torch.full(
        (imgs.size(0), imgs.size(1), imgs.size(2), max_img_len),
        -1.0,
        dtype=imgs.dtype,
        device=device
    )
    for i, (clip_img, clip_len) in enumerate(zip(clip_imgs, clip_img_lens)):
        pad_imgs[i, :, :, :clip_len] = clip_img

    out_lens = torch.tensor(clip_img_lens, dtype=torch.int32, device=device)
    return pad_imgs, out_lens



def _recalc_len(leng, scale):
    tmp = leng % scale
    return leng + scale - tmp if tmp != 0 else leng


def augment_images(imgs, img_lens, lbs, lb_lens):
    bz, c, h, w = imgs.size()
    ref_img_lens = []
    for img_len in img_lens:
        ratio = (np.random.random() - 0.5) * 2 * 0.4
        new_width = int(img_len * (1 + ratio))
        ref_img_lens.append(_recalc_len(new_width, scale=CharWidth))

    target_idx = np.argsort(ref_img_lens)[::-1].copy()

    ref_img_lens = np.array(ref_img_lens, dtype=int)
    max_ref_len = _recalc_len(int(ref_img_lens.max()), scale=CharWidth)
    pad_imgs = -np.ones((bz, c, h, max_ref_len), dtype=np.float32)
    for i, (img, img_len, ref_img_len) in enumerate(zip(imgs.detach(), img_lens, ref_img_lens)):
        mode = 'area' if img_len > ref_img_len else 'bilinear'
        align_corners = None if img_len > ref_img_len else False
        resized_img = F.interpolate(img[:, :, :img_len].unsqueeze(dim=0),
                                    (h, int(ref_img_len)),
                                    mode=mode,
                                    align_corners=align_corners)
        org_img = resized_img[0, 0].cpu().numpy()
        pad_imgs[i, :, :, :ref_img_len] = org_img

    pad_imgs = np.stack([pad_imgs[idx] for idx in target_idx])
    ref_img_lens = np.stack([ref_img_lens[idx] for idx in target_idx])
    resized_imgs = torch.from_numpy(pad_imgs).float().to(imgs.device).detach()
    resized_img_lens = torch.from_numpy(ref_img_lens).int().to(imgs.device).detach()
    sort_lbs = lbs[target_idx]
    sort_lb_lens = lb_lens[target_idx]
    return resized_imgs, resized_img_lens, sort_lbs, sort_lb_lens


def rescale_images(imgs, img_lens, ref_img_lens):
    bz, c, h, w = imgs.size()
    max_ref = int(torch.as_tensor(ref_img_lens).max().item())
    pad_imgs = -np.ones((bz, c, h, _recalc_len(max_ref, h)))
    for i, (img, img_len, ref_img_len) in enumerate(zip(imgs, img_lens, ref_img_lens)):
        i_len = int(img_len)
        r_len = int(ref_img_len)
        mode = 'area' if i_len > r_len else 'bilinear'
        align_corners = None if i_len > r_len else False
        resized_img = F.interpolate(img[:, :, :i_len].unsqueeze(dim=0),
                                    (h, r_len),
                                    mode=mode,
                                    align_corners=align_corners)
        pad_imgs[i, :, :, :r_len] = resized_img[0].cpu().numpy()

    resized_imgs = torch.from_numpy(pad_imgs).float().to(imgs.device)
    return resized_imgs, ref_img_lens


def rescale_images2(imgs, img_lens, lb_lens, ref_img_lens, ref_lb_lens):
    target_img_lens = (ref_img_lens / ref_lb_lens) * lb_lens
    resized_imgs, target_img_lens = rescale_images(imgs, img_lens, target_img_lens.int())
    return resized_imgs, target_img_lens


def pad_image_lengths(img_lens, scale=ImgHeight):
    tmp = img_lens % scale
    return torch.where(tmp != 0, img_lens + scale - tmp, img_lens).detach()


def ensure_dim3(tensor):
    """Ensures input tensor is 3D (B, S, D) by unsqueezing 2D (B, D) tensors."""
    return tensor.unsqueeze(1) if tensor.dim() == 2 else tensor