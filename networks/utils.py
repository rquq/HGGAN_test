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
        min_lr_ratio = float(getattr(opt, 'min_lr_ratio', 0.001))
        start_decay_epoch = int(opt.start_decay_epoch)
        n_epochs_decay = int(opt.n_epochs_decay)
        if not 0.0 < min_lr_ratio <= 1.0:
            raise ValueError('min_lr_ratio must be in (0, 1]')
        if start_decay_epoch < 1:
            raise ValueError('start_decay_epoch must be at least 1')
        if n_epochs_decay < 1:
            raise ValueError('n_epochs_decay must be at least 1')

        def lambda_rule(scheduler_epoch):
            # LambdaLR uses zero-based scheduler epochs. Convert that index to
            # the one-based training epoch whose updates will use this LR. Thus
            # start_decay_epoch=24 means epochs 1-24 stay at the base LR and
            # epoch 25 is the first reduced-LR epoch.
            training_epoch = scheduler_epoch + 1
            progress = max(
                0.0,
                (training_epoch - start_decay_epoch)
                / float(n_epochs_decay),
            )
            return max(min_lr_ratio, 1.0 - progress)

        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch
        )
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


SPECIAL_CHARS = '0123456789\'-"/,.+_!#&():;?ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def _rare_lexicon_sampler(lexicon):
    """Return corpus words weighted toward genuinely rare alphabet symbols.

    Fabricated punctuation/digit strings can shift the training text
    distribution away from IAM. This sampler only reuses words in the configured
    corpus, while inverse-square-root character weights give Q/X/Z/J and other
    low-frequency glyphs more chances to reach the generator.
    """
    cache = getattr(idx_to_words, '_rare_cache', None)
    if cache is None:
        cache = {}
        idx_to_words._rare_cache = cache
    cache_key = (id(lexicon), len(lexicon))
    if cache_key in cache and cache[cache_key][0] is lexicon:
        return cache[cache_key][1]

    counts = {}
    for raw_word in lexicon:
        for char in str(raw_word).casefold():
            counts[char] = counts.get(char, 0) + 1
    if not counts:
        result = (np.arange(len(lexicon), dtype=np.int64), None)
        cache[cache_key] = (lexicon, result)
        return result

    inv_sqrt = {
        char: 1.0 / np.sqrt(float(count)) for char, count in counts.items()
    }
    scores = np.asarray([
        np.mean([inv_sqrt.get(char, 1.0) for char in str(word).casefold()])
        if str(word) else 0.0
        for word in lexicon
    ], dtype=np.float64)
    valid = np.isfinite(scores) & (scores > 0)
    if not np.any(valid):
        result = (np.arange(len(lexicon), dtype=np.int64), None)
    else:
        indices = np.flatnonzero(valid)
        weights = scores[indices]
        weights /= weights.sum()
        cumulative = np.cumsum(weights)
        cumulative /= cumulative[-1]
        result = (indices, cumulative)
    # Keep the source alive so Python cannot reuse its id for another lexicon.
    cache[cache_key] = (lexicon, result)
    return result


def idx_to_words(idx, lexicon, max_word_len=0, capitalize_ratio=0.5,
                 blank_ratio=0., sort=True, rare_ratio=0.15,
                 rare_lexicon=None):
    """Decode sampled lexicon IDs with a corpus-faithful rare-word policy.

    ``rare_ratio`` now controls oversampling of real rare-character words; it
    never fabricates arbitrary symbol sequences.  Set it to zero for fixed
    validation/sample text.
    """
    rare_source = lexicon if rare_lexicon is None else rare_lexicon
    rare_indices, rare_cumulative = (
        _rare_lexicon_sampler(rare_source) if rare_ratio > 0 else (None, None)
    )
    if isinstance(idx, torch.Tensor):
        # One device-to-host copy per batch is cheaper than synchronizing once
        # for every CUDA scalar during Python-side lexicon lookup.
        indices = idx.detach().cpu().tolist()
    else:
        indices = idx
    words = []
    for i in indices:
        base_index = int(i)
        word = str(lexicon[base_index])

        if rare_cumulative is not None and np.random.random() < rare_ratio:
            # Reuse the CDF instead of rebuilding it over the whole IAM corpus
            # for each sampled word (as np.random.choice(p=...) does).
            rare_index = rare_indices[np.searchsorted(
                rare_cumulative, np.random.random(), side='right'
            )]
            word = str(rare_source[int(rare_index)])

        # Capitalization is applied after rare-word selection so the sampled
        # glyph still follows the same case policy as ordinary corpus words.
        if np.random.random() < capitalize_ratio:
            word = word.capitalize() if np.random.random() < 0.8 else word.upper()

        if len(word) > max_word_len >= 1:
            pos = np.random.randint(0, len(word) - max_word_len + 1)
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


def adaptive_crop_count(valid_width, patch_size=32, min_crops=4, max_crops=8):
    """Choose bounded local coverage proportional to the valid word width."""
    if patch_size < 1 or min_crops < 1 or max_crops < min_crops:
        raise ValueError('invalid adaptive crop configuration')
    width_crops = (max(int(valid_width), 1) + patch_size - 1) // patch_size
    return max(min_crops, min(max_crops, width_crops))


def sample_character_patches(
    images,
    image_lens,
    labels,
    label_lens,
    patch_size=32,
    min_crops=4,
    max_crops=8,
    horizontal_jitter=4,
    fill_value=-1.0,
    return_confidence=False,
):
    """Sample character-aligned stroke crops and return their character IDs.

    Each crop is centered on a character-width stratum, while alternate crops
    cover the upper and lower writing bands.  This preserves the old bounded
    4--8 crop budget but gives StrokePatchD an explicit allograph label.
    """
    if images.ndim != 4:
        raise ValueError('images must have shape (B, C, H, W)')
    if labels.ndim != 2:
        raise ValueError('labels must have shape (B, L)')
    if images.size(0) != len(image_lens) or images.size(0) != labels.size(0):
        raise ValueError('images, image_lens, and labels must share a batch size')
    if images.size(0) != len(label_lens):
        raise ValueError('label_lens must contain one length per image')
    if patch_size < 1 or min_crops < 1 or max_crops < min_crops:
        raise ValueError('invalid adaptive crop configuration')
    if horizontal_jitter < 0:
        raise ValueError('horizontal_jitter must be non-negative')

    pad_bottom = max(0, patch_size - images.size(-2))
    pad_right = max(0, patch_size - images.size(-1))
    if pad_bottom or pad_right:
        images = F.pad(
            images, (0, pad_right, 0, pad_bottom), value=float(fill_value)
        )

    image_height, image_width = images.shape[-2:]
    device = images.device
    batch_size = images.size(0)
    if batch_size == 0 or labels.size(1) == 0:
        raise ValueError('character crops require a nonempty batch and label dimension')
    # Length tensors may already have this device/dtype: avoid mutating the
    # caller's lengths, which are also consumed by OCR and reconstruction.
    widths = image_lens.to(device=device, dtype=torch.long).clamp(1, image_width)
    lengths = label_lens.to(device=device, dtype=torch.long).clamp(1, labels.size(1))
    crop_counts = ((widths + patch_size - 1) // patch_size).clamp_(min_crops, max_crops)
    crop_slots = torch.arange(max_crops, device=device)
    row_indices, crop_indices = (
        crop_slots[None, :] < crop_counts[:, None]
    ).nonzero(as_tuple=True)
    total_crops = row_indices.numel()
    valid_widths = widths[row_indices]
    valid_lengths = lengths[row_indices]

    char_start = torch.div(
        crop_indices * valid_lengths, crop_counts[row_indices], rounding_mode='floor'
    )
    char_start = torch.minimum(char_start, valid_lengths - 1)
    char_end = torch.div(
        (crop_indices + 1) * valid_lengths,
        crop_counts[row_indices], rounding_mode='floor'
    )
    char_end = torch.maximum(char_end, char_start + 1)
    char_end = torch.minimum(char_end, valid_lengths)
    char_index = char_start + torch.floor(
        torch.rand(total_crops, device=device) * (char_end - char_start).to(torch.float32)
    ).to(torch.long)

    span_start = char_index.to(torch.float32) * valid_widths.to(torch.float32) / valid_lengths
    span_end = (char_index + 1).to(torch.float32) * valid_widths.to(torch.float32) / valid_lengths
    left = torch.round((span_start + span_end - float(patch_size)) / 2.0).to(torch.long)
    if horizontal_jitter:
        left += torch.randint(
            -horizontal_jitter, horizontal_jitter + 1, (total_crops,), device=device
        )
    left = left.clamp_(min=0)
    left = torch.minimum(left, (valid_widths - patch_size).clamp_min(0))

    max_top = max(image_height - patch_size, 0)
    vertical_positions = max(max_top + 1, 1)
    vertical_slot = crop_indices.remainder(2)
    top_start = vertical_slot * (vertical_positions // 2)
    top_end = torch.maximum(
        (vertical_slot + 1) * vertical_positions // 2,
        top_start + 1,
    )
    top = top_start + torch.floor(
        torch.rand(total_crops, device=device) * (top_end - top_start).to(torch.float32)
    ).to(torch.long)
    top = top.clamp_(max=max_top)

    # Gather only the selected pixels. Indexing an unfold view makes backward
    # allocate gradients for *every* sliding window, even for just a few crops.
    offset = torch.arange(patch_size, device=device)
    patches = images[
        row_indices[:, None, None, None],
        torch.arange(images.size(1), device=device)[None, :, None, None],
        (top[:, None] + offset)[:, None, :, None],
        (left[:, None] + offset)[:, None, None, :],
    ]

    # The crop label is approximate because IAM stores word boxes, not per-glyph
    # boxes.  Pass a soft confidence to StrokePatchD: partial/blank crops still
    # train its unconditional stroke critic but cannot inject a wrong class code.
    overlap = (
        torch.minimum(span_end, left.to(torch.float32) + patch_size)
        - torch.maximum(span_start, left.to(torch.float32))
    ).clamp_min(0.0)
    span_capacity = (span_end - span_start).clamp_min(1.0).clamp_max(float(patch_size))
    geometry_confidence = (overlap / span_capacity).clamp(0.0, 1.0)
    ink_fraction = (patches > -0.75).to(torch.float32).mean(dim=(1, 2, 3))
    ink_confidence = ((ink_fraction - 0.005) / 0.04).clamp(0.0, 1.0)
    patch_confidence = (geometry_confidence * ink_confidence).to(patches.dtype)

    labels_device = labels.device
    row_for_labels = row_indices.to(labels_device)
    char_for_labels = char_index.to(labels_device)
    character_ids = labels[row_for_labels, char_for_labels].long().to(device)
    result = (
        patches,
        crop_counts.to(device=device),
        character_ids,
    )
    if return_confidence:
        result = result + (patch_confidence,)
    return result


def augment_word_batch(
    images,
    image_lens,
    max_translation=4,
    width_scale=0.05,
    fill_value=-1.0,
):
    """Apply mild differentiable word-safe geometry to a D input batch.

    The canvas and valid widths do not change.  Only horizontal scale and
    translation are used so text content, baseline, and label lengths remain
    valid.  Applying this same policy family to real and generated words avoids
    teaching D an augmentation shortcut.
    """
    if images.ndim != 4:
        raise ValueError('images must have shape (B, C, H, W)')
    if images.size(0) != len(image_lens):
        raise ValueError('image_lens must contain one length per image')
    if max_translation < 0 or not 0.0 <= width_scale < 1.0:
        raise ValueError('invalid discriminator augmentation configuration')
    if max_translation == 0 and width_scale == 0:
        return images, image_lens

    output = torch.full_like(images, float(fill_value))
    lengths = image_lens.detach().cpu().long().tolist()
    canvas_width = images.size(-1)
    for row, raw_width in enumerate(lengths):
        valid_width = max(1, min(int(raw_width), canvas_width))
        word = images[row:row + 1, :, :, :valid_width]
        if width_scale:
            scale = 1.0 + (2.0 * torch.rand(()).item() - 1.0) * width_scale
            scaled_width = max(1, int(round(valid_width * scale)))
            word = F.interpolate(
                word, size=(images.size(-2), scaled_width),
                mode='bilinear', align_corners=False,
            )
        else:
            scaled_width = valid_width

        if scaled_width >= valid_width:
            excess = scaled_width - valid_width
            crop_left = int(torch.randint(0, excess + 1, (1,)).item()) if excess else 0
            word = word[..., crop_left:crop_left + valid_width]
        else:
            pad_total = valid_width - scaled_width
            pad_left = int(torch.randint(0, pad_total + 1, (1,)).item())
            word = F.pad(
                word, (pad_left, pad_total - pad_left, 0, 0),
                value=float(fill_value),
            )

        shift = int(torch.randint(
            -max_translation, max_translation + 1, (1,)
        ).item()) if max_translation else 0
        shift = max(-(valid_width - 1), min(shift, valid_width - 1))
        if shift > 0:
            word = F.pad(
                word[..., :valid_width - shift], (shift, 0, 0, 0),
                value=float(fill_value),
            )
        elif shift < 0:
            amount = -shift
            word = F.pad(
                word[..., amount:], (0, amount, 0, 0),
                value=float(fill_value),
            )
        output[row:row + 1, :, :, :valid_width] = word

    return output, image_lens


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
    pad_width = _recalc_len(max_ref, h)
    pad_imgs = torch.full((bz, c, h, pad_width), -1.0, dtype=imgs.dtype, device=imgs.device)
    for i, (img, img_len, ref_img_len) in enumerate(zip(imgs, img_lens, ref_img_lens)):
        i_len = int(img_len)
        r_len = int(ref_img_len)
        mode = 'area' if i_len > r_len else 'bilinear'
        align_corners = None if i_len > r_len else False
        resized_img = F.interpolate(img[:, :, :i_len].unsqueeze(dim=0),
                                    (h, r_len),
                                    mode=mode,
                                    align_corners=align_corners)
        pad_imgs[i, :, :, :r_len] = resized_img[0]

    return pad_imgs, ref_img_lens


def rescale_images2(imgs, img_lens, lb_lens, ref_img_lens, ref_lb_lens):
    target_img_lens = (ref_img_lens / ref_lb_lens) * lb_lens
    resized_imgs, target_img_lens = rescale_images(imgs, img_lens, target_img_lens.int())
    return resized_imgs, target_img_lens


def pad_image_lengths(img_lens, scale=None):
    if scale is None:
        import lib.path_config as path_cfg
        scale = path_cfg.ImgHeight
    tmp = img_lens % scale
    return torch.where(tmp != 0, img_lens + scale - tmp, img_lens).detach()


def ensure_dim3(tensor):
    """Ensures input tensor is 3D (B, S, D) by unsqueezing 2D (B, D) tensors."""
    return tensor.unsqueeze(1) if tensor.dim() == 2 else tensor
