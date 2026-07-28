import torch
import random


def apply_vertical_stripe_mask(imgs, img_lens, mask_ratio_range=(0.04, 0.10), stripe_width_range=(1, 3)):
    """
    Apply light vertical stripe masking (columns) to real images before style encoding.
    This helps disentangle content from horizontal style features.

    Args:
        imgs: [B, C, H, W] image tensor
        img_lens: [B] length tensor
        mask_ratio_range: tuple of (min, max) ratio of image width to mask (light masking)
        stripe_width_range: tuple of (min, max) stripe width in pixels (thin stripes)

    Returns:
        masked_imgs: [B, C, H, W] masked image tensor
    """
    B, C, H, W = imgs.shape
    masked_imgs = imgs.clone()

    for i in range(B):
        valid_len = int(img_lens[i].item())
        if valid_len <= 0:
            continue

        mask_ratio = random.uniform(mask_ratio_range[0], mask_ratio_range[1])
        num_pixels_to_mask = int(valid_len * mask_ratio)

        stripe_width = random.randint(stripe_width_range[0], stripe_width_range[1])

        num_stripes = max(1, num_pixels_to_mask // stripe_width)
        for _ in range(num_stripes):
            start_pos = random.randint(0, max(0, valid_len - stripe_width - 1))
            end_pos = min(start_pos + stripe_width, valid_len)
            # Mask value is -1 (background is white/blank in normalized space)
            masked_imgs[i, :, :, start_pos:end_pos] = -1

    return masked_imgs


def apply_horizontal_stripe_mask(imgs, img_lens, mask_ratio_range=(0.04, 0.10), stripe_height_range=(1, 3)):
    """
    Apply light horizontal stripe masking (rows) to real images before style encoding.
    This forces the model to learn vertical alignments and line differences (e.g. descenders in 'g' and 'y').

    Args:
        imgs: [B, C, H, W] image tensor
        img_lens: [B] length tensor
        mask_ratio_range: tuple of (min, max) ratio of image height to mask (light masking)
        stripe_height_range: tuple of (min, max) stripe height in pixels (thin stripes)

    Returns:
        masked_imgs: [B, C, H, W] masked image tensor
    """
    B, C, H, W = imgs.shape
    masked_imgs = imgs.clone()

    for i in range(B):
        mask_ratio = random.uniform(mask_ratio_range[0], mask_ratio_range[1])
        num_pixels_to_mask = int(H * mask_ratio)

        stripe_height = random.randint(stripe_height_range[0], stripe_height_range[1])

        num_stripes = max(1, num_pixels_to_mask // stripe_height)
        valid_len = int(img_lens[i].item())

        for _ in range(num_stripes):
            start_pos = random.randint(0, max(0, H - stripe_height - 1))
            end_pos = min(start_pos + stripe_height, H)
            masked_imgs[i, :, start_pos:end_pos, :valid_len] = -1

    return masked_imgs


def apply_combined_stripe_mask(imgs, img_lens, mask_ratio_range=(0.04, 0.10),
                                stripe_width_range=(1, 3), stripe_height_range=(1, 3)):
    """
    Apply both vertical and horizontal light stripe masking to real images.

    Args:
        imgs: [B, C, H, W] image tensor
        img_lens: [B] length tensor
        mask_ratio_range: tuple of (min, max) ratio to mask
        stripe_width_range: tuple of (min, max) vertical stripe width
        stripe_height_range: tuple of (min, max) horizontal stripe height

    Returns:
        masked_imgs: [B, C, H, W] masked image tensor
    """
    masked_imgs = apply_vertical_stripe_mask(imgs, img_lens, mask_ratio_range, stripe_width_range)
    masked_imgs = apply_horizontal_stripe_mask(masked_imgs, img_lens, mask_ratio_range, stripe_height_range)
    return masked_imgs


def apply_light_mixed_patch_mask(patches, mask_ratio_range=(0.02, 0.05), stripe_size_range=(1, 2)):
    """
    Apply extremely light mixed vertical and horizontal stripe masking to 32x32 local patches.
    Optimized to run entirely in PyTorch using vectorized tensor operations.

    Args:
        patches: [N, C, H, W] patch tensor (typically H=W=32)
        mask_ratio_range: tuple of (min, max) ratio of pixels to mask (extremely light)
        stripe_size_range: tuple of (min, max) size of stripe in pixels (thin stripes)

    Returns:
        masked_patches: [N, C, H, W] masked patch tensor
    """
    if patches.numel() == 0:
        return patches

    N, C, H, W = patches.shape
    device = patches.device

    # 1. Horizontal stripe (row masking) - helps model learn descenders/vertical placement
    apply_h = torch.rand((N, 1, 1, 1), device=device) < 0.7
    heights = torch.randint(stripe_size_range[0], stripe_size_range[1] + 1, (N, 1, 1, 1), device=device)
    starts_h = (torch.rand((N, 1, 1, 1), device=device) * (H - heights)).long()
    row_idx = torch.arange(H, device=device).view(1, 1, H, 1)
    h_mask = apply_h & (row_idx >= starts_h) & (row_idx < starts_h + heights)

    # 2. Vertical stripe (column masking) - helps model learn spacing/character connects
    apply_w = torch.rand((N, 1, 1, 1), device=device) < 0.7
    widths = torch.randint(stripe_size_range[0], stripe_size_range[1] + 1, (N, 1, 1, 1), device=device)
    starts_w = (torch.rand((N, 1, 1, 1), device=device) * (W - widths)).long()
    col_idx = torch.arange(W, device=device).view(1, 1, 1, W)
    w_mask = apply_w & (col_idx >= starts_w) & (col_idx < starts_w + widths)

    combined_mask = h_mask | w_mask
    masked_patches = torch.where(combined_mask, torch.tensor(-1.0, device=device, dtype=patches.dtype), patches)

    return masked_patches
