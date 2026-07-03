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
    This is used to refine local patches in the Patch Discriminator pipeline, helping the model
    learn line-level and stroke-level alignment details (like 'g' and 'y' descenders).

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
    masked_patches = patches.clone()

    for i in range(N):
        # 1. Horizontal stripe (row masking) - helps model learn descenders/vertical placement
        if random.random() < 0.7:  # 70% chance to apply
            mask_ratio_h = random.uniform(mask_ratio_range[0], mask_ratio_range[1])
            num_pixels_h = int(H * mask_ratio_h)
            stripe_h = random.randint(stripe_size_range[0], stripe_size_range[1])
            num_stripes_h = max(1, num_pixels_h // stripe_h)
            for _ in range(num_stripes_h):
                start_h = random.randint(0, max(0, H - stripe_h - 1))
                end_h = min(start_h + stripe_h, H)
                # Background in normalized image space is represented by -1
                masked_patches[i, :, start_h:end_h, :] = -1

        # 2. Vertical stripe (column masking) - helps model learn spacing/character connects
        if random.random() < 0.7:  # 70% chance to apply
            mask_ratio_w = random.uniform(mask_ratio_range[0], mask_ratio_range[1])
            num_pixels_w = int(W * mask_ratio_w)
            stripe_w = random.randint(stripe_size_range[0], stripe_size_range[1])
            num_stripes_w = max(1, num_pixels_w // stripe_w)
            for _ in range(num_stripes_w):
                start_w = random.randint(0, max(0, W - stripe_w - 1))
                end_w = min(start_w + stripe_w, W)
                masked_patches[i, :, :, start_w:end_w] = -1

    return masked_patches
