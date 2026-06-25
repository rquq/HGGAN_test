import torch


def apply_vertical_stripe_mask(imgs, img_lens, mask_ratio_range=(0.04, 0.10), stripe_width_range=(1, 3)):
    """
    Vertical stripe masking is scrapped to focus entirely on horizontal/line-level learning.
    Returns the original image tensor unchanged.
    """
    return imgs


def apply_horizontal_stripe_mask(imgs, img_lens, mask_ratio_range=(0.04, 0.10), stripe_height_range=(2, 5)):
    """
    Apply horizontal stripe masking (rows) to real images before style encoding.
    Improved version targeting key whole-image alignment lines (baseline and mean-line)
    to improve line-level learning.

    Args:
        imgs: [B, C, H, W] image tensor (typically H=64)
        img_lens: [B] length tensor
        mask_ratio_range: tuple of (min, max) ratio of image height to mask (unused in new targeted logic)
        stripe_height_range: tuple of (min, max) stripe height in pixels (thin stripes)

    Returns:
        masked_imgs: [B, C, H, W] masked image tensor
    """
    B, C, H, W = imgs.shape
    masked_imgs = imgs.clone()

    for i in range(B):
        valid_len = int(img_lens[i].item())
        if valid_len <= 0:
            continue

        # 1. Primary Targeted Alignment Mask (either baseline at y=40 or mean-line at y=24)
        stripe_height = torch.randint(stripe_height_range[0], stripe_height_range[1] + 1, (1,)).item()
        target_line = 40 if torch.rand(1).item() < 0.5 else 24
        
        # Add small random jitter to the target line center
        jitter = torch.randint(-2, 3, (1,)).item()
        start_pos = int(target_line - stripe_height // 2 + jitter)
        start_pos = max(0, min(start_pos, H - stripe_height))
        end_pos = min(start_pos + stripe_height, H)
        masked_imgs[i, :, start_pos:end_pos, :valid_len] = -1

        # 2. Auxiliary Thin Detail Masks (1-2 thin random stripes)
        num_aux = torch.randint(1, 3, (1,)).item()
        for _ in range(num_aux):
            aux_h = torch.randint(1, 3, (1,)).item()
            aux_start = torch.randint(0, H - aux_h, (1,)).item()
            aux_end = aux_start + aux_h
            masked_imgs[i, :, aux_start:aux_end, :valid_len] = -1

    return masked_imgs


def apply_combined_stripe_mask(imgs, img_lens, mask_ratio_range=(0.04, 0.10),
                                stripe_width_range=(1, 3), stripe_height_range=(2, 5)):
    """
    Combined stripe mask returns only horizontal stripe mask, as vertical is scrapped.
    """
    return apply_horizontal_stripe_mask(imgs, img_lens, mask_ratio_range, stripe_height_range)




