"""Random crop helpers for the Recognizer (R) auxiliary loss.

Motivation
----------
HiGAN+ feeds every fake image (random / style / recn) through the OCR Recognizer
``R`` and adds a CTC loss against the full word.  ``R`` is a strong, frozen
teacher: forcing every fake to be readable end-to-end on its own makes ``G``
collapse toward "neat, OCR-friendly" handwriting and erodes the style signal
coming from ``E``.

Idea 1 (this module): crop the fake image *before* it reaches ``R`` and only
ask ``R`` to read the cropped slice.  Three deterministic crop strategies are
provided so we can A/B test how much weakening of ``R`` is needed:

* ``left_half`` -- always keep the leftmost 50% of the image and the
  corresponding 50% prefix of the label.
* ``left_three_quarter`` -- keep the leftmost 75%.
* ``char_aligned`` -- per-sample uniformly random ``[i, j]`` slice of
  characters and map it back to pixels via the fixed ``char_width``.

The crop is applied with probability ``recog_crop.prob`` per training step.
When skipped, the original full image flows through ``R`` exactly as in the
baseline.

Geometry assumptions
--------------------
HiGAN+'s generator emits images of width ``lb_len * char_width`` (default
``char_width = 32``).  The recognizer downsamples width by ``len_scale = 16``
(see ``Recognizer.len_scale``).  So a single character occupies ``char_width``
pixels and ``char_width / len_scale = 2`` CTC time-steps.  Cropping at
character boundaries therefore keeps the width-to-CTC-length relationship the
caller expects when computing ``input_lengths = img_len // len_scale``.

Background pad value is ``-1`` (matching ``nn.ConstantPad2d(2, -1)`` inside
``Recognizer`` and the ``-1`` background convention used elsewhere in the
repo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


VALID_MODES = ("left_half", "left_three_quarter", "char_aligned")
PAD_VALUE = -1.0  # background convention used across HiGAN+ tensors


@dataclass
class CropConfig:
    """Resolved crop configuration.

    ``mode`` selects the crop strategy.  ``prob`` is the per-iteration
    probability of applying the crop at all; when not applied the caller
    should fall back to the full-image CTC pathway.  ``min_chars`` guards
    against degenerate crops (e.g. an empty slice or a slice shorter than
    one character) by re-routing those samples through the original tensors.
    """

    mode: str
    prob: float = 0.5
    min_chars: int = 1
    char_width: int = 32

    @classmethod
    def from_opt(cls, opt_section, char_width: int) -> Optional["CropConfig"]:
        """Build a :class:`CropConfig` from a parsed YAML section.

        Returns ``None`` when crop is disabled or the section is missing,
        which makes it cheap for callers to short-circuit.
        """

        if opt_section is None:
            return None
        # ``munch.Munch`` supports both dict access and attribute access.
        enabled = getattr(opt_section, "enabled", True)
        if not enabled:
            return None
        mode = getattr(opt_section, "mode", None)
        if mode is None:
            return None
        if mode not in VALID_MODES:
            raise ValueError(
                f"recog_crop.mode={mode!r} not in {VALID_MODES}"
            )
        prob = float(getattr(opt_section, "prob", 0.5))
        prob = max(0.0, min(1.0, prob))
        min_chars = int(getattr(opt_section, "min_chars", 1))
        return cls(mode=mode, prob=prob, min_chars=min_chars, char_width=char_width)


def _slice_for_sample(
    mode: str,
    lb_len: int,
    char_width: int,
    min_chars: int,
    generator: torch.Generator,
) -> tuple[int, int]:
    """Return character-index slice ``(i, j)`` to keep, with ``j > i``.

    All three modes operate at character granularity so the resulting pixel
    crop is always ``(j - i) * char_width`` wide and label substring ``[i:j]``
    has length ``j - i``.
    """

    lb_len = int(lb_len)
    if lb_len <= 0:
        return 0, 0
    min_chars = max(1, min(min_chars, lb_len))

    if mode == "left_half":
        # ceil so a 1-char word is kept whole and a 3-char word keeps 2 chars.
        keep = max(min_chars, (lb_len + 1) // 2)
        return 0, min(keep, lb_len)

    if mode == "left_three_quarter":
        keep = max(min_chars, (3 * lb_len + 3) // 4)
        return 0, min(keep, lb_len)

    if mode == "char_aligned":
        # Uniformly sample contiguous slice [i, j) with length >= min_chars.
        # Using torch.randint with the supplied generator keeps the crop
        # deterministic w.r.t. the trainer's seed when desired.
        max_start = lb_len - min_chars
        i = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
        max_len = lb_len - i
        # Length sampled in [min_chars, max_len].
        slice_len = int(
            torch.randint(min_chars, max_len + 1, (1,), generator=generator).item()
        )
        return i, i + slice_len

    raise ValueError(f"unknown crop mode {mode!r}")


def crop_for_recognizer(
    imgs: torch.Tensor,
    lbs: torch.Tensor,
    lb_lens: torch.Tensor,
    cfg: CropConfig,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop a fake-image batch and align the label tensor for CTC.

    Parameters
    ----------
    imgs : ``(B, C, H, W)`` tensor produced by ``G``.  ``W`` is assumed to be
        ``max(lb_lens) * char_width`` (HiGAN+'s collate already pads to that).
    lbs : ``(B, L)`` long tensor of label indices.
    lb_lens : ``(B,)`` long tensor with the true label length per sample.
    cfg : resolved :class:`CropConfig`.
    generator : optional torch RNG.  When ``None`` the global RNG is used.

    Returns
    -------
    cropped_imgs : ``(B, C, H, W')`` tensor padded to the new max width with
        ``PAD_VALUE``.
    cropped_img_lens : ``(B,)`` long tensor with valid widths after crop.
    cropped_lbs : ``(B, L')`` long tensor with the matching label substrings.
    cropped_lb_lens : ``(B,)`` long tensor.
    """

    if cfg is None:
        raise ValueError("crop_for_recognizer called with cfg=None")

    device = imgs.device
    batch_size = imgs.size(0)
    char_width = cfg.char_width
    img_height = imgs.size(2)
    channels = imgs.size(1)
    # ``strLabelConverter.encode`` pads label tensors with 0 (CTC blank),
    # so we mirror that convention for slots past ``cropped_lb_lens``.
    lb_lens_cpu = lb_lens.detach().cpu().tolist()

    new_img_lens = []
    new_lb_lens = []
    starts_chars = []
    ends_chars = []
    for b in range(batch_size):
        lb_len = lb_lens_cpu[b]
        i, j = _slice_for_sample(cfg.mode, lb_len, char_width, cfg.min_chars, generator)
        # Defensive: keep at least one char so CTC has something to score.
        if j - i < 1:
            i, j = 0, max(1, lb_len)
        starts_chars.append(i)
        ends_chars.append(j)
        new_lb_lens.append(j - i)
        new_img_lens.append((j - i) * char_width)

    max_new_w = max(new_img_lens) if new_img_lens else char_width
    max_new_lb_len = max(new_lb_lens) if new_lb_lens else 1

    cropped_imgs = imgs.new_full(
        (batch_size, channels, img_height, max_new_w), PAD_VALUE
    )
    cropped_lbs = lbs.new_zeros((batch_size, max_new_lb_len))

    for b in range(batch_size):
        i, j = starts_chars[b], ends_chars[b]
        x0 = i * char_width
        x1 = j * char_width
        slice_w = x1 - x0
        # Width on the input might be smaller than ``j*char_width`` if the
        # collate clipped extra padding; clamp defensively.
        x1 = min(x1, imgs.size(-1))
        slice_w = max(0, x1 - x0)
        if slice_w > 0:
            cropped_imgs[b, :, :, :slice_w] = imgs[b, :, :, x0:x1]
        slice_lb_len = j - i
        if slice_lb_len > 0:
            cropped_lbs[b, :slice_lb_len] = lbs[b, i:j]

    cropped_img_lens = torch.tensor(new_img_lens, dtype=lb_lens.dtype, device=device)
    cropped_lb_lens = torch.tensor(new_lb_lens, dtype=lb_lens.dtype, device=device)

    # The repo's BLSTM in ``Recognizer`` uses ``pack_padded_sequence`` with
    # ``enforce_sorted=True``, so its input must be sorted by length DESC.
    # The dataset's collate already sorts the original batch, but random
    # char-aligned crops can break that order.  Re-sort here to keep the
    # contract intact.  The trainer only uses these tensors to compute
    # CTC loss (a batch-mean scalar), so the permutation is invisible
    # outside this helper.
    sorted_lens, perm = torch.sort(cropped_img_lens, descending=True)
    cropped_imgs = cropped_imgs.index_select(0, perm)
    cropped_lbs = cropped_lbs.index_select(0, perm)
    cropped_img_lens = sorted_lens
    cropped_lb_lens = cropped_lb_lens.index_select(0, perm)

    return cropped_imgs, cropped_img_lens, cropped_lbs, cropped_lb_lens


def should_apply(cfg: Optional[CropConfig], generator: Optional[torch.Generator] = None) -> bool:
    """Bernoulli draw on whether to apply the configured crop this iter."""

    if cfg is None or cfg.prob <= 0.0:
        return False
    if cfg.prob >= 1.0:
        return True
    return float(torch.rand((), generator=generator).item()) < cfg.prob
