import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from lib.path_config import ImgHeight, CharWidth

# Pillow >=9.1 moved resampling constants to Image.Resampling;
# older versions expose them directly on Image.
_Resampling = getattr(Image, 'Resampling', Image)


class RandomClip:
    """Randomly crops an image along its width if width exceeds min_clip_width."""
    def __init__(self, min_clip_width=None, align_scale=None):
        import lib.path_config as path_cfg
        self.min_clip_width = max(1, int(min_clip_width if min_clip_width is not None else path_cfg.ImgHeight * 2))
        self.align_scale = max(1, int(align_scale if align_scale is not None else path_cfg.CharWidth))

    def _recalc_len(self, leng, scale=None):
        if scale is None:
            import lib.path_config as path_cfg
            scale = path_cfg.CharWidth
        scale = max(1, int(scale))
        tmp = leng % scale
        return leng - tmp if tmp != 0 else leng

    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            return pic
        width, height = pic.size[0], pic.size[1]
        if width > self.min_clip_width:
            crop_width = int(np.random.randint(self.min_clip_width, width))
            if self.align_scale is not None:
                crop_width = self._recalc_len(crop_width, scale=self.align_scale)
                crop_width = max(self.min_clip_width, crop_width)
            crop_width = min(width, crop_width)
            max_pos = width - crop_width
            if max_pos > 0 and self.align_scale is not None:
                max_unit = max_pos // self.align_scale
                rand_pos = int(np.random.randint(0, max_unit + 1)) * self.align_scale
            else:
                rand_pos = int(np.random.randint(0, max_pos + 1)) if max_pos > 0 else 0
            pic = pic.crop((rand_pos, 0, rand_pos + crop_width, height))
        return pic

    def __repr__(self):
        return f"{self.__class__.__name__}(min_clip_width={self.min_clip_width}, align_scale={self.align_scale})"


class RandomScale:
    """Randomly rescales image width within a specified variance."""
    def __init__(self, var=0.4, scale=None):
        self.var = float(var)
        self.scale = scale

    def _recalc_len(self, leng, scale=None):
        if scale is None:
            scale = self.scale
        if scale is None:
            import lib.path_config as path_cfg
            scale = path_cfg.CharWidth
        scale = max(1, int(scale))
        tmp = leng % scale
        return leng + scale - tmp if tmp != 0 else leng

    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            return pic
        width, height = pic.size[0], pic.size[1]
        ratio = (np.random.random() - 0.5) * 2 * self.var
        new_width = max(1, int(width * (1 + ratio)))
        new_width = max(1, self._recalc_len(new_width))
        if ratio > 0:
            pic = pic.resize((new_width, height), _Resampling.BILINEAR)
        else:
            pic = pic.resize((new_width, height), _Resampling.LANCZOS)
        return pic

    def __repr__(self):
        return f"{self.__class__.__name__}(var={self.var})"


class MildHandwritingAugment:
    """Small, label-preserving perturbations for recognizer pretraining.

    This deliberately avoids crops, rotations, elastic warps, and strong noise:
    those can alter character identity or writer traits.  Width scaling remains
    separate and aligned to CharWidth in ``RandomScale``.
    """
    def __init__(self, affine_probability=0.55, tone_probability=0.65,
                 blur_probability=0.12, noise_probability=0.12,
                 max_translate_x=3, max_translate_y=1, max_shear=0.06,
                 contrast_range=(0.88, 1.12), brightness_range=(0.94, 1.06),
                 max_blur_radius=0.45, noise_std=2.0):
        self.affine_probability = float(affine_probability)
        self.tone_probability = float(tone_probability)
        self.blur_probability = float(blur_probability)
        self.noise_probability = float(noise_probability)
        self.max_translate_x = int(max_translate_x)
        self.max_translate_y = int(max_translate_y)
        self.max_shear = float(max_shear)
        self.contrast_range = tuple(float(value) for value in contrast_range)
        self.brightness_range = tuple(float(value) for value in brightness_range)
        self.max_blur_radius = float(max_blur_radius)
        self.noise_std = float(noise_std)

    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            return pic
        if np.random.random() < self.affine_probability:
            shift_x = float(np.random.uniform(-self.max_translate_x, self.max_translate_x))
            shift_y = float(np.random.uniform(-self.max_translate_y, self.max_translate_y))
            shear = float(np.random.uniform(-self.max_shear, self.max_shear))
            # PIL maps output coordinates back to source coordinates, hence -shift.
            affine = (1.0, shear, -shift_x, 0.0, 1.0, -shift_y)
            try:
                pic = pic.transform(
                    pic.size, Image.Transform.AFFINE, affine,
                    resample=_Resampling.BILINEAR, fillcolor=255,
                )
            except AttributeError:  # Pillow < 9.1 compatibility
                pic = pic.transform(
                    pic.size, Image.AFFINE, affine,
                    resample=_Resampling.BILINEAR, fillcolor=255,
                )
        if np.random.random() < self.tone_probability:
            pic = ImageEnhance.Contrast(pic).enhance(
                float(np.random.uniform(*self.contrast_range))
            )
            pic = ImageEnhance.Brightness(pic).enhance(
                float(np.random.uniform(*self.brightness_range))
            )
        if np.random.random() < self.blur_probability:
            pic = pic.filter(ImageFilter.GaussianBlur(
                radius=float(np.random.uniform(0.05, self.max_blur_radius))
            ))
        if np.random.random() < self.noise_probability:
            array = np.asarray(pic, dtype=np.float32)
            array += np.random.normal(0.0, self.noise_std, size=array.shape)
            pic = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode='L')
        return pic

    def __repr__(self):
        return (f"{self.__class__.__name__}(affine_probability={self.affine_probability}, "
                f"tone_probability={self.tone_probability}, "
                f"blur_probability={self.blur_probability}, "
                f"noise_probability={self.noise_probability})")
