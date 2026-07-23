import numpy as np
from PIL import Image
from lib.path_config import ImgHeight, CharWidth

# Pillow >=9.1 moved resampling constants to Image.Resampling;
# older versions expose them directly on Image.
_Resampling = getattr(Image, 'Resampling', Image)


class RandomClip:
    """Randomly crops an image along its width if width exceeds min_clip_width."""
    def __init__(self, min_clip_width: int = ImgHeight * 2, align_scale: int = CharWidth):
        self.min_clip_width = max(1, int(min_clip_width))
        self.align_scale = max(1, int(align_scale)) if align_scale is not None else None

    def _recalc_len(self, leng: int, scale: int = CharWidth) -> int:
        scale = max(1, int(scale))
        tmp = leng % scale
        return leng - tmp if tmp != 0 else leng

    def __call__(self, pic: Image.Image) -> Image.Image:
        if not isinstance(pic, Image.Image):
            return pic
        width, height = pic.size[0], pic.size[1]
        if width > self.min_clip_width:
            crop_width = int(np.random.randint(self.min_clip_width, width))
            if self.align_scale is not None:
                crop_width = self._recalc_len(crop_width, scale=self.align_scale)
                crop_width = max(self.min_clip_width, crop_width)
            max_pos = width - crop_width
            rand_pos = int(np.random.randint(0, max_pos)) if max_pos > 0 else 0
            pic = pic.crop((rand_pos, 0, rand_pos + crop_width, height))
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_clip_width={self.min_clip_width}, align_scale={self.align_scale})"


class RandomScale:
    """Randomly rescales image width within a specified variance."""
    def __init__(self, var: float = 0.4):
        self.var = float(var)

    def _recalc_len(self, leng: int, scale: int = CharWidth) -> int:
        scale = max(1, int(scale))
        tmp = leng % scale
        return leng + scale - tmp if tmp != 0 else leng

    def __call__(self, pic: Image.Image) -> Image.Image:
        if not isinstance(pic, Image.Image):
            return pic
        width, height = pic.size[0], pic.size[1]
        ratio = (np.random.random() - 0.5) * 2 * self.var
        new_width = max(1, int(width * (1 + ratio)))
        new_width = max(1, self._recalc_len(new_width, scale=CharWidth))
        if ratio > 0:
            pic = pic.resize((new_width, height), _Resampling.BILINEAR)
        else:
            pic = pic.resize((new_width, height), _Resampling.LANCZOS)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(var={self.var})"