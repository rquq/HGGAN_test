"""Streaming reader/writer for the seven-array concatenated IAM HDF5 layout.

Pixel polarity is preserved verbatim: uint8 grayscale, white background. No
appearance-based filtering, resizing, transcript normalization, or writer remap.
"""
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

_VECTORS = ('lbs', 'img_seek_idxs', 'lb_seek_idxs', 'img_lens', 'lb_lens', 'wids')


class IAMCorpus:
    def __init__(self, path):
        self.path = Path(path)
        # Retain enough decompressed chunks for a whole-height IAM window.
        # A persistent dataset handle keeps that bounded cache across rows.
        self._file = h5py.File(path, 'r', rdcc_nbytes=32 * 1024 * 1024, rdcc_nslots=10007)
        try:
            f = self._file
            if not {'imgs', *_VECTORS} <= set(f):
                raise ValueError('IAM storage is missing required arrays')
            self._images = f['imgs']
            if self._images.ndim != 2 or self._images.dtype != np.uint8 or self._images.shape[0] <= 0:
                raise ValueError('imgs must be positive-height uint8 H x total-width')
            for name in _VECTORS:
                if f[name].ndim != 1 or f[name].dtype.kind not in 'iu':
                    raise ValueError(f'{name} must be a one-dimensional integer array')
            self.height = self._images.shape[0]
            arrays = {n: f[n][:] for n in _VECTORS}
            self.widths, self.writers = arrays['img_lens'], arrays['wids']
            self._image_starts = arrays['img_seek_idxs']
            n = len(self.widths)
            if any(len(arrays[k]) != n for k in _VECTORS if k != 'lbs'):
                raise ValueError('IAM row array lengths disagree')
            for seeks, lens, total, positive in [
                (self._image_starts, self.widths, self._images.shape[1], True),
                (arrays['lb_seek_idxs'], arrays['lb_lens'], len(arrays['lbs']), True),
            ]:
                end = 0
                for seek, length in zip(seeks, lens):
                    seek, length = int(seek), int(length)
                    if seek != end or length < int(positive):
                        raise ValueError('invalid/noncontiguous IAM offsets or lengths')
                    end += length
                if end != total:
                    raise ValueError('IAM offsets do not cover stored arrays')
            labels = arrays['lbs']
            if np.any(labels > 0x10FFFF) or np.any(labels < 0) or np.any((labels >= 0xD800) & (labels <= 0xDFFF)):
                raise ValueError('labels contain invalid Unicode code points')
            self.texts = [''.join(chr(int(c)) for c in labels[int(s):int(s)+int(l)])
                          for s, l in zip(arrays['lb_seek_idxs'], arrays['lb_lens'])]
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.widths)

    def image(self, index):
        start, width = int(self._image_starts[index]), int(self.widths[index])
        return self._images[:, start:start + width]

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def validate_iam(path, expected_count=None):
    with IAMCorpus(path) as corpus:
        if expected_count is not None and len(corpus) != expected_count:
            raise ValueError(f'expected {expected_count} IAM rows, found {len(corpus)}')
        return {'count': len(corpus), 'height': corpus.height}


def write_iam(path, records, height, metadata=None):
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError('height must be a positive integer')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as f:
            f.create_dataset('imgs', shape=(height, 0), maxshape=(height, None), dtype=np.uint8)
            for key in _VECTORS:
                f.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.int64)
            if metadata is not None:
                f.attrs['metadata_json'] = json.dumps(metadata, sort_keys=True, allow_nan=False)
            image_offset = label_offset = 0
            for index, (image, text, writer) in enumerate(records):
                if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 2 or image.shape[0] != height or image.shape[1] <= 0:
                    raise ValueError('record image must be uint8 H x positive-width')
                if not isinstance(text, str) or not text or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
                    raise ValueError('record transcript must be nonempty valid Unicode')
                if isinstance(writer, (bool, np.bool_)) or not isinstance(writer, (int, np.integer)):
                    raise ValueError('writer must be an integer')
                width = image.shape[1]
                f['imgs'].resize(image_offset + width, axis=1)
                f['imgs'][:, image_offset:image_offset + width] = image
                f['lbs'].resize(label_offset + len(text), axis=0)
                f['lbs'][label_offset:label_offset + len(text)] = [ord(c) for c in text]
                values = dict(img_seek_idxs=image_offset, lb_seek_idxs=label_offset,
                              img_lens=width, lb_lens=len(text), wids=int(writer))
                for key, value in values.items():
                    f[key].resize(index + 1, axis=0)
                    f[key][index] = value
                image_offset += width
                label_offset += len(text)
            f.flush()
        validate_iam(temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
