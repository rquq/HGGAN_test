"""Shared pure OCR payload builder; never creates jobs or loads models."""
from pathlib import Path

from .common import read_json, sha256_file
from .config import ocr_settings


def ocr_payload(cfg, arm, seed, prepared, synthetic_metadata=None):
    if arm not in {'real_only','spis_gan','fw_gan','higanplus'}:
        raise ValueError(f'unknown OCR arm: {arm}')
    prepared=Path(prepared)
    options=ocr_settings(cfg)
    payload=dict(arm=arm,seed=int(seed),ocr={k:v for k,v in options.items() if k!='seeds'},
        prepared=read_json(prepared/'identity.json')['sha256'],
        real_train=sha256_file(prepared/'real_train.hdf5'),
        validation=sha256_file(prepared/'real_val.hdf5'),test=sha256_file(cfg['dataset']['test']))
    if arm!='real_only':
        if synthetic_metadata is None:
            raise ValueError('verified synthetic metadata is required')
        synthetic=Path(cfg['output_dir'])/'datasets'/arm/'synthetic.hdf5'
        payload['synthetic']=dict(identity=synthetic_metadata['identity_sha256'],sha256=sha256_file(synthetic))
    return payload
