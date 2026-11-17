"""Shared pure OCR payload builder; never creates jobs or loads models."""
from pathlib import Path

from .common import read_json, sha256_file, bind_identity, digest_json, atomic_json
from .config import ocr_settings


def bind_ocr_identity(directory, payload):
    """Repair only a proven zero-step failure caused by numeric-string parsing."""
    directory = Path(directory)
    path, status = directory/'identity.json', directory/'status.json'
    if path.is_file() and status.is_file():
        saved, state = read_json(path), read_json(status)
        if saved != dict(payload=payload,sha256=digest_json(payload)):
            prior = saved.get('payload', {})
            normalized = dict(prior)
            options = ocr_settings({'ocr':prior.get('ocr', {})})
            normalized['ocr'] = {k:v for k,v in options.items() if k!='seeds'}
            protected = any(directory.glob('checkpoint-*')) or any(
                (directory/name).exists() or (directory/name).is_symlink()
                for name in ('best','last','test_metrics.json','test_predictions.jsonl'))
            if (saved.get('sha256')==digest_json(prior) and normalized==payload
                    and state.get('status')=='failed' and state.get('global_step')==0 and not protected):
                backup = directory/'identity.before-numeric-fix.json'
                if backup.exists() and read_json(backup)!=saved:
                    raise ValueError('conflicting numeric-fix identity backup')
                atomic_json(backup,saved)
                atomic_json(path,dict(payload=payload,sha256=digest_json(payload)))
    return bind_identity(directory,payload)


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
