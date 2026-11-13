"""Read-only checks, explicit portable restore and atomic prepared downloads."""
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from generators import ACTIVE_MODELS
from .common import digest_json, read_json, sha256_file
from .data import validate_iam
from .manifest import prepare


def export_prepared(cfg):
    prepared = prepare(cfg)
    target = Path(cfg['output_dir']) / 'prepared.zip'
    fd, temporary = tempfile.mkstemp(prefix='.prepared-', dir=target.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(prepared.iterdir()):
                if path.is_file():
                    archive.write(path, path.name)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def check(cfg, stage='generation'):
    if stage not in {'generation', 'ocr', 'all'}:
        raise ValueError('stage must be generation, ocr or all')
    names = list(cfg['generation']['models'])
    if any(name not in ACTIVE_MODELS for name in names):
        raise ValueError('unsupported generator in configuration')
    for key in ('train', 'test', 'validation'):
        if cfg['dataset'].get(key):
            validate_iam(cfg['dataset'][key])
    prepared = prepare(cfg)
    if stage in {'generation', 'all'}:
        from .generation import _context
        for name in names:
            _context(cfg, name)  # No adapter/model construction.
    else:
        from .generation import verify_dataset
        for name in names:
            verify_dataset(cfg, name)
    return dict(status='complete', stage=stage, models=names,
                prepared_identity=read_json(prepared/'identity.json')['sha256'])


def restore_outputs(previous, destination):
    """Merge only explicit artifacts, validating *all* conflicts before copying.

    Paused shard caches are allowed; prepared data and saved OCR checkpoints
    must be complete. Never follow arbitrary links.
    """
    previous, destination = Path(previous).resolve(), Path(destination).resolve()
    if (previous == destination or previous.is_relative_to(destination)
            or destination.is_relative_to(previous)):
        raise ValueError('previous and output directories must be separate')
    required = ['identity.json','metadata.json','split.json','manifest.jsonl',
                'real_train.hdf5','real_val.hdf5']
    if any(not (previous/'prepared'/name).is_file() for name in required):
        raise ValueError('previous output has incomplete prepared artifacts')
    identity = read_json(previous/'prepared/identity.json')
    if identity.get('sha256') != digest_json(identity.get('payload')):
        raise ValueError('previous prepared identity is corrupt')
    from .manifest import _verify_existing
    _verify_existing(previous/'prepared', identity['payload'])
    alias_plans = {}
    for directory in (previous/'ocr').glob('*/seed_*'):
        if directory.is_dir() and not directory.is_symlink():
            from .ocr import _checkpoint_aliases, _check_alias_representation, _check_best_receipt
            plan = _checkpoint_aliases(directory)
            alias_plans[directory] = plan
            target_dir = destination / directory.relative_to(previous)
            for name,canonical in plan.items():
                _check_alias_representation(target_dir/name,directory/canonical)
            recorded = _check_best_receipt(target_dir/'best.json',plan.get('best'))
            if recorded and 'best' not in plan:
                raise ValueError('destination best receipt has no native selection')
    entries = []
    allowed = ['prepared', 'datasets', 'ocr', 'logs', 'prepared.zip', 'runner_status.json',
               'results.json', 'results.csv']
    def visit(source):
        target = destination / source.relative_to(previous)
        if source.parent in alias_plans and source.name in {'best','last','best.json'}:
            # Canonical saves are copied once. Derived mirrors are rebuilt.
            return
        if source.is_symlink():
            relative = os.readlink(source)
            resolved = source.resolve()
            if (Path(relative).is_absolute() or not resolved.is_relative_to(previous)
                    or source.name not in {'best','last'} or not resolved.is_dir()):
                raise ValueError(f'unsafe saved output link: {source}')
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or os.readlink(target) != relative:
                    raise ValueError(f'restore conflict: {target}')
            entries.append((source,target,'link'))
        elif source.is_dir():
            if source.name.startswith('checkpoint-'):
                from .ocr import _validate_checkpoint
                _validate_checkpoint(source)
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise ValueError(f'restore conflict: {target}')
            entries.append((source,target,'directory'))
            for child in sorted(source.iterdir()):
                visit(child)
        elif source.is_file() and source.suffix != '.pid':
            if target.is_symlink() or (target.exists() and
                    (not target.is_file() or sha256_file(source) != sha256_file(target))):
                raise ValueError(f'restore conflict: {target}')
            entries.append((source,target,'file'))
    for name in allowed:
        path = previous/name
        if path.exists() or path.is_symlink():
            visit(path)
    destination.mkdir(parents=True, exist_ok=True)
    for source,target,kind in entries:
        if kind == 'directory':
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists() and not target.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            if kind == 'link':
                target.symlink_to(os.readlink(source), target_is_directory=True)
            else:
                fd, temporary = tempfile.mkstemp(prefix='.restore-',dir=target.parent)
                os.close(fd)
                try:
                    shutil.copyfile(source,temporary)
                    os.replace(temporary,target)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
    for directory,plan in alias_plans.items():
        from .ocr import _normalize_checkpoint_aliases
        target_dir = destination / directory.relative_to(previous)
        _normalize_checkpoint_aliases(target_dir,plan)
    return dict(status='complete', restored_files=sum(k!='directory' for _,_,k in entries))
