"""Atomic, paired, generator-only datasets with strict resumable shard identities."""
import importlib
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from generators import ACTIVE_MODELS
from generators.external import provenance, read_model_config
from .common import atomic_json, bind_identity, digest_json, read_json, sha256_file
from .data import IAMCorpus, validate_iam, write_iam
from .manifest import _image_sha, prepare, read_cases


class _BudgetExpired(Exception):
    """Internal boundary: abandon an uncommitted shard, preserve committed rows."""


def _load_adapter(name, model_cfg, device):
    return importlib.import_module('generators.' + name).load_adapter(model_cfg, device)


def _context(cfg, name):
    if name not in ACTIVE_MODELS:
        raise ValueError(f'unsupported model: {name}')
    model = read_model_config(cfg['generation']['models'][name])
    if model['model'] != name:
        raise ValueError('model configuration name mismatch')
    height, native = cfg['dataset']['height'], model['native_height']
    conversion = model.get('export_conversion', 'none')
    if not ((native == height and conversion == 'none') or
            (native == 64 and height == 32 and conversion == 'native64_to_export32_bilinear')):
        raise ValueError('native/export height requires explicit supported conversion')
    shard_size = cfg['generation']['shard_size']
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size <= 0:
        raise ValueError('shard_size must be a positive integer')
    prepared = prepare(cfg)
    cases = read_cases(prepared)
    meta = read_json(prepared / 'metadata.json')
    source = provenance(model)
    # Include the actual adapter/worker implementation so changed export code
    # cannot silently reuse old pixels under the same checkpoint identity.
    code_files = [Path(__file__), *sorted((Path(__file__).parents[1] / 'generators').glob('*.py'))]
    payload = dict(schema='paired-generator-export-v1', model=name,
                   prepared_identity_sha256=meta['identity_sha256'],
                   manifest_sha256=meta['manifest_sha256'],
                   reference_pool_sha256=meta['artifact_sha256']['real_train.hdf5'],
                   count=cfg['generation']['count'], seed=cfg['generation']['seed'],
                   height=height, shard_size=shard_size, model_provenance=source,
                   implementation_sha256=digest_json({p.name: sha256_file(p) for p in code_files}))
    return model, prepared, cases, payload


def _shard_rows(directory, cases, identity, height, shard_size):
    """Validate each committed shard; orphan/uncommitted files are not reused."""
    result = []
    for start in range(0, len(cases), shard_size):
        stop = min(start + shard_size, len(cases))
        path = directory / f'shard_{start // shard_size:06d}.hdf5'
        marker = path.with_suffix('.json')
        if not marker.exists():
            result.append((start, stop, path, False))
            continue
        info = read_json(marker)
        expected = dict(identity_sha256=identity, sample_ids=list(range(start, stop)),
                        cases_sha256=digest_json(cases[start:stop]))
        if any(info.get(k) != v for k, v in expected.items()) or not path.exists() or info.get('sha256') != sha256_file(path):
            raise ValueError(f'shard cache identity/hash mismatch: {path.name}')
        validate_iam(path, stop - start)
        with IAMCorpus(path) as corpus:
            if corpus.height != height or corpus.texts != [c['target_text'] for c in cases[start:stop]] or list(corpus.writers) != [c['writer'] for c in cases[start:stop]]:
                raise ValueError('shard labels/writers/height mismatch')
        result.append((start, stop, path, True))
    return result


def _atomic_preview(path, hdf, cases, reference_hdf):
    with IAMCorpus(hdf) as corpus, IAMCorpus(reference_hdf) as references:
        count = min(12, len(corpus))
        column_width = max(300, max(int(w) for w in corpus.widths[:count]) + 20,
                           max(int(references.widths[c['reference_pool_index']]) for c in cases[:count]) + 20)
        row_height = max(corpus.height, references.height) + 32
        canvas = Image.new('RGB', (column_width * 2, count * row_height), 'white')
        draw = ImageDraw.Draw(canvas)
        for i in range(count):
            case = cases[i]
            top = i * row_height
            draw.text((10, top + 3), f'Reference #{case["sample_id"]}: {case["reference_text"]}', fill='black')
            draw.text((column_width + 10, top + 3), f'Target #{case["sample_id"]}: {case["target_text"]}', fill='black')
            canvas.paste(Image.fromarray(references.image(case['reference_pool_index'])).convert('RGB'), (10, top + 20))
            canvas.paste(Image.fromarray(corpus.image(i)).convert('RGB'), (column_width + 10, top + 20))
        canvas.save(path, format='PNG')


def _verify_pixels(directory, cases, payload):
    validate_iam(directory / 'synthetic.hdf5', len(cases))
    with IAMCorpus(directory / 'synthetic.hdf5') as corpus:
        source = payload['model_provenance']
        widths = [len(c['target_text']) * source['char_width'] * payload['height'] // source['native_height'] for c in cases]
        if (corpus.height != payload['height'] or corpus.texts != [c['target_text'] for c in cases]
                or list(corpus.writers) != [c['writer'] for c in cases] or list(corpus.widths) != widths):
            raise ValueError('dataset labels/writers/geometry mismatch')


def _verify_portable(directory, cases, payload):
    identity = digest_json(payload)
    metadata = read_json(directory / 'metadata.json')
    if (metadata.get('status') != 'complete' or metadata.get('identity_sha256') != identity
            or metadata.get('identity_payload') != payload or metadata.get('model') != payload['model']
            or metadata.get('sample_ids') != list(range(len(cases))) or metadata.get('count') != len(cases)
            or metadata.get('manifest_sha256') != payload['manifest_sha256']
            or metadata.get('model_provenance') != payload['model_provenance']):
        raise ValueError('dataset metadata identity/count/order mismatch')
    required = {'synthetic.hdf5', 'manifest.jsonl', 'preview.png'}
    if set(metadata.get('artifact_sha256', {})) != required:
        raise ValueError('missing portable dataset artifact hashes')
    for name, expected in metadata['artifact_sha256'].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f'dataset artifact hash mismatch: {name}')
    if sha256_file(directory / 'manifest.jsonl') != payload['manifest_sha256']:
        raise ValueError('dataset manifest identity mismatch')
    _verify_pixels(directory, cases, payload)
    if (directory / 'dataset.zip').exists():
        with zipfile.ZipFile(directory / 'dataset.zip') as archive:
            if set(archive.namelist()) != required | {'metadata.json'} or archive.testzip() is not None:
                raise ValueError('invalid dataset ZIP')
            for name in archive.namelist():
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != sha256_file(directory / name):
                    raise ValueError(f'ZIP content mismatch: {name}')
    return metadata


def _verify_complete(directory, cases, payload):
    identity = bind_identity(directory, payload)
    try:
        status, metadata = read_json(directory / 'status.json'), read_json(directory / 'metadata.json')
        if status['status'] != 'complete' or status['identity_sha256'] != identity or metadata['identity_sha256'] != identity:
            raise ValueError('dataset is not complete or identity mismatches')
        required = {'synthetic.hdf5', 'metadata.json', 'manifest.jsonl', 'preview.png', 'dataset.zip'}
        if set(status['artifact_sha256']) != required:
            raise ValueError('missing dataset artifact hashes')
        for name, expected in status['artifact_sha256'].items():
            if sha256_file(directory / name) != expected:
                raise ValueError(f'dataset artifact hash mismatch: {name}')
        rows = _shard_rows(directory / 'shards', cases, identity, payload['height'], payload['shard_size'])
        if not all(ready for _, _, _, ready in rows):
            raise ValueError('missing complete generation shard')
        return _verify_portable(directory, cases, payload)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f'incomplete/corrupt dataset: {directory}') from exc


def verify_dataset(cfg, model_name, verify_model=False):
    """Verify a portable bundle without loading or requiring generator binaries.

    Generation/resume additionally recomputes current model provenance and checks
    every cache shard. OCR binds the recorded provenance and exact dataset hashes.
    """
    if model_name not in ACTIVE_MODELS:
        raise ValueError(f'unsupported model: {model_name}')
    directory = Path(cfg['output_dir']) / 'datasets' / model_name
    if verify_model:
        _, _, cases, payload = _context(cfg, model_name)
        if not (directory / 'identity.json').exists():
            raise ValueError('missing dataset identity')
        return _verify_complete(directory, cases, payload)
    try:
        metadata = read_json(directory / 'metadata.json')
        payload = metadata['identity_payload']
        prepared = prepare(cfg)
        prep = read_json(prepared / 'metadata.json')
        expected = dict(model=model_name, prepared_identity_sha256=prep['identity_sha256'],
                        manifest_sha256=prep['manifest_sha256'],
                        reference_pool_sha256=prep['artifact_sha256']['real_train.hdf5'],
                        count=cfg['generation']['count'], seed=cfg['generation']['seed'], height=cfg['dataset']['height'])
        if any(payload.get(k) != v for k, v in expected.items()):
            raise ValueError('dataset/config/prepared identity mismatch')
        cases = read_cases(prepared)
        result = _verify_portable(directory, cases, payload)
        # An original run receipt, if retained, binds metadata and ZIP too.
        if (directory / 'status.json').exists():
            status = read_json(directory / 'status.json')
            if status.get('status') != 'complete' or status.get('identity_sha256') != result['identity_sha256']:
                raise ValueError('dataset receipt mismatch')
            for name, expected_hash in status['artifact_sha256'].items():
                if sha256_file(directory / name) != expected_hash:
                    raise ValueError(f'dataset receipt hash mismatch: {name}')
        return result
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f'incomplete/corrupt dataset: {directory}') from exc


def generate(cfg, model_name, device='cpu', time_budget_seconds=None):
    start_time = time.monotonic()
    model, prepared, cases, payload = _context(cfg, model_name)
    directory = Path(cfg['output_dir']) / 'datasets' / model_name
    directory.mkdir(parents=True, exist_ok=True)
    identity = bind_identity(directory, payload)
    if (directory / 'status.json').exists() and read_json(directory / 'status.json').get('status') == 'complete':
        _verify_complete(directory, cases, payload)
        return read_json(directory / 'status.json')
    height, shard_size = payload['height'], payload['shard_size']
    shards = directory / 'shards'
    shards.mkdir(exist_ok=True)
    completed = 0
    def status(state, **kwargs):
        result = dict(status=state, model=model_name, count=completed, total_count=len(cases),
                      identity_sha256=identity, **kwargs)
        atomic_json(directory / 'status.json', result)
        return result
    try:
        rows = _shard_rows(shards, cases, identity, height, shard_size)
        completed = sum(stop - first for first, stop, _, ready in rows if ready)
        adapter = None
        with IAMCorpus(prepared / 'real_train.hdf5') as references:
            for first, stop, path, ready in rows:
                if ready:
                    continue
                if time_budget_seconds is not None and time.monotonic() - start_time >= time_budget_seconds:
                    return status('paused')
                if adapter is None:
                    adapter = _load_adapter(model_name, model, device)
                def records():
                    for case in cases[first:stop]:
                        if time_budget_seconds is not None and time.monotonic() - start_time >= time_budget_seconds:
                            raise _BudgetExpired()
                        index = case['reference_pool_index']
                        reference = references.image(index)
                        if references.texts[index] != case['reference_text'] or _image_sha(reference) != case['reference_sha256'] or int(references.writers[index]) != case['writer']:
                            raise ValueError('paired manifest reference identity mismatch')
                        image = adapter.generate(reference, case['reference_text'], case['target_text'], case['seed'])
                        native = model['native_height']
                        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (native, len(case['target_text']) * model['char_width']):
                            raise ValueError('adapter returned invalid native geometry/dtype')
                        if native != height:
                            width = max(1, image.shape[1] * height // native)
                            image = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))
                        yield image, case['target_text'], case['writer']
                write_iam(path, records(), height, {'identity_sha256': identity, 'sample_ids': list(range(first, stop))})
                atomic_json(path.with_suffix('.json'), dict(identity_sha256=identity,
                            sample_ids=list(range(first, stop)), cases_sha256=digest_json(cases[first:stop]), sha256=sha256_file(path)))
                completed += stop - first
        # Recheck external files before publishing complete artifacts.
        if provenance(model) != payload['model_provenance']:
            raise ValueError('generator content identity changed during inference')
        temporary = Path(tempfile.mkdtemp(prefix='.bundle-', dir=directory))
        try:
            def all_records():
                for _, _, path, _ in rows:
                    with IAMCorpus(path) as corpus:
                        for i in range(len(corpus)):
                            yield corpus.image(i), corpus.texts[i], int(corpus.writers[i])
            write_iam(temporary / 'synthetic.hdf5', all_records(), height,
                      {'identity_sha256': identity, 'sample_ids': list(range(len(cases)))})
            shutil.copyfile(prepared / 'manifest.jsonl', temporary / 'manifest.jsonl')
            _atomic_preview(temporary / 'preview.png', temporary / 'synthetic.hdf5', cases, prepared / 'real_train.hdf5')
            artifact_hashes = {name: sha256_file(temporary / name) for name in ('synthetic.hdf5', 'manifest.jsonl', 'preview.png')}
            metadata = dict(status='complete', count=len(cases), sample_ids=list(range(len(cases))),
                            identity_sha256=identity, identity_payload=payload, manifest_sha256=payload['manifest_sha256'],
                            prepared_identity_sha256=payload['prepared_identity_sha256'],
                            model=model_name, native_height=model['native_height'], export_height=height,
                            export_conversion=model['export_conversion'], precision='float32',
                            model_provenance=payload['model_provenance'], artifact_sha256=artifact_hashes)
            atomic_json(temporary / 'metadata.json', metadata)
            with zipfile.ZipFile(temporary / 'dataset.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name in ('synthetic.hdf5', 'metadata.json', 'manifest.jsonl', 'preview.png'):
                    archive.write(temporary / name, name)
            for path in temporary.iterdir():
                os.replace(path, directory / path.name)
        finally:
            shutil.rmtree(temporary)
        hashes = {name: sha256_file(directory / name) for name in ('synthetic.hdf5', 'metadata.json', 'manifest.jsonl', 'preview.png', 'dataset.zip')}
        result = status('complete', artifact_sha256=hashes)
        _verify_complete(directory, cases, payload)
        return result
    except _BudgetExpired:
        return status('paused')
    except BaseException as exc:
        status('paused' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        raise
