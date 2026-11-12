"""Frozen train-only reference/lexicon manifest shared by all generator arms."""
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

from .common import atomic_json, bind_identity, digest_json, read_json, sha256_file
from .data import IAMCorpus, validate_iam, write_iam


def _integer(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ValueError(f'{name} must be an integer in [1,{maximum}]')
    return value


def _image_sha(image):
    # Geometry is part of the image identity; bytes alone do not encode width.
    digest = hashlib.sha256()
    digest.update(str(tuple(image.shape)).encode('ascii'))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _groups(corpus):
    return [digest_json([_image_sha(corpus.image(i)), text]) for i, text in enumerate(corpus.texts)]


def _balanced(rows, writers, count, rng):
    pools = {}
    for row in rows:
        pools.setdefault(int(writers[row]), []).append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    order = sorted(pools)
    rng.shuffle(order)
    selected = []
    while len(selected) < count:
        progress = False
        for writer in order:
            if pools[writer]:
                selected.append(pools[writer].pop())
                progress = True
                if len(selected) == count:
                    return selected
        if not progress:
            raise ValueError(f'not enough eligible training rows for real_count={count}')
    return selected


def read_cases(path):
    path = Path(path)
    if path.is_dir():
        path /= 'manifest.jsonl'
    with path.open(encoding='utf-8') as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    if [r['sample_id'] for r in cases] != list(range(len(cases))):
        raise ValueError('manifest sample IDs must be ordered and contiguous')
    return cases


def _verify_existing(directory, payload):
    if not (directory / 'identity.json').exists():
        raise ValueError('incomplete prepared directory: missing identity')
    identity = bind_identity(directory, payload)
    try:
        metadata = read_json(directory / 'metadata.json')
        if metadata['identity_sha256'] != identity or metadata['dataset_sha256'] != payload['dataset_sha256']:
            raise ValueError('prepared metadata identity mismatch')
        if metadata['status'] != 'complete':
            raise ValueError('prepared export is not complete')
        required = {'split.json', 'manifest.jsonl', 'real_train.hdf5', 'real_val.hdf5'}
        if set(metadata['artifact_sha256']) != required:
            raise ValueError('prepared metadata is missing required artifact hashes')
        if metadata['manifest_sha256'] != metadata['artifact_sha256']['manifest.jsonl']:
            raise ValueError('prepared manifest hash metadata disagrees')
        for filename, expected in metadata['artifact_sha256'].items():
            if sha256_file(directory / filename) != expected:
                raise ValueError(f'prepared artifact hash mismatch: {filename}')
        validate_iam(directory / 'real_train.hdf5', metadata['real_train_count'])
        validate_iam(directory / 'real_val.hdf5', metadata['real_val_count'])
        if len(read_cases(directory)) != metadata['generation_count']:
            raise ValueError('prepared manifest count mismatch')
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f'incomplete/corrupt prepared export: {directory}') from exc
    return directory


def prepare(cfg):
    dataset, generation = cfg['dataset'], cfg['generation']
    count = _integer(generation['count'], 'generation.count', 25000)
    real_count = _integer(dataset['real_count'], 'dataset.real_count', 5000)
    height = _integer(dataset['height'], 'dataset.height', 4096)
    max_length = _integer(generation['max_text_length'], 'max_text_length', 10000)
    alphabet = generation['alphabet']
    if not isinstance(alphabet, str) or not alphabet:
        raise ValueError('generation.alphabet must be a nonempty literal character string')
    fraction = dataset.get('validation_fraction', .1)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError('validation_fraction must lie strictly between zero and one')
    hashes = {key: sha256_file(dataset[key]) if dataset.get(key) else None
              for key in ('train', 'test', 'validation')}
    payload = dict(schema_version=1, dataset_sha256=hashes, height=height,
                   real_count=real_count, validation_fraction=fraction,
                   split_seed=cfg['split_seed'], generation_count=count,
                   generation_seed=generation['seed'], max_text_length=max_length, alphabet=alphabet)
    directory = Path(cfg['output_dir']) / 'prepared'
    if directory.exists():
        return _verify_existing(directory, payload)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.prepared-', dir=directory.parent))
    try:
        with ExitStack() as stack:
            train = stack.enter_context(IAMCorpus(dataset['train']))
            test = stack.enter_context(IAMCorpus(dataset['test']))
            explicit = bool(dataset.get('validation'))
            val = stack.enter_context(IAMCorpus(dataset['validation'])) if explicit else train
            if any(c.height != height for c in (train, test, val)):
                raise ValueError('all supplied IAM datasets must match declared cohort height')
            train_groups, test_groups = _groups(train), set(_groups(test))
            val_groups = _groups(val) if explicit else train_groups
            rng = random.Random(cfg['split_seed'])
            if explicit:
                validation_rows = list(range(len(val)))
                heldout_writers = set(int(w) for w in val.writers)
            else:
                writers = sorted(set(int(w) for w in train.writers))
                if len(writers) < 2:
                    raise ValueError('writer holdout requires at least two training writers')
                rng.shuffle(writers)
                heldout_writers = set(writers[:max(1, math.ceil(len(writers) * fraction))])
                validation_rows = [i for i, w in enumerate(train.writers) if int(w) in heldout_writers]
            # Block ALL original validation groups, even those excluded by test collision.
            blocked_groups = test_groups | {val_groups[i] for i in validation_rows}
            validation_rows = [i for i in validation_rows if val_groups[i] not in test_groups]
            if not validation_rows:
                raise ValueError('no validation rows remain after test collision exclusion')
            eligible = [i for i, w in enumerate(train.writers)
                        if int(w) not in heldout_writers and train_groups[i] not in blocked_groups]
            selected = _balanced(eligible, train.writers, real_count, rng)
            lexicon = sorted({train.texts[i] for i in selected
                              if 0 < len(train.texts[i]) <= max_length and set(train.texts[i]) <= set(alphabet)})
            if len(lexicon) < 2:
                raise ValueError('selected training pool must provide at least two valid distinct target texts')
            split = dict(train_rows=selected, validation_rows=validation_rows,
                         validation_source='validation' if explicit else 'train',
                         heldout_writers=sorted(heldout_writers), eligible_training_count=len(eligible),
                         test_count=len(test), test_rows_unchanged=True)
            atomic_json(temporary / 'split.json', split)
            for name, corpus, rows in [('real_train', train, selected), ('real_val', val, validation_rows)]:
                write_iam(temporary / f'{name}.hdf5',
                          ((corpus.image(i), corpus.texts[i], int(corpus.writers[i])) for i in rows),
                          height, metadata={'source_sha256': hashes['train' if corpus is train else 'validation']})
            case_rng = random.Random(generation['seed'])
            with (temporary / 'manifest.jsonl').open('w', encoding='utf-8') as stream:
                for sample_id in range(count):
                    pool_index = case_rng.randrange(real_count)
                    row = selected[pool_index]
                    reference_text = train.texts[row]
                    target = case_rng.choice([text for text in lexicon if text != reference_text])
                    case = dict(sample_id=sample_id, reference_row=row, reference_pool_index=pool_index,
                                reference_text=reference_text, reference_sha256=_image_sha(train.image(row)),
                                writer=int(train.writers[row]), target_text=target,
                                seed=case_rng.randrange(2**31))
                    stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            # Reject source mutation during the read/export window.
            if any(dataset.get(k) and sha256_file(dataset[k]) != v for k, v in hashes.items()):
                raise ValueError('dataset identity changed during preparation')
            identity = bind_identity(temporary, payload)
            filenames = ('split.json', 'manifest.jsonl', 'real_train.hdf5', 'real_val.hdf5')
            artifact_hashes = {name: sha256_file(temporary / name) for name in filenames}
            atomic_json(temporary / 'metadata.json', dict(
                status='complete', identity_sha256=identity, dataset_sha256=hashes,
                manifest_sha256=artifact_hashes['manifest.jsonl'], artifact_sha256=artifact_hashes,
                real_train_count=real_count, real_val_count=len(validation_rows), test_count=len(test),
                generation_count=count, lexicon_count=len(lexicon), lexicon=lexicon,
                eligible_training_count=len(eligible), heldout_writer_count=len(heldout_writers),
                training_writer_count=len(set(int(train.writers[i]) for i in selected))))
        os.rename(temporary, directory)
        return directory
    finally:
        # Only the task-owned temporary export; never touch an existing prepared result.
        if temporary.exists():
            shutil.rmtree(temporary)
