"""Small atomic artifacts and machine-independent content identities."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(obj, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(obj):
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def bind_identity(directory, payload):
    directory = Path(directory)
    identity = {'sha256': digest_json(payload), 'payload': payload}
    path = directory / 'identity.json'
    if path.exists():
        if read_json(path) != identity:
            raise ValueError(f'identity mismatch: {directory}')
    else:
        atomic_json(path, identity)
    return identity['sha256']
