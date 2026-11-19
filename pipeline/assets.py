"""Download released inference weights atomically, verifying their exact bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def download_checkpoint(name, cache, manifest=None):
    manifest = Path(manifest) if manifest else Path(__file__).resolve().parents[1]/'configs/checkpoint_assets.json'
    asset = json.loads(manifest.read_text())[name]
    filename = asset['filename']
    if Path(filename).name != filename or filename in ('', '.', '..'):
        raise ValueError('Invalid checkpoint asset filename')
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    target = cache/filename
    if target.exists():
        if target.is_symlink() or not target.is_file() or digest(target) != asset['sha256']:
            raise ValueError(f'Existing checkpoint does not match release checksum: {target}')
        return target.resolve()
    fd, temporary = tempfile.mkstemp(prefix='.'+filename+'-', dir=cache)
    try:
        with os.fdopen(fd, 'wb') as stream:
            print('Downloading checkpoint: '+name, file=sys.stderr, flush=True)
            request = urllib.request.Request(asset['url'], headers={'User-Agent': 'HGGAN-OCR-benchmark'})
            with urllib.request.urlopen(request, timeout=120) as source:
                shutil.copyfileobj(source, stream, length=1024*1024)
        if digest(temporary) != asset['sha256']:
            raise ValueError('Downloaded checkpoint checksum mismatch: '+name)
        os.link(temporary, target)  # Atomic no-overwrite publication.
        return target.resolve()
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['spis_gan','higanplus','fw_gan','main'])
    parser.add_argument('--cache', required=True)
    parser.add_argument('--manifest')
    args = parser.parse_args()
    print(download_checkpoint(args.model, args.cache, args.manifest))


if __name__ == '__main__': main()
