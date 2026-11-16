"""Create the benchmark's Python 3.11 worker without the host's ensurepip."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

UV_VERSION = '0.12.14'


def healthy(python):
    if not Path(python).is_file(): return False
    try:
        check = subprocess.run([str(python), '-c',
            'import sys,pip; assert sys.version_info[:2] == (3,11)'],
            capture_output=True, text=True)
    except OSError:
        return False
    return check.returncode == 0


def create_runtime(runtime, uv, env=None):
    runtime = Path(runtime).absolute()
    python = runtime/'bin/python'
    if healthy(python): return python
    if runtime.exists():
        backup = runtime.with_name(runtime.name+f'.incomplete-{time.time_ns()}')
        runtime.rename(backup)  # Preserve the failed environment; never delete it.
        print('Preserved incomplete worker environment: '+str(backup), file=sys.stderr)
    subprocess.run([str(uv), 'venv', '--python', '3.11', '--managed-python',
                    '--seed', str(runtime)], check=True, env=env, stdout=sys.stderr)
    if not healthy(python):
        raise RuntimeError('Worker must have Python 3.11 and pip; creation did not pass its health check')
    return python


def ensure_runtime(runtime):
    runtime = Path(runtime).absolute()
    if healthy(runtime/'bin/python'): return runtime/'bin/python'
    bootstrap = runtime.with_name(runtime.name+'_bootstrap')
    uv = bootstrap/'bin/uv'
    valid = uv.is_file() and subprocess.run([str(uv),'--version'],capture_output=True,
        text=True).stdout.strip().startswith('uv '+UV_VERSION+' ')
    if not valid:
        # Kaggle's existing pip installs only this bootstrap tool into a private
        # target, without changing the notebook's own packages or running venv.
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--upgrade', '--target', str(bootstrap), 'uv=='+UV_VERSION],
                       check=True, stdout=sys.stderr)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(bootstrap/'python'),
               UV_CACHE_DIR=str(bootstrap/'cache'))
    return create_runtime(runtime, uv, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    print(ensure_runtime(args.runtime))
