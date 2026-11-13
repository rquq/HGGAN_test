"""Explicit W&B lifecycle shared by OCR and generator worker wrappers.

Call tracking_run BEFORE loading a model. Credentials belong in environment
variables only; never pass a complete user configuration as run metadata.
"""
from contextlib import contextmanager
from pathlib import Path

from .common import atomic_json, digest_json


class TrackingRun:
    def __init__(self, run, run_id):
        self._run, self.id = run, run_id

    def log(self, values, step=None):
        self._run.log(values, step=step)


@contextmanager
def tracking_run(cfg, directory, stage, arm, seed, metadata=None):
    """Yield a .log(values, step=None)/.id handle; finish on success/failure.

    Public Task4 API: tracking_run(cfg, job_dir, 'generation', model_name,
    generation_seed, {'identity': content_sha}). Enter around generate(), not
    after adapter construction. Online resumes the same job; offline retains the
    stable ID in separate local invocation records for subsequent wandb sync.
    """
    import wandb
    options = cfg.get('wandb', {})
    mode = options.get('mode', 'disabled')
    if mode not in {'online', 'offline', 'disabled'}:
        raise ValueError('wandb.mode must be online, offline or disabled')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe = {key: value for key, value in (metadata or {}).items()
            if key in {'identity', 'count', 'max_steps', 'precision'}}
    run_id = digest_json(dict(stage=stage, arm=arm, seed=int(seed), metadata=safe))[:24]
    atomic_json(directory / f'wandb_{stage}.json',
                dict(id=run_id, mode=mode, stage=stage, arm=arm, seed=int(seed)))
    run = wandb.init(project=options.get('project', 'iam-ocr-augmentation'),
                     entity=options.get('entity'), mode=mode, id=run_id,
                     resume='allow' if mode == 'online' else None,
                     name=f'{stage}-{arm}-{seed}', dir=str(directory),
                     config=dict(stage=stage, arm=arm, seed=int(seed), **safe),
                     settings=wandb.Settings(console='auto'))
    try:
        yield TrackingRun(run, run_id)
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)
