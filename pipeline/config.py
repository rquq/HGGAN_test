"""Configuration without model imports or model-file existence requirements."""
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'output_dir': 'outputs/benchmark32', 'split_seed': 123456,
    'dataset': {'train': '../../iam/iam_train_32.hdf5', 'test': '../../iam/iam_test_32.hdf5',
                'validation': None, 'height': 32, 'real_count': 5000, 'validation_fraction': .1},
    'generation': {'count': 25000, 'seed': 123456, 'max_text_length': 20,
                   'alphabet': ' ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',
                   'shard_size': 100, 'models': {n: f'configs/generators/{n}.yml'
                                               for n in ('spis_gan', 'fw_gan', 'higanplus')}},
    'ocr': {'model_name': 'microsoft/trocr-small-stage1',
            'revision': 'd94c4048159050bb51e4105bad0a45dbfd5f3a02', 'max_steps': 10000,
            'micro_batch_size': 2, 'gradient_accumulation_steps': 16,
            'learning_rate': 5e-5, 'weight_decay': .01, 'max_grad_norm': 1.,
            'warmup_ratio': .05, 'eval_steps': 500, 'save_steps': 500,
            'max_new_tokens': 64, 'num_beams': 1, 'fp16': False, 'bf16': False,
            'seeds': [123456]},
    'wandb': {'mode': 'disabled', 'project': 'iam-ocr-augmentation'},
}


def ocr_settings(cfg):
    """Validated semantic recipe, usable without importing model libraries."""
    result = dict(DEFAULTS['ocr'], **cfg.get('ocr', {}))
    if (result['model_name'] != DEFAULTS['ocr']['model_name'] or
            result['revision'] != DEFAULTS['ocr']['revision']):
        raise ValueError('only the pinned TrOCR-small-stage1 model is supported')
    for key in ('max_steps', 'micro_batch_size', 'gradient_accumulation_steps',
                'eval_steps', 'save_steps', 'max_new_tokens'):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] <= 0:
            raise ValueError(f'ocr.{key} must be a positive integer')
    if result['save_steps'] != result['eval_steps']:
        raise ValueError('evaluation and save intervals must be aligned')
    if result['num_beams'] != 1 or result['fp16'] or result['bf16']:
        raise ValueError('this benchmark requires greedy generation and FP32')
    if not 0 <= result['warmup_ratio'] <= 1:
        raise ValueError('invalid warmup fraction')
    return result


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def resolve_path(path):
    value = Path(path).expanduser()
    return str((PROJECT_ROOT / value).resolve())


def load_config(path):
    with open(path, encoding='utf-8') as stream:
        supplied = yaml.safe_load(stream) or {}
    if not isinstance(supplied, dict):
        raise ValueError('configuration must be a mapping')
    cfg = _merge(deepcopy(DEFAULTS), supplied)
    # An explicit pool replaces defaults, including the native64-only cohort.
    if 'models' in supplied.get('generation', {}):
        cfg['generation']['models'] = dict(supplied['generation']['models'])
    cfg['output_dir'] = resolve_path(cfg['output_dir'])
    for key in ('train', 'test', 'validation'):
        if cfg['dataset'].get(key) is not None:
            cfg['dataset'][key] = resolve_path(cfg['dataset'][key])
    # Models are paths, not adapter instances. Preflight opens these later.
    cfg['generation']['models'] = {n: resolve_path(p) for n, p in cfg['generation']['models'].items()}
    return cfg
