"""Pinned external sources, bounded imports and strict G/E/B-only inference.

No full GAN constructor, recognizer, discriminator, writer head or metric model
is instantiated. Effective source hashes include local compatibility changes.
"""
import argparse
import importlib
import importlib.machinery
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import yaml

from pipeline.common import digest_json, sha256_file
from pipeline.config import resolve_path


def read_model_config(path):
    with open(path, encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError('model configuration must be a mapping')
    for key in ('source_root', 'source_config', 'checkpoint'):
        cfg[key] = resolve_path(cfg[key])
    return cfg


def provenance(cfg):
    """Machine-independent identities; paths are operational, never semantic."""
    root = Path(cfg['source_root'])
    try:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError('generator source must be a pinned Git checkout') from exc
    if revision != cfg['source_revision']:
        raise ValueError(f'source revision mismatch: expected {cfg["source_revision"]}, found {revision}')
    source_files = {str(p.relative_to(root)): sha256_file(p)
                    for folder in ('networks', 'lib')
                    for p in sorted((root / folder).rglob('*.py'))}
    if not source_files:
        raise ValueError('missing generator source files')
    semantic = {k: v for k, v in cfg.items() if k not in ('source_root', 'source_config', 'checkpoint')}
    semantic.update(checkpoint_sha256=sha256_file(cfg['checkpoint']),
                    source_config_sha256=sha256_file(cfg['source_config']),
                    effective_source_sha256=digest_json(source_files))
    return dict(model_config_sha256=digest_json(semantic), **semantic,
                effective_source_files=source_files)


def bounded_namespace(name, path):
    """Provide only a selected package path; never execute its __init__.py."""
    path = Path(path).resolve()
    for key, value in list(sys.modules.items()):
        if key != name and not key.startswith(name + '.'):
            continue
        locations = getattr(value, '__path__', None)
        if locations is None:
            filename = getattr(value, '__file__', None)
            if filename is None or not Path(filename).resolve().is_relative_to(path):
                raise RuntimeError(f'external package namespace collision: {key}; use a separate process')
        elif [str(Path(p).resolve()) for p in locations] != [str(path)]:
            raise RuntimeError(f'external package namespace collision: {key}; use a separate process')
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[name] = package
    importlib.invalidate_caches()


def load_weights(path):
    """Legacy released FID scalars, narrowly allowlisted without arbitrary pickle."""
    import torch
    from numpy._core.multiarray import scalar
    safe = [(scalar, 'numpy._core.multiarray.scalar'), (scalar, 'numpy.core.multiarray.scalar'),
            np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


class GeneratorAdapter:
    def __init__(self, cfg, device, family):
        import torch
        self.torch = torch
        self.cfg, self.device = cfg, torch.device(device)
        self.metadata = provenance(cfg)
        native = 64 if family == 'higanplus' else 32
        if cfg['model'] != family or cfg['native_height'] != native or cfg['char_width'] != native // 2:
            raise ValueError('adapter family/native geometry mismatch')
        if cfg['internal_polarity'] != ('black' if family == 'higanplus' else 'white'):
            raise ValueError('wrong checkpoint pixel polarity')
        self.native_height, self.char_width = native, native // 2
        root = Path(cfg['source_root'])
        bounded_namespace('networks', root / 'networks')
        bounded_namespace('lib', root / 'lib')
        networks = importlib.import_module('networks.BigGAN_networks')
        module = importlib.import_module('networks.module32' if family == 'spis_gan' else 'networks.module')
        self.alphabet = importlib.import_module('lib.alphabet').Alphabets['all']
        with open(cfg['source_config'], encoding='utf-8') as stream:
            model = yaml.safe_load(stream)
        backbone_key = 'StyBackbone' if family == 'higanplus' else 'SharedBackbone'
        if model['img_height'] != native or model['GenModel']['resolution'] != native:
            raise ValueError('source config does not describe native checkpoint geometry')
        if model['GenModel']['n_class'] != len(self.alphabet):
            raise ValueError('source alphabet/config mismatch')
        self.generator = networks.Generator(**model['GenModel'])
        self.encoder = module.StyleEncoder(**model['EncModel'])
        self.backbone = getattr(module, 'StyleBackbone' if family == 'higanplus' else 'SharedBackbone')(**model[backbone_key])
        # Memory-map CPU storage; never deserialize arbitrary pickle objects or
        # transfer unused D/R/W/P checkpoint tensors to the accelerator.
        checkpoint = load_weights(cfg['checkpoint'])
        required = ('Generator', 'StyleEncoder', 'StyleBackbone' if family == 'higanplus' else 'SharedBackbone')
        for network, key in zip((self.generator, self.encoder, self.backbone), required):
            network.load_state_dict(checkpoint[key], strict=True)
            network.to(self.device, dtype=torch.float32).eval().requires_grad_(False)
        del checkpoint
        self.noise_dim = model['GenModel']['style_dim'] - model['EncModel']['style_dim']
        if self.noise_dim != (0 if family == 'higanplus' else 32):
            raise ValueError('unexpected native latent dimensions')
        self.metadata.update(native_height=native, char_width=self.char_width,
                             latent_policy='mean_style' if not self.noise_dim else 'cpu_seeded_noise32_then_mean_style96',
                             loaded_components=list(required), precision='float32')

    def generate(self, reference_uint8, reference_text, target_text, seed):
        torch = self.torch
        import torch.nn.functional as functional
        if (not isinstance(reference_uint8, np.ndarray) or reference_uint8.dtype != np.uint8
                or reference_uint8.ndim != 2 or min(reference_uint8.shape) <= 0):
            raise ValueError('reference must be positive uint8 H x W')
        if not reference_text or not target_text:
            raise ValueError('reference and target text must be nonempty')
        if any(c not in self.alphabet for c in target_text):
            raise ValueError('target contains unsupported model alphabet character')
        # Native style geometry is transcript-length based, not source width.
        # All symbols share the same encoder geometry floor (no special cases).
        width = max(len(reference_text) * self.char_width, int(self.cfg['encoder_min_width']))
        image = torch.from_numpy(reference_uint8.copy()).float()[None, None] / 127.5 - 1
        image = functional.interpolate(image, size=(self.native_height, width), mode='bilinear', align_corners=False)
        if self.cfg['internal_polarity'] == 'black':
            image = -image
        image = image.to(self.device)
        lengths = torch.tensor([width], dtype=torch.long, device=self.device)
        labels = torch.tensor([[self.alphabet.index(c) for c in target_text]], dtype=torch.long, device=self.device)
        text_lengths = torch.tensor([len(target_text)], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            style = self.encoder(image, lengths, self.backbone, vae_mode=False)
            if self.noise_dim:
                rng = torch.Generator(device='cpu').manual_seed(int(seed))
                noise = torch.randn(1, self.noise_dim, generator=rng, dtype=torch.float32, device='cpu').to(self.device)
                style = torch.cat((noise, style), dim=-1)
            output = self.generator(style, labels, text_lengths)
            valid_width = len(target_text) * self.char_width
            if output.shape[:3] != (1, 1, self.native_height) or output.shape[-1] < valid_width:
                raise ValueError(f'generator output geometry mismatch: {tuple(output.shape)}')
            output = output[0, 0, :, :valid_width]
            if not torch.isfinite(output).all():
                raise ValueError('nonfinite generator output')
            if self.cfg['internal_polarity'] == 'black':
                output = -output
            return ((output.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()


def adapter_cli(model_name):
    parser = argparse.ArgumentParser(description=f'Generate a paired downloadable {model_name} IAM dataset')
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-config')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--time-budget-seconds', type=float)
    args = parser.parse_args()
    from pipeline.config import load_config
    from pipeline.generation import generate
    cfg = load_config(args.config)
    if args.model_config:
        cfg['generation']['models'][model_name] = resolve_path(args.model_config)
    print(json.dumps(generate(cfg, model_name, args.device, args.time_budget_seconds), sort_keys=True))
