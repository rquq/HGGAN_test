#!/usr/bin/env python3
"""Evaluate an x32 FW-GAN or SpiS-GAN checkpoint with DEV's x64 metrics.

The external model keeps its native 32-pixel generator and style encoder. IAM
style references are downsampled from x64 to x32 before style encoding, and
generated words are upsampled back to x64 before every DEV metric. This is the
only resolution adaptation; metric implementations and evaluator checkpoints
come directly from the DEV branch.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import h5py
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from munch import Munch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml


METRIC_COLUMNS = (
    "fid", "kid", "hwd", "cmmd", "cer", "wer",
    "is_gen", "is_org", "psnr", "mssim", "wier",
)

CACHE_VERSION = 3
CACHE_SHARD_BATCHES = 64


class DiskBatchCache:
    """Disk-backed generated batches; only one small shard is resident at once."""

    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.manifest = manifest
        self.paths = [root / name for name in manifest["shards"]]
        self.num_batches = int(manifest["num_batches"])
        self.num_images = int(manifest["num_images"])

    def __len__(self):
        return self.num_batches

    def iter_raw(self):
        for path in self.paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            for batch in shard:
                yield batch
            del shard

    def first(self):
        return next(self.iter_raw())


def current_memory_gib():
    values = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("VmRSS:", "VmSwap:")):
                    key, amount, _unit = line.split()
                    values[key.rstrip(":")] = int(amount) / (1024 ** 2)
    except OSError:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    _key, amount, _unit = line.split()
                    values["MemAvailable"] = int(amount) / (1024 ** 2)
                    break
    except OSError:
        pass
    return values


def report_resources(stage: str, device=None, reset_peak=False):
    memory = current_memory_gib()
    parts = [
        f"RSS={memory.get('VmRSS', float('nan')):.2f} GiB",
        f"swap={memory.get('VmSwap', 0.0):.2f} GiB",
        f"RAM_available={memory.get('MemAvailable', float('nan')):.2f} GiB",
    ]
    if device is not None and torch.device(device).type == "cuda" and torch.cuda.is_available():
        cuda_device = torch.device(device)
        # reset_peak_memory_stats requires an initialized CUDA context on some
        # PyTorch/WSL combinations, even when is_available() already returned True.
        torch.cuda.init()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        free, total = torch.cuda.mem_get_info(cuda_device)
        parts.extend([
            f"GPU_alloc={torch.cuda.memory_allocated(cuda_device) / 2**30:.2f} GiB",
            f"GPU_reserved={torch.cuda.memory_reserved(cuda_device) / 2**30:.2f} GiB",
            f"GPU_peak={torch.cuda.max_memory_allocated(cuda_device) / 2**30:.2f} GiB",
            f"GPU_free={free / 2**30:.2f}/{total / 2**30:.2f} GiB",
        ])
    print(f"[resources:{stage}] " + "; ".join(parts), flush=True)


class LimitedLoader:
    """A reiterable view over at most ``limit`` batches of a DataLoader."""

    def __init__(self, loader, limit: int | None):
        self.loader = loader
        self.limit = len(loader) if not limit or limit < 1 else min(limit, len(loader))

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.limit)

    def __len__(self):
        return self.limit


class NativeX32Hdf5Dataset(Dataset):
    """Direct reader for canonical x32 IAM HDF5 files."""

    def __init__(self, path: Path, converter):
        self.path = Path(path)
        self.converter = converter
        with h5py.File(self.path, "r") as handle:
            self.imgs = handle["imgs"][:]
            self.lbs = handle["lbs"][:]
            self.img_seek_idxs = handle["img_seek_idxs"][:]
            self.lb_seek_idxs = handle["lb_seek_idxs"][:]
            self.img_lens = handle["img_lens"][:]
            self.lb_lens = handle["lb_lens"][:]
            self.wids = (
                handle["wids"][:]
                if "wids" in handle
                else np.zeros(len(self.img_lens), dtype=np.int64)
            )

    def __len__(self):
        return len(self.img_lens)

    def __getitem__(self, index):
        image_start = int(self.img_seek_idxs[index])
        image_length = int(self.img_lens[index])
        label_start = int(self.lb_seek_idxs[index])
        label_length = int(self.lb_lens[index])
        image = np.array(
            self.imgs[:, image_start:image_start + image_length], copy=True
        )
        text = "".join(
            chr(int(value))
            for value in self.lbs[label_start:label_start + label_length]
        )
        image = torch.from_numpy(image).unsqueeze(0).float().div(127.5).sub(1.0)
        return {
            "image": image,
            "label": self.converter.encode(text),
            "wid": int(self.wids[index]),
        }


def collect_native_x32(batch):
    """Collate native x32 words without any image resizing."""
    image_lens = torch.tensor(
        [sample["image"].size(-1) for sample in batch], dtype=torch.int32
    )
    label_lens = torch.tensor(
        [len(sample["label"]) for sample in batch], dtype=torch.int32
    )
    max_width = int(((int(image_lens.max()) + 15) // 16) * 16)
    images = torch.full(
        (len(batch), 1, 32, max_width), -1.0, dtype=torch.float32
    )
    labels = torch.zeros(
        len(batch), int(label_lens.max()), dtype=torch.long
    )
    for row, sample in enumerate(batch):
        width = sample["image"].size(-1)
        length = len(sample["label"])
        images[row, :, :, :width] = sample["image"]
        labels[row, :length] = torch.as_tensor(sample["label"], dtype=torch.long)
    return {
        "org_imgs": images,
        "org_img_lens": image_lens,
        "style_imgs": images.clone(),
        "style_img_lens": image_lens.clone(),
        "lbs": labels,
        "lb_lens": label_lens,
        "wids": torch.tensor([sample["wid"] for sample in batch], dtype=torch.long),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FW-GAN/SpiS-GAN with the complete DEV metric suite."
    )
    parser.add_argument("--baseline", required=True, choices=("spis", "fw"))
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-file", default=None,
                        help="Explicit HDF5 file used with --native-x32.")
    parser.add_argument("--native-x32", action="store_true",
                        help="Evaluate native x32 data with no spatial resizing.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 evaluates the complete IAM test split.")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated DEV metrics, 'all', or 'none' for a generation smoke test.",
    )
    return parser.parse_args()


def fail_missing(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)


def purge_packages(*package_names):
    prefixes = tuple(f"{name}." for name in package_names)
    for module_name in list(sys.modules):
        if module_name in package_names or module_name.startswith(prefixes):
            del sys.modules[module_name]


def import_dev_api(dev_root: Path, dataset_root: Path):
    """Import and retain DEV evaluator objects before baseline package imports."""
    sys.path.insert(0, str(dev_root))
    try:
        from lib import path_config
        from lib.datasets import get_dataset, get_collect_fn
        from lib.alphabet import strLabelConverter
        from networks.utils import rescale_images, ctc_greedy_decoder
        from networks.module import Recognizer, WriterIdentifier, StyleBackbone
        from metric.val_metrics import (
            InceptionV3,
            ImageListDataset,
            batch_tensor_to_pil_list,
            calculate_activation_statistics,
            get_activations,
            calculate_frechet_distance,
            calculate_inception_score,
            polynomial_mmd_averages,
            compute_real_embeddings,
            ClipEmbeddingModel,
            HWDScore,
        )
        from metric.cmmd import preprocess_images_gpu
        from metric.hwd import ProcessedDataset
        from metric.mssim_psnr import calculate_mssim_psnr

        path_config.data_roots["iam"] = str(dataset_root.resolve()) + os.sep

        api = SimpleNamespace(
            get_dataset=get_dataset,
            get_collect_fn=get_collect_fn,
            converter=strLabelConverter("all"),
            rescale_images=rescale_images,
            ctc_greedy_decoder=ctc_greedy_decoder,
            Recognizer=Recognizer,
            WriterIdentifier=WriterIdentifier,
            StyleBackbone=StyleBackbone,
            InceptionV3=InceptionV3,
            ImageListDataset=ImageListDataset,
            ProcessedDataset=ProcessedDataset,
            batch_tensor_to_pil_list=batch_tensor_to_pil_list,
            calculate_activation_statistics=calculate_activation_statistics,
            get_activations=get_activations,
            calculate_frechet_distance=calculate_frechet_distance,
            calculate_inception_score=calculate_inception_score,
            polynomial_mmd_averages=polynomial_mmd_averages,
            compute_real_embeddings=compute_real_embeddings,
            preprocess_images_gpu=preprocess_images_gpu,
            ClipEmbeddingModel=ClipEmbeddingModel,
            HWDScore=HWDScore,
            calculate_mssim_psnr=calculate_mssim_psnr,
        )
    finally:
        sys.path.pop(0)

    # Baseline projects also use top-level packages named ``lib`` and
    # ``networks``. The DEV functions/classes retained above keep their module
    # globals alive, while removing these names lets the baseline import its own
    # implementation without cross-repository module contamination.
    purge_packages("lib", "networks")
    return api


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return Munch.fromDict(yaml.safe_load(handle))


def checkpoint_signature(baseline: str, checkpoint: dict):
    generator_keys = list(checkpoint.get("Generator", {}))
    has_spiral = any("star_spiral" in key for key in generator_keys)
    has_wave = any("wave_attn" in key for key in generator_keys)
    if baseline == "spis" and not has_spiral:
        raise RuntimeError(
            "The selected SpiS-GAN checkpoint has no StarSpiral generator keys. "
            "It is probably an FW-GAN checkpoint placed in the wrong folder."
        )
    if baseline == "fw" and not has_wave:
        raise RuntimeError(
            "The selected FW-GAN checkpoint has no wave_attn generator keys. "
            "It does not match the FW-GAN code in --baseline-root."
        )
    return "StarSpiral" if has_spiral else "WaveMLP"


def import_baseline_models(baseline: str, baseline_root: Path):
    sys.path.insert(0, str(baseline_root))
    importlib.invalidate_caches()
    try:
        try:
            baseline_networks = importlib.import_module("networks.BigGAN_networks")
        except ModuleNotFoundError as exc:
            if exc.name == "timm":
                raise ModuleNotFoundError(
                    "SpiS-GAN requires timm. Install the pinned notebook dependency "
                    "with: pip install timm==1.0.19"
                ) from exc
            raise
        module_name = "networks.module32" if baseline == "spis" else "networks.module"
        baseline_module = importlib.import_module(module_name)
        baseline_alphabet = importlib.import_module("lib.alphabet")
        return SimpleNamespace(
            Generator=baseline_networks.Generator,
            StyleEncoder=baseline_module.StyleEncoder,
            SharedBackbone=baseline_module.SharedBackbone,
            alphabet=baseline_alphabet.Alphabets["all"],
        )
    finally:
        # Keep the path in sys.path while these imported modules are used. The
        # benchmark runs one baseline per process, so there is no second import
        # collision to resolve here.
        pass


def load_baseline(args, baseline_api, cfg, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signature = checkpoint_signature(args.baseline, checkpoint)

    if int(cfg.GenModel.resolution) != 32 or int(cfg.img_height) != 32:
        raise ValueError(
            "This adapter is deliberately for native x32 checkpoints; the selected "
            f"config declares resolution={cfg.GenModel.resolution}, img_height={cfg.img_height}."
        )

    generator = baseline_api.Generator(**cfg.GenModel)
    encoder = baseline_api.StyleEncoder(**cfg.EncModel)
    backbone = baseline_api.SharedBackbone(**cfg.SharedBackbone)
    generator.load_state_dict(checkpoint["Generator"], strict=True)
    encoder.load_state_dict(checkpoint["StyleEncoder"], strict=True)
    backbone.load_state_dict(checkpoint["SharedBackbone"], strict=True)
    generator.to(device).eval()
    encoder.to(device).eval()
    backbone.to(device).eval()

    metadata = {
        "checkpoint_epoch": checkpoint.get("Epoch"),
        "checkpoint_reported_fid": scalar(checkpoint.get("FID")),
        "checkpoint_reported_kid": scalar(checkpoint.get("KID")),
        "generator_family": signature,
    }
    return generator, encoder, backbone, metadata


def encode_for_baseline(texts, alphabet: str, device):
    mapping = {char: index for index, char in enumerate(alphabet)}
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.int32, device=device)
    labels = torch.zeros(
        len(texts), int(lengths.max().item()), dtype=torch.long, device=device
    )
    for row, text in enumerate(texts):
        try:
            values = [mapping[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"Character {exc.args[0]!r} is unsupported by the baseline alphabet") from exc
        labels[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    return labels, lengths


def generated_cache_signature(args, checkpoint_path: Path, dataset_path: Path):
    checkpoint_stat = checkpoint_path.stat()
    dataset_stat = dataset_path.stat()
    return {
        "cache_version": CACHE_VERSION,
        "baseline": args.baseline,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "dataset": str(dataset_path),
        "dataset_size": dataset_stat.st_size,
        "dataset_mtime_ns": dataset_stat.st_mtime_ns,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "seed": args.seed,
        "source_height": 32,
        "evaluation_height": 32 if args.native_x32 else 64,
        "native_x32": bool(args.native_x32),
    }


def load_generated_cache(cache_root: Path, signature: dict):
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("signature") != signature:
            return None
        cache = DiskBatchCache(cache_root, manifest)
        if not cache.paths or any(not path.is_file() for path in cache.paths):
            return None
        print(f"Reusing complete generated-image cache: {cache_root}")
        return cache
    except (OSError, ValueError, KeyError, TypeError):
        return None


def prepare_generated_cache(cache_root: Path):
    """Remove only this runner's incomplete/stale cache files."""
    cache_root.mkdir(parents=True, exist_ok=True)
    for path in cache_root.glob("shard_*.pt"):
        path.unlink()
    manifest_path = cache_root / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def save_cache_shard(cache_root: Path, shard_index: int, batches):
    name = f"shard_{shard_index:05d}.pt"
    path = cache_root / name
    temporary = cache_root / f".{name}.partial"
    torch.save(batches, temporary)
    os.replace(temporary, path)
    return name


@torch.inference_mode()
def generate_x64_batches(
    loader,
    generator,
    encoder,
    backbone,
    cfg,
    baseline_alphabet,
    dev_api,
    device,
    cache_root: Path,
    cache_signature: dict,
):
    cached = load_generated_cache(cache_root, cache_signature)
    if cached is not None:
        return cached

    prepare_generated_cache(cache_root)
    shard = []
    shard_names = []
    num_batches = 0
    num_images = 0
    noise_dim = int(cfg.GenModel.style_dim) - int(cfg.EncModel.style_dim)
    if noise_dim < 0:
        raise ValueError("GenModel.style_dim must be >= EncModel.style_dim")

    for batch in tqdm(loader, total=len(loader), desc="Generate native x32 -> evaluate x64"):
        texts = dev_api.converter.decode(batch["lbs"], batch["lb_lens"])
        baseline_lbs, baseline_lb_lens = encode_for_baseline(
            texts, baseline_alphabet, device
        )

        style_x64 = batch["style_imgs"].to(device, non_blocking=True)
        style_lens_x64 = batch["style_img_lens"].to(device, non_blocking=True)
        style_x32 = F.interpolate(
            style_x64,
            size=(32, max(1, style_x64.size(-1) // 2)),
            mode="bilinear",
            align_corners=False,
        )
        style_lens_x32 = torch.clamp(
            torch.div(style_lens_x64, 2, rounding_mode="trunc"), min=1
        )

        # The released x32 checkpoints use white background / dark ink, while
        # DEV IAM x64 tensors use black background / light ink. Bridge that
        # representation at the baseline boundary and preserve -1 padding.
        columns = torch.arange(style_x32.size(-1), device=device)
        valid_style = columns.view(1, 1, 1, -1) < style_lens_x32.view(-1, 1, 1, 1)
        style_x32 = torch.where(valid_style, -style_x32, -torch.ones_like(style_x32))

        encoded_style = encoder(style_x32, style_lens_x32, backbone)
        noise = torch.randn(
            style_x32.size(0), noise_dim, dtype=encoded_style.dtype, device=device
        )
        latent = torch.cat([noise, encoded_style], dim=-1)
        fake_x32 = generator(latent, baseline_lbs.long(), baseline_lb_lens.long())
        if fake_x32.size(-2) != 32:
            raise RuntimeError(f"Expected native x32 output, got {tuple(fake_x32.shape)}")

        # Convert output back to DEV black-background / light-ink convention.
        fake_x32 = -fake_x32

        fake_x64 = F.interpolate(
            fake_x32, scale_factor=(2, 2), mode="bilinear", align_corners=False
        )
        canonical_lens_x64 = batch["lb_lens"].to(device) * 32
        org_x64, org_lens_x64 = dev_api.rescale_images(
            fake_x64,
            canonical_lens_x64,
            batch["org_img_lens"],
        )

        fake_batch = {
            "org_imgs": org_x64,
            "org_img_lens": org_lens_x64,
            "style_imgs": fake_x64,
            "style_img_lens": canonical_lens_x64,
            "lbs": batch["lbs"],
            "lb_lens": batch["lb_lens"],
            "wids": batch["wids"],
        }
        cpu_batch = {}
        for key, value in fake_batch.items():
            if not isinstance(value, torch.Tensor):
                cpu_batch[key] = value
            elif key in ("org_imgs", "style_imgs"):
                cpu_batch[key] = (
                    value.detach().cpu().clamp(-1.0, 1.0) * 127.0
                ).round().to(torch.int8)
            else:
                cpu_batch[key] = value.detach().cpu()
        shard.append(cpu_batch)
        num_batches += 1
        num_images += int(cpu_batch["org_imgs"].size(0))
        if len(shard) >= CACHE_SHARD_BATCHES:
            shard_names.append(save_cache_shard(cache_root, len(shard_names), shard))
            shard = []
    if shard:
        shard_names.append(save_cache_shard(cache_root, len(shard_names), shard))

    manifest = {
        "signature": cache_signature,
        "num_batches": num_batches,
        "num_images": num_images,
        "shards": shard_names,
    }
    manifest_path = cache_root / "manifest.json"
    temporary_manifest = cache_root / ".manifest.json.partial"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary_manifest, manifest_path)
    return DiskBatchCache(cache_root, manifest)


@torch.inference_mode()
def generate_native_x32_batches(
    loader,
    generator,
    encoder,
    backbone,
    cfg,
    baseline_alphabet,
    dev_api,
    device,
    cache_root: Path,
    cache_signature: dict,
):
    """Generate/cache native x32 words without polarity or size conversion."""
    cached = load_generated_cache(cache_root, cache_signature)
    if cached is not None:
        return cached

    prepare_generated_cache(cache_root)
    shard, shard_names = [], []
    num_batches = num_images = 0
    noise_dim = int(cfg.GenModel.style_dim) - int(cfg.EncModel.style_dim)
    if noise_dim < 0:
        raise ValueError("GenModel.style_dim must be >= EncModel.style_dim")

    for batch in tqdm(loader, total=len(loader), desc="Generate/evaluate native x32"):
        texts = dev_api.converter.decode(batch["lbs"], batch["lb_lens"])
        baseline_lbs, baseline_lb_lens = encode_for_baseline(
            texts, baseline_alphabet, device
        )
        style_imgs = batch["style_imgs"].to(device, non_blocking=True)
        style_lens = batch["style_img_lens"].to(device, non_blocking=True)
        encoded_style = encoder(style_imgs, style_lens, backbone)
        noise = torch.randn(
            style_imgs.size(0), noise_dim,
            dtype=encoded_style.dtype, device=device,
        )
        latent = torch.cat([noise, encoded_style], dim=-1)
        fake_imgs = generator(
            latent, baseline_lbs.long(), baseline_lb_lens.long()
        )
        if fake_imgs.size(-2) != 32:
            raise RuntimeError(
                f"Expected native x32 output, got {tuple(fake_imgs.shape)}"
            )

        fake_lens = torch.clamp(
            baseline_lb_lens * 16, max=fake_imgs.size(-1)
        )
        columns = torch.arange(fake_imgs.size(-1), device=device)
        valid = columns.view(1, 1, 1, -1) < fake_lens.view(-1, 1, 1, 1)
        fake_imgs = torch.where(valid, fake_imgs, -torch.ones_like(fake_imgs))
        fake_batch = {
            "org_imgs": fake_imgs,
            "org_img_lens": fake_lens,
            "style_imgs": fake_imgs,
            "style_img_lens": fake_lens,
            "lbs": batch["lbs"],
            "lb_lens": batch["lb_lens"],
            "wids": batch["wids"],
        }

        cpu_batch = {}
        for key, value in fake_batch.items():
            if key in ("org_imgs", "style_imgs"):
                cpu_batch[key] = (
                    value.detach().cpu().clamp(-1.0, 1.0) * 127.0
                ).round().to(torch.int8)
            else:
                cpu_batch[key] = value.detach().cpu()
        shard.append(cpu_batch)
        num_batches += 1
        num_images += int(cpu_batch["org_imgs"].size(0))
        if len(shard) >= CACHE_SHARD_BATCHES:
            shard_names.append(
                save_cache_shard(cache_root, len(shard_names), shard)
            )
            shard = []
    if shard:
        shard_names.append(
            save_cache_shard(cache_root, len(shard_names), shard)
        )

    manifest = {
        "signature": cache_signature,
        "num_batches": num_batches,
        "num_images": num_images,
        "shards": shard_names,
    }
    manifest_path = cache_root / "manifest.json"
    temporary_manifest = cache_root / ".manifest.json.partial"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary_manifest, manifest_path)
    return DiskBatchCache(cache_root, manifest)


def fresh_batches(cached):
    for batch in cached.iter_raw():
        restored = {}
        for key, value in batch.items():
            if key in ("org_imgs", "style_imgs"):
                restored[key] = value.to(torch.float32) / 127.0
            else:
                restored[key] = value
        yield restored


def cached_fake_inception_statistics(
    dev_api,
    cached,
    inception,
    dims,
    device,
    crop,
    eval_is,
    cache_dir: Path,
    cache_tag: str,
):
    """Extract fake Inception activations in durable, resumable cache shards."""
    signature = json.dumps(
        cached.manifest.get("signature", {}), sort_keys=True
    ).encode("utf-8")
    signature_key = hashlib.sha1(signature).hexdigest()[:12]
    shard_dir = (
        cache_dir
        / f"{cache_tag}_{cached.root.name}_{signature_key}_fake_inception"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    pause_seconds = max(
        0.0, float(os.environ.get("HTG_INCEPTION_SHARD_PAUSE", "0.5"))
    )

    for index, generated_path in enumerate(
        tqdm(cached.paths, desc="Fake Inception cache shards")
    ):
        activation_path = shard_dir / f"activations_{index:05d}.npz"
        if activation_path.exists():
            continue

        raw_batches = torch.load(
            generated_path, map_location="cpu", weights_only=True
        )

        def decompressed():
            for batch in raw_batches:
                restored = {}
                for key, value in batch.items():
                    if key in ("org_imgs", "style_imgs"):
                        restored[key] = value.to(torch.float32) / 127.0
                    else:
                        restored[key] = value
                yield restored

        activations, logits = dev_api.get_activations(
            decompressed(),
            len(raw_batches),
            inception,
            dims,
            device,
            crop=crop,
            eval_is=eval_is,
        )
        stored_logits = (
            logits if logits is not None else np.empty((0,), dtype=np.float32)
        )
        temporary = shard_dir / f".activations_{index:05d}.partial"
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, act=activations, logits=stored_logits
            )
        os.replace(temporary, activation_path)
        del raw_batches, activations, logits, stored_logits
        if pause_seconds:
            time.sleep(pause_seconds)

    activation_parts, logit_parts = [], []
    for index in range(len(cached.paths)):
        activation_path = shard_dir / f"activations_{index:05d}.npz"
        with np.load(activation_path, allow_pickle=False) as payload:
            activation_parts.append(payload["act"])
            if eval_is:
                logit_parts.append(payload["logits"])
    activations = np.concatenate(activation_parts, axis=0)
    logits = np.concatenate(logit_parts, axis=0) if eval_is else None
    mean = np.mean(activations, axis=0)
    covariance = np.cov(activations, rowvar=False)
    return activations, mean, covariance, logits


def real_cache_tag(dataset, max_batches, batch_size):
    sample_count = len(dataset)
    if max_batches and max_batches > 0:
        sample_count = min(sample_count, max_batches * batch_size)
    return f"iam_test_words64_n{sample_count}"


def metric_config(requested):
    return Munch(
        validate_fid="fid" in requested,
        validate_kid="kid" in requested,
        validate_is_gen="is_gen" in requested,
        validate_is_org="is_org" in requested,
        dims=2048,
        mmd_degree=3,
        mmd_gamma=None,
        mmd_coef0=1.0,
        mmd_subsets=50,
        mmd_subset_size=1000,
        mmd_var=True,
    )


def low_memory_fid(dev_api, mu1, sigma1, mu2, sigma2, device):
    """FID with the standard formula, avoiding SciPy's large CPU sqrtm peak."""
    if torch.device(device).type != "cuda":
        return dev_api.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    report_resources("FID CPU eigensolver start", device, reset_peak=True)
    with torch.inference_mode():
        mu1_t = torch.as_tensor(mu1, dtype=torch.float64, device="cpu")
        mu2_t = torch.as_tensor(mu2, dtype=torch.float64, device="cpu")
        sigma1_t = torch.as_tensor(sigma1, dtype=torch.float64, device="cpu")
        sigma2_t = torch.as_tensor(sigma2, dtype=torch.float64, device="cpu")
        sigma1_t = 0.5 * (sigma1_t + sigma1_t.T)
        sigma2_t = 0.5 * (sigma2_t + sigma2_t.T)

        eigenvalues1, eigenvectors1 = torch.linalg.eigh(sigma1_t)
        sqrt_sigma1 = (
            eigenvectors1 * eigenvalues1.clamp_min(0).sqrt().unsqueeze(0)
        ) @ eigenvectors1.T
        middle = sqrt_sigma1 @ sigma2_t @ sqrt_sigma1
        middle = 0.5 * (middle + middle.T)
        trace_covmean = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum()
        difference = mu1_t - mu2_t
        score = (
            difference.dot(difference)
            + torch.trace(sigma1_t)
            + torch.trace(sigma2_t)
            - 2.0 * trace_covmean
        ).item()
    del mu1_t, mu2_t, sigma1_t, sigma2_t, eigenvalues1, eigenvectors1
    del sqrt_sigma1, middle, trace_covmean, difference
    clear_cuda()
    report_resources("FID CPU eigensolver done", device)
    return score


def distribution_metrics(
    dev_api, loader, cached, device, cache_dir: Path, cache_tag: str, requested, seed
):
    cfg = metric_config(requested)
    report_resources("Inception features start", device, reset_peak=True)
    block_idx = dev_api.InceptionV3.BLOCK_INDEX_BY_DIM[cfg.dims]
    inception = dev_api.InceptionV3([block_idx]).to(device).eval()
    cache_path = cache_dir / f"{cache_tag}_inception_stats.npz"
    eval_is = bool(requested.intersection({"is_gen", "is_org"}))
    real_stats = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            real_stats = tuple(
                payload[name] for name in ("act", "mu", "sigma", "logits")
            )
        if eval_is and real_stats[3].size == 0:
            real_stats = None
        else:
            print(f"Loaded shared real Inception statistics: {cache_path}")
    if real_stats is None:
        print("Computing shared real Inception statistics...")
        real_stats = dev_api.calculate_activation_statistics(
            loader,
            len(loader),
            inception,
            cfg.dims,
            device,
            crop=True,
            eval_is=eval_is,
        )
        stored_logits = (
            real_stats[3]
            if real_stats[3] is not None
            else np.empty((0,), dtype=np.float32)
        )
        np.savez_compressed(
            cache_path,
            act=real_stats[0],
            mu=real_stats[1],
            sigma=real_stats[2],
            logits=stored_logits,
        )
    fake_stats = cached_fake_inception_statistics(
        dev_api,
        cached,
        inception,
        cfg.dims,
        device,
        crop=True,
        eval_is=eval_is,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
    )
    del inception
    clear_cuda()
    report_resources("Inception features done", device)

    act1, mu1, sigma1, logits1 = real_stats
    act2, mu2, sigma2, logits2 = fake_stats
    result = {}
    if "fid" in requested:
        result["fid"] = low_memory_fid(
            dev_api, mu1, sigma1, mu2, sigma2, device
        )
    if "is_gen" in requested:
        result["is_gen"] = dev_api.calculate_inception_score(logits2)
    if "is_org" in requested:
        result["is_org"] = dev_api.calculate_inception_score(logits1)
    if "kid" in requested:
        # A dedicated seed makes KID identical after a disk-cache resume.
        np.random.seed(seed + 991)
        ret = dev_api.polynomial_mmd_averages(
            act1,
            act2,
            degree=cfg.mmd_degree,
            gamma=cfg.mmd_gamma,
            coef0=cfg.mmd_coef0,
            ret_var=cfg.mmd_var,
            n_subsets=cfg.mmd_subsets,
            subset_size=cfg.mmd_subset_size,
        )
        mmd2s = ret[0] if cfg.mmd_var else ret
        result["kid"] = mmd2s.mean() * 100
    del real_stats, fake_stats, act1, mu1, sigma1, logits1
    del act2, mu2, sigma2, logits2
    clear_cuda()
    report_resources("FID/KID/IS done", device)
    return result


def hwd_metric(dev_api, loader, cached, device, cache_dir: Path, cache_tag: str):
    report_resources("HWD start", device, reset_peak=True)
    scorer = dev_api.HWDScore(batchsize=64).to(device)
    cache_path = cache_dir / f"{cache_tag}_hwd_real.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        real_features = dev_api.ProcessedDataset(
            payload["ids"], payload["labels"], payload["features"]
        )
        print(f"Loaded shared real HWD features: {cache_path}")
    else:
        real_images, real_authors = [], []
        for batch in tqdm(loader, total=len(loader), desc="HWD real images"):
            real_images.extend(dev_api.batch_tensor_to_pil_list(
                batch["org_imgs"], batch["org_img_lens"]
            ))
            real_authors.extend(str(int(wid)) for wid in batch["wids"])
        real_dataset = dev_api.ImageListDataset(real_images, real_authors)
        real_features = scorer.digest(real_dataset)
        del real_dataset, real_images, real_authors
        torch.save(
            {
                "ids": real_features.ids.cpu(),
                "labels": real_features.labels,
                "features": real_features.features.cpu(),
            },
            cache_path,
        )

    fake_images, fake_authors = [], []
    for batch in tqdm(fresh_batches(cached), total=len(cached), desc="HWD fake images"):
        fake_images.extend(dev_api.batch_tensor_to_pil_list(
            batch["org_imgs"], batch["org_img_lens"]
        ))
        fake_authors.extend(str(int(wid)) for wid in batch["wids"])
    fake_dataset = dev_api.ImageListDataset(fake_images, fake_authors)
    fake_features = scorer.digest(fake_dataset)
    del fake_dataset, fake_images, fake_authors
    score = scorer.distance(fake_features, real_features)
    del scorer, real_features, fake_features
    clear_cuda()
    report_resources("HWD done", device)
    return score


def compute_fake_embeddings(dev_api, cached, embedding_model, device):
    size = embedding_model.input_image_size
    embeddings = []
    for batch in tqdm(fresh_batches(cached), total=len(cached), desc="CMMD fake embeddings"):
        images = batch["org_imgs"].to(device, non_blocking=True)
        batch_images = dev_api.preprocess_images_gpu(
            images, batch["org_img_lens"], size, device
        )
        embeddings.append(embedding_model.embed(batch_images).numpy())
    return np.concatenate(embeddings, axis=0).astype("float32")


def rbf_kernel_mean_chunked(x, y, device, sigma=10.0, block_size=512):
    """Exact mean RBF kernel without allocating the full N-by-N matrix."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    total = 0.0
    count = int(x.size(0)) * int(y.size(0))
    gamma = 1.0 / (2.0 * sigma ** 2)
    with torch.inference_mode():
        for row in range(0, x.size(0), block_size):
            x_block = x[row:row + block_size].to(device, non_blocking=True)
            x_norm = (x_block * x_block).sum(dim=1, keepdim=True)
            for column in range(0, y.size(0), block_size):
                y_block = y[column:column + block_size].to(device, non_blocking=True)
                y_norm = (y_block * y_block).sum(dim=1).unsqueeze(0)
                distance = (x_norm + y_norm - 2.0 * (x_block @ y_block.T)).clamp_min_(0)
                total += torch.exp(-gamma * distance).sum(dtype=torch.float64).item()
                del y_block, y_norm, distance
            del x_block, x_norm
    return total / max(count, 1)


def chunked_cmmd(real_embeddings, fake_embeddings, device):
    report_resources("CMMD chunked RBF start", device, reset_peak=True)
    mean_xx = rbf_kernel_mean_chunked(real_embeddings, real_embeddings, device)
    mean_yy = rbf_kernel_mean_chunked(fake_embeddings, fake_embeddings, device)
    mean_xy = rbf_kernel_mean_chunked(real_embeddings, fake_embeddings, device)
    result = 1000.0 * (mean_xx + mean_yy - 2.0 * mean_xy)
    clear_cuda()
    report_resources("CMMD chunked RBF done", device)
    return result


def cmmd_metric(dev_api, loader, cached, device, cache_dir: Path, cache_tag: str):
    report_resources("CMMD CLIP features start", device, reset_peak=True)
    embedding_model = dev_api.ClipEmbeddingModel(device)
    cache_path = cache_dir / f"{cache_tag}_cmmd_real.npy"
    if cache_path.exists():
        real_embeddings = np.load(cache_path)
        print(f"Loaded shared real CMMD embeddings: {cache_path}")
    else:
        real_embeddings = dev_api.compute_real_embeddings(
            loader, embedding_model, n_batches=len(loader), device=device
        )
        np.save(cache_path, real_embeddings)
    fake_embeddings = compute_fake_embeddings(
        dev_api, cached, embedding_model, device
    )
    del embedding_model
    clear_cuda()
    report_resources("CMMD CLIP features done", device)
    score = chunked_cmmd(real_embeddings, fake_embeddings, device)
    del real_embeddings, fake_embeddings
    return score


def ocr_metrics(dev_api, cached, dev_root: Path, device):
    report_resources("OCR CER/WER start", device, reset_peak=True)
    recognizer = dev_api.Recognizer(
        n_class=80,
        resolution=16,
        max_dim=256,
        in_channel=1,
        norm="bn",
        init="none",
        dropout=0.0,
        rnn_depth=2,
        bidirectional=True,
    ).to(device)
    checkpoint = torch.load(
        dev_root / "pretrained" / "ocr_iam_new.pth",
        map_location="cpu",
        weights_only=False,
    )
    recognizer.load_state_dict(checkpoint["Recognizer"], strict=False)
    recognizer.eval()

    char_trans = total_chars = word_trans = total_words = 0
    with torch.inference_mode():
        for batch in tqdm(fresh_batches(cached), total=len(cached), desc="DEV OCR CER/WER"):
            imgs = batch["style_imgs"].to(device, non_blocking=True)
            # Some native x32 generators emit one character cell less than the
            # nominal len*32 canvas. Packed RNN lengths must not exceed the
            # recognizer features produced from the actual padded image width.
            img_lens_cpu = torch.clamp(batch["style_img_lens"], max=imgs.size(-1))
            img_lens = img_lens_cpu.to(device, non_blocking=True)
            logits = recognizer(imgs, img_lens).softmax(dim=2).cpu().numpy()
            predictions = []
            for logit, img_len in zip(logits, img_lens_cpu.numpy()):
                labels = dev_api.ctc_greedy_decoder(
                    logit[: int(img_len) // recognizer.len_scale]
                )
                predictions.append(dev_api.converter.decode(labels))
            targets = dev_api.converter.decode(batch["lbs"], batch["lb_lens"])
            from distance import levenshtein
            for prediction, target in zip(predictions, targets):
                edits = levenshtein(prediction, target)
                char_trans += edits
                total_chars += len(target)
                total_words += 1
                word_trans += int(edits > 0)
    del recognizer, checkpoint
    clear_cuda()
    report_resources("OCR CER/WER done", device)
    return (
        char_trans / max(total_chars, 1),
        word_trans / max(total_words, 1),
    )


def writer_error_metric(dev_api, loader, cached, dev_root: Path, device):
    report_resources("WIER start", device, reset_peak=True)
    checkpoint = torch.load(
        dev_root / "pretrained" / "wid_iam_test.pth",
        map_location="cpu",
        weights_only=False,
    )
    writer = dev_api.WriterIdentifier(n_writer=128, in_dim=256, init="none").to(device)
    backbone = dev_api.StyleBackbone(
        resolution=16,
        max_dim=256,
        in_channel=1,
        init="N02",
        dropout=0.0,
        norm="bn",
    ).to(device)
    writer.load_state_dict(checkpoint["WriterIdentifier"], strict=False)
    backbone.load_state_dict(checkpoint["StyleBackbone"], strict=False)
    writer.eval()
    backbone.eval()

    correct = total = 0
    with torch.inference_mode():
        pairs = zip(loader, fresh_batches(cached))
        for real_batch, fake_batch in tqdm(pairs, total=len(loader), desc="DEV WIER"):
            real_logits = writer(
                real_batch["style_imgs"].to(device, non_blocking=True),
                real_batch["style_img_lens"].to(device, non_blocking=True),
                backbone,
            )
            fake_logits = writer(
                fake_batch["style_imgs"].to(device, non_blocking=True),
                fake_batch["style_img_lens"].to(device, non_blocking=True),
                backbone,
            )
            real_pred = real_logits.argmax(dim=1)
            fake_pred = fake_logits.argmax(dim=1)
            correct += real_pred.eq(fake_pred).sum().item()
            total += real_pred.numel()
    del writer, backbone, checkpoint
    clear_cuda()
    report_resources("WIER done", device)
    return 1.0 - correct / max(total, 1)


def output_paths(output_root: Path, baseline: str):
    output_root.mkdir(parents=True, exist_ok=True)
    stem = "spis_gan" if baseline == "spis" else "fw_gan"
    return output_root / f"{stem}_dev_all_metrics.csv", output_root / f"{stem}_dev_all_metrics.json"


def write_outputs(csv_path: Path, json_path: Path, row: dict):
    metadata_columns = (
        "model", "checkpoint", "checkpoint_epoch", "generator_family",
        "source_height", "evaluation_height", "num_images", "seed", "protocol",
    )
    fieldnames = metadata_columns + METRIC_COLUMNS
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in fieldnames})
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, ensure_ascii=False)


def restore_existing_metrics(csv_path: Path, row: dict):
    """Keep completed metric groups when resuming the same validation run."""
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        existing = next(csv.DictReader(handle), None)
    if not existing:
        return
    identity = ("model", "checkpoint", "num_images", "seed", "protocol")
    if any(str(existing.get(key, "")) != str(row.get(key, "")) for key in identity):
        return
    restored = []
    for name in METRIC_COLUMNS:
        value = existing.get(name, "")
        if value not in (None, ""):
            row[name] = float(value)
            restored.append(name)
    if restored:
        print(f"Preserved completed metrics from prior run: {restored}")


def main():
    args = parse_args()
    baseline_root = Path(args.baseline_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    dev_root = Path(args.dev_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()

    if args.native_x32:
        if not args.dataset_file:
            raise ValueError("--native-x32 requires --dataset-file")
        dataset_path = Path(args.dataset_file).resolve()
        dataset_description = "native x32 IAM test set"
    else:
        dataset_path = dataset_root / "testset_words64_OrgSz.hdf5"
        dataset_description = "x64 IAM test set"

    for path, description in (
        (baseline_root, "baseline repository"),
        (checkpoint_path, "baseline checkpoint"),
        (config_path, "baseline config"),
        (dev_root / "metric" / "val_metrics.py", "DEV metric implementation"),
        (dataset_path, dataset_description),
        (dev_root / "pretrained" / "ocr_iam_new.pth", "DEV OCR evaluator"),
        (dev_root / "pretrained" / "wid_iam_test.pth", "DEV WIER evaluator"),
    ):
        fail_missing(path, description)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    seed_everything(args.seed)

    if args.metrics.strip().lower() == "all":
        requested = set(METRIC_COLUMNS)
    elif args.metrics.strip().lower() == "none":
        requested = set()
    else:
        requested = {item.strip().lower() for item in args.metrics.split(",") if item.strip()}
        unknown = requested.difference(METRIC_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown metrics: {sorted(unknown)}")

    cache_dir = output_root / "shared_real_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output_paths(output_root, args.baseline)
    dev_api = import_dev_api(dev_root, dataset_root)
    cfg = load_yaml(config_path)
    baseline_api = import_baseline_models(args.baseline, baseline_root)
    generator, encoder, backbone, checkpoint_meta = load_baseline(
        args, baseline_api, cfg, device
    )

    if args.native_x32:
        dataset = NativeX32Hdf5Dataset(dataset_path, dev_api.converter)
        collate_fn = collect_native_x32
    else:
        dataset = dev_api.get_dataset(
            "iam_word_org", "test", process_style=True
        )
        collate_fn = dev_api.get_collect_fn(sort_input=False)
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    limited_loader = LimitedLoader(loader, args.max_batches)

    evaluation_name = (
        "native x32 IAM/DEV evaluation"
        if args.native_x32
        else "native x32 checkpoint -> x64 IAM/DEV evaluation"
    )
    print(
        f"\n[{args.baseline.upper()}] {evaluation_name}\n"
        f"checkpoint: {checkpoint_path}\n"
        f"batches: {len(limited_loader)} / {len(loader)}; batch_size={args.batch_size}\n"
        f"metrics: {sorted(requested) if requested else 'generation smoke test'}\n"
    )
    print(
        "Device map (full precision; no FP16):\n"
        f"  baseline G/E/backbone: {device}\n"
        f"  Inception FID/KID/IS features: {device}\n"
        f"  FID covariance eigensolver: CPU (safe; features remain on {device})\n"
        f"  HWD VGG16 features: {device}; final distance: CPU\n"
        f"  CMMD CLIP features + chunked RBF: {device}\n"
        f"  OCR/WIER networks: {device}; PSNR/MS-SSIM: CPU\n",
        flush=True,
    )
    report_resources("baseline generation start", device, reset_peak=True)
    generated_cache_root = output_root / "generated_cache" / args.baseline
    cache_signature = generated_cache_signature(
        args, checkpoint_path, dataset_path
    )
    generate_batches = (
        generate_native_x32_batches if args.native_x32 else generate_x64_batches
    )
    cached = generate_batches(
        limited_loader,
        generator,
        encoder,
        backbone,
        cfg,
        baseline_api.alphabet,
        dev_api,
        device,
        generated_cache_root,
        cache_signature,
    )
    if not cached:
        raise RuntimeError("The evaluation loader produced no batches")
    first = cached.first()
    print(
        "Adapter smoke check: "
        f"style={tuple(first['style_imgs'].shape)}, "
        f"org={tuple(first['org_imgs'].shape)}, "
        f"height={first['style_imgs'].shape[-2]}"
    )

    num_images = cached.num_images
    row = {
        "model": "SpiS-GAN" if args.baseline == "spis" else "FW-GAN",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_meta["checkpoint_epoch"],
        "generator_family": checkpoint_meta["generator_family"],
        "source_height": 32,
        "evaluation_height": 32 if args.native_x32 else 64,
        "num_images": num_images,
        "seed": args.seed,
        "protocol": (
            "Native x32 style-guided test validation; no polarity or spatial "
            "resize; DEV FID/KID/HWD"
            if args.native_x32
            else "DEV style-guided test validation; polarity bridge; x64 refs "
                 "-> x32 E/G -> bilinear x64 -> DEV metrics"
        ),
    }
    row.update({name: None for name in METRIC_COLUMNS})
    restore_existing_metrics(csv_path, row)
    write_outputs(csv_path, json_path, row)

    # Generation is complete. Metric networks run sequentially after the x32
    # baseline models leave VRAM, which keeps the full suite within an 8 GB GPU.
    del generator, encoder, backbone
    clear_cuda()
    report_resources("baseline generation done/models released", device)
    if args.native_x32:
        sample_count = len(dataset)
        if args.max_batches and args.max_batches > 0:
            sample_count = min(
                sample_count, args.max_batches * args.batch_size
            )
        cache_tag = f"iam_{dataset_path.stem}_native32_n{sample_count}"
    else:
        cache_tag = real_cache_tag(
            dataset, args.max_batches, args.batch_size
        )

    if requested.intersection({"fid", "kid", "is_gen", "is_org"}):
        distribution = distribution_metrics(
            dev_api, limited_loader, cached, device, cache_dir, cache_tag, requested,
            args.seed,
        )
        for name in ("fid", "kid", "is_gen", "is_org"):
            if name in requested and name in distribution:
                row[name] = scalar(distribution[name])
        write_outputs(csv_path, json_path, row)

    if "hwd" in requested:
        row["hwd"] = scalar(hwd_metric(
            dev_api, limited_loader, cached, device, cache_dir, cache_tag
        ))
        write_outputs(csv_path, json_path, row)

    if "cmmd" in requested:
        row["cmmd"] = scalar(cmmd_metric(
            dev_api, limited_loader, cached, device, cache_dir, cache_tag
        ))
        write_outputs(csv_path, json_path, row)

    if requested.intersection({"cer", "wer"}):
        cer, wer = ocr_metrics(dev_api, cached, dev_root, device)
        if "cer" in requested:
            row["cer"] = scalar(cer)
        if "wer" in requested:
            row["wer"] = scalar(wer)
        write_outputs(csv_path, json_path, row)

    if "wier" in requested:
        row["wier"] = scalar(writer_error_metric(
            dev_api, limited_loader, cached, dev_root, device
        ))
        write_outputs(csv_path, json_path, row)

    if requested.intersection({"psnr", "mssim"}):
        report_resources("PSNR/MS-SSIM CPU start", device)
        reconstruction = dev_api.calculate_mssim_psnr(
            limited_loader, fresh_batches(cached)
        )
        if "psnr" in requested:
            row["psnr"] = scalar(reconstruction["psnr"])
        if "mssim" in requested:
            row["mssim"] = scalar(reconstruction["mssim"])
        write_outputs(csv_path, json_path, row)
        report_resources("PSNR/MS-SSIM CPU done", device)

    print("\nCompleted DEV-equivalent baseline validation:")
    for name in METRIC_COLUMNS:
        print(f"  {name:8s}: {row[name]}")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
