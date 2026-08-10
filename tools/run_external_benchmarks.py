#!/usr/bin/env python3
"""Run SpiS-GAN and FW-GAN sequentially with DEV's complete metric suite."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys


METRICS = (
    "fid", "kid", "hwd", "cmmd", "cer", "wer",
    "is_gen", "is_org", "psnr", "mssim", "wier",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--spis-checkpoint", default="")
    parser.add_argument("--fw-checkpoint", default="")
    parser.add_argument("--no-clone", action="store_true")
    return parser.parse_args()


def ensure_repository(path: Path, url: str, clone_missing: bool):
    if path.exists():
        return
    if not clone_missing:
        raise FileNotFoundError(f"Missing baseline repository: {path}")
    print(f"Cloning {url} -> {path}")
    subprocess.run(["git", "clone", url, str(path)], check=True)


def resolve_checkpoint(explicit: str, default_path: Path, expected_name: str):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(default_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.rglob(expected_name))
        if matches:
            print(f"Auto-detected Kaggle checkpoint: {matches[0]}")
            return matches[0].resolve()
    raise FileNotFoundError(
        f"Could not find {expected_name}. Expected {default_path}, or pass an explicit checkpoint argument."
    )


def read_result(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def print_table(rows):
    columns = ("model",) + METRICS
    widths = {
        name: max(len(name), *(len(str(row.get(name, ""))) for row in rows))
        for name in columns
    }
    print("\n" + " | ".join(name.ljust(widths[name]) for name in columns))
    print("-+-".join("-" * widths[name] for name in columns))
    for row in rows:
        print(" | ".join(str(row.get(name, "")).ljust(widths[name]) for name in columns))


def main():
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    dev_root = workspace / "HGGAN_test" / "dev"
    runner = dev_root / "tools" / "benchmark_external_gan.py"
    dataset_root = workspace / "iam"
    output_root = workspace / "baseline_dev_metric_benchmark"
    output_root.mkdir(parents=True, exist_ok=True)

    spis_root = workspace / "SpiS-GAN"
    fw_root = workspace / "FW-GAN"
    ensure_repository(
        spis_root, "https://github.com/DAIR-Group/SpiS-GAN.git", not args.no_clone
    )
    ensure_repository(
        fw_root, "https://github.com/DAIR-Group/FW-GAN.git", not args.no_clone
    )

    spis_checkpoint = resolve_checkpoint(
        args.spis_checkpoint,
        workspace / "chkpoints" / "SpiS-GAN" / "bestIAM.pth",
        "bestIAM.pth",
    )
    fw_checkpoint = resolve_checkpoint(
        args.fw_checkpoint,
        workspace / "chkpoints" / "FW-GAN" / "FW-GAN.pth",
        "FW-GAN.pth",
    )

    specs = (
        {
            "baseline": "spis",
            "root": spis_root,
            "checkpoint": spis_checkpoint,
            "config": spis_root / "configs" / "SpiS_gan_iam_32.yml",
            "result": output_root / "spis_gan_dev_all_metrics.csv",
        },
        {
            "baseline": "fw",
            "root": fw_root,
            "checkpoint": fw_checkpoint,
            "config": fw_root / "configs" / "fw_gan_iam.yml",
            "result": output_root / "fw_gan_dev_all_metrics.csv",
        },
    )

    failures = []
    rows = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for spec in specs:
        command = [
            sys.executable,
            str(runner),
            "--baseline", spec["baseline"],
            "--baseline-root", str(spec["root"]),
            "--checkpoint", str(spec["checkpoint"]),
            "--config", str(spec["config"]),
            "--dev-root", str(dev_root),
            "--dataset-root", str(dataset_root),
            "--output-root", str(output_root),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--max-batches", str(args.max_batches),
            "--seed", str(args.seed),
            "--metrics", args.metrics,
        ]
        print(
            f"\n===== Running {spec['baseline'].upper()} first-party DEV metrics =====",
            flush=True,
        )
        try:
            subprocess.run(command, check=True, env=env)
            rows.append(read_result(spec["result"]))
        except subprocess.CalledProcessError as exc:
            failures.append((spec["baseline"], exc.returncode))
            print(f"{spec['baseline']} failed with exit code {exc.returncode}; continuing to the next baseline.")

    if rows:
        combined_path = output_root / "spis_fw_dev_all_metrics.csv"
        with combined_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print_table(rows)
        print(f"\nCombined CSV: {combined_path}")

    if failures:
        details = ", ".join(f"{name} (exit {code})" for name, code in failures)
        raise RuntimeError(f"One or more baseline validations failed: {details}")


if __name__ == "__main__":
    main()
