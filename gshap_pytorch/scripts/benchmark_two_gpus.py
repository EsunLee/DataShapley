#!/usr/bin/env python3
"""Compare two independent G-Shapley groups on one GPU and two GPUs."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = ("train_indices.npy", "test_indices.npy", "lr_val_indices.npy", "split_meta.json")


def prepare_split(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in SPLIT_FILES:
        shutil.copy2(source / name, destination / name)


def command(args: argparse.Namespace, result_dir: Path, seed: int) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / "run_group.py"),
            "--data-dir", str(args.data_dir), "--results-dir", str(result_dir),
            "--seed", str(seed), "--device", "cuda:0",
            "--iterations", str(args.iterations), "--learning-rate", str(args.learning_rate),
            "--parallel-perms"]


def gpu_environment(device: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    return env


def run_case(args: argparse.Namespace, name: str, devices: tuple[str, str],
             source_split: Path, output: Path, parallel: bool) -> float:
    result_dir = output / name
    prepare_split(source_split, result_dir / "split")
    commands = [command(args, result_dir, seed) for seed in args.seeds]
    started = time.perf_counter()
    if parallel:
        processes = [subprocess.Popen(item, env=gpu_environment(device))
                     for item, device in zip(commands, devices, strict=True)]
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"{name} failed with exit codes {codes}; outputs: {result_dir}")
    else:
        for item, device in zip(commands, devices, strict=True):
            subprocess.run(item, env=gpu_environment(device), check=True)
    elapsed = time.perf_counter() - started
    print(f"{name}: {elapsed:.2f} s", flush=True)
    return elapsed


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(
            f"Python 3.10+ is required by gshap_pytorch (current: {sys.version.split()[0]}). "
            "Activate the same environment used for the completed seed 5922 run."
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True,
                        help="Existing fixed split, e.g. ../results/split")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    parser.add_argument("--seeds", type=int, nargs=2, default=[13764, 14933])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    if args.devices[0] == args.devices[1]:
        parser.error("--devices must name two distinct GPUs")
    if any(re.fullmatch(r"cuda:\d+", item) is None for item in args.devices):
        parser.error("--devices must use physical GPU indices, e.g. cuda:0 cuda:3")
    if args.seeds[0] == args.seeds[1]:
        parser.error("--seeds must be distinct")
    if args.iterations < 1 or args.learning_rate <= 0:
        parser.error("iterations and learning rate must be positive")
    for name in SPLIT_FILES:
        if not (args.split_dir / name).is_file():
            parser.error(f"missing split file: {args.split_dir / name}")

    output = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Benchmark output: {output}", flush=True)
    serial = run_case(args, "serial", (args.devices[0], args.devices[0]),
                      args.split_dir, output, parallel=False)
    dual = run_case(args, "two_gpus", tuple(args.devices),
                    args.split_dir, output, parallel=True)

    differences = {}
    for seed in args.seeds:
        one = np.load(output / "serial" / f"group_seed{seed}" / "group_shap.npy")
        two = np.load(output / "two_gpus" / f"group_seed{seed}" / "group_shap.npy")
        differences[str(seed)] = float(np.max(np.abs(one - two)))
    report = {
        "seeds": args.seeds, "devices": args.devices, "iterations_per_group": args.iterations,
        "serial_seconds": round(serial, 2), "two_gpu_seconds": round(dual, 2),
        "speedup": round(serial / dual, 3),
        "time_reduction_percent": round(100 * (1 - dual / serial), 2),
        "max_abs_shap_difference_by_seed": differences,
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
