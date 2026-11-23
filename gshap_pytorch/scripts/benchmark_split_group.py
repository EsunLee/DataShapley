#!/usr/bin/env python3
"""Benchmark one G-Shapley group on one GPU versus splitting it across two."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gshap.config import DataConfig, ExperimentConfig, RuntimeConfig, TrainConfig
from gshap.data import load_or_create_bundle
from gshap.seeds import derive_iteration_seeds
from gshap.trainer import GShapTrainer


def worker(args: argparse.Namespace) -> None:
    bundle = load_or_create_bundle(args.data_dir, args.split_dir, DataConfig())
    config = ExperimentConfig(
        args.data_dir, args.output_file.parent,
        train=TrainConfig(iterations=args.iterations, learning_rate=args.learning_rate),
        runtime=RuntimeConfig(device="cuda:0", parallel_perms=True),
    )
    trainer = GShapTrainer(bundle, config, args.learning_rate)
    seeds = derive_iteration_seeds(args.seed, args.iterations)
    selected = [int(value) for value in seeds[args.start:args.stop]]
    if args.mode == "stacked":
        marginals, aucs, *_ = trainer._run_permutations_stacked(selected)
        marginal_sum = marginals.sum(axis=0)
    else:
        marginal_sum = np.zeros(len(trainer.train_x), dtype=np.float64)
        aucs = []
        for index, seed in enumerate(selected, start=args.start + 1):
            marginal, auc, *_ = trainer._run_permutation(seed)
            marginal_sum += marginal
            aucs.append(auc)
            print(f"Finished permutation {index}/{args.iterations}", flush=True)
        aucs = np.asarray(aucs, dtype=np.float64)
    np.savez(args.output_file, marginal_sum=marginal_sum, final_aucs=aucs)
    print(f"Finished permutations {args.start + 1}-{args.stop} on visible GPU 0", flush=True)


def child_command(args: argparse.Namespace, output: Path, start: int, stop: int) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()),
            "--worker", "--data-dir", str(args.data_dir),
            "--split-dir", str(args.split_dir), "--output-file", str(output),
            "--seed", str(args.seed), "--iterations", str(args.iterations),
            "--learning-rate", str(args.learning_rate), "--mode", args.mode,
            "--start", str(start), "--stop", str(stop)]


def child_env(device: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    return env


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(f"Python 3.10+ required; current: {sys.version.split()[0]}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:3"])
    parser.add_argument("--seed", type=int, default=5922)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--mode", choices=("stacked", "sequential"), default="stacked",
                        help="Use the optimized stacked path or the original one-permutation path")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.iterations < 2 or args.learning_rate <= 0:
        parser.error("iterations must be at least 2 and learning rate positive")
    if args.worker:
        if args.output_file is None or args.stop is None or not 0 <= args.start < args.stop <= args.iterations:
            parser.error("invalid worker range or output file")
        worker(args)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.devices[0] == args.devices[1] or any(
        re.fullmatch(r"cuda:\d+", device) is None for device in args.devices
    ):
        parser.error("--devices must name two distinct physical GPUs, e.g. cuda:0 cuda:3")
    for name in ("train_indices.npy", "test_indices.npy", "lr_val_indices.npy", "split_meta.json"):
        if not (args.split_dir / name).is_file():
            parser.error(f"missing split file: {args.split_dir / name}")

    output = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Benchmark output: {output}", flush=True)
    single_file = output / "single_gpu.npz"
    started = time.perf_counter()
    subprocess.run(child_command(args, single_file, 0, args.iterations),
                   env=child_env(args.devices[0]), check=True)
    single_seconds = time.perf_counter() - started
    print(f"single_gpu: {single_seconds:.2f} s", flush=True)

    midpoint = args.iterations // 2
    first_file, second_file = output / "two_gpu_first.npz", output / "two_gpu_second.npz"
    started = time.perf_counter()
    processes = [
        subprocess.Popen(child_command(args, first_file, 0, midpoint),
                         env=child_env(args.devices[0])),
        subprocess.Popen(child_command(args, second_file, midpoint, args.iterations),
                         env=child_env(args.devices[1])),
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"Two-GPU run failed with exit codes {codes}; outputs: {output}")
    dual_seconds = time.perf_counter() - started
    print(f"two_gpus: {dual_seconds:.2f} s", flush=True)

    with np.load(single_file) as single, np.load(first_file) as first, np.load(second_file) as second:
        combined_sum = first["marginal_sum"] + second["marginal_sum"]
        max_difference = float(np.max(np.abs(single["marginal_sum"] - combined_sum)))
        combined_aucs = np.concatenate([first["final_aucs"], second["final_aucs"]])
        aucs_equal = bool(np.array_equal(single["final_aucs"], combined_aucs))
        max_auc_difference = float(np.max(np.abs(single["final_aucs"] - combined_aucs)))
    report = {
        "master_seed": args.seed, "iterations": args.iterations, "devices": args.devices,
        "mode": args.mode,
        "single_gpu_seconds": round(single_seconds, 2),
        "two_gpu_seconds": round(dual_seconds, 2),
        "speedup": round(single_seconds / dual_seconds, 3),
        "time_reduction_percent": round(100 * (1 - dual_seconds / single_seconds), 2),
        "max_abs_marginal_sum_difference": max_difference,
        "final_aucs_exactly_equal": aucs_equal,
        "max_abs_final_auc_difference": max_auc_difference,
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
