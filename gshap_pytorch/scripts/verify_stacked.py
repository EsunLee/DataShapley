#!/usr/bin/env python3
"""Verify the stacked parallel-permutation path on a small synthetic dataset.

Runs the same seeds through both the sequential and the stacked runner and
reports whether the marginals match bit-for-bit, per-stream efficiency errors,
and the parallel wall times. Not part of the formal experiment.

Usage:
    PYTHONPATH=gshap_pytorch python gshap_pytorch/scripts/verify_stacked.py [device]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gshap.config import ExperimentConfig, RuntimeConfig, TrainConfig
from gshap.data import DatasetBundle, create_split
from gshap.trainer import GShapTrainer


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(11)
    n = 4_096
    features = rng.normal(size=(n, 512)).astype(np.float32)
    labels = (rng.random(n) < 0.5).astype(np.float32)  # 1-D: trainer contract, cf. load_or_create_bundle
    train, test, lr_val = create_split(labels, test_size=96, lr_val_size=32, seed=0)
    bundle = DatasetBundle(features, labels, train, test, lr_val)
    config = ExperimentConfig(
        data_dir=Path("."), results_dir=Path("."),
        train=TrainConfig(batch_size=128, iterations=1, learning_rate=0.1),
        runtime=RuntimeConfig(device=device, deterministic=True),
    )
    trainer = GShapTrainer(bundle, config, 0.1)
    seeds = [101, 202, 303]
    sequential = np.stack([trainer._run_permutation(seed)[0] for seed in seeds])
    stacked, final_aucs, train_time, eval_time, gpu_train, gpu_eval, _ = (
        trainer._run_permutations_stacked(seeds))
    diff = np.abs(stacked - sequential)
    print(f"device={device}  n_train={len(train)}  streams={len(seeds)}  batches={len(train) // 128 + 1}")
    print(f"max |stacked - sequential| = {diff.max():.3e}   bit-identical: {np.array_equal(stacked, sequential)}")
    for index, seed in enumerate(seeds):
        error = abs(float(stacked[index].sum()) - (float(final_aucs[index]) - 0.5))
        print(f"  seed {seed}: final_auc={float(final_aucs[index]):.6f}  efficiency_error={error:.3e}")
    print(f"stacked wall: train {train_time:.3f}s, eval {eval_time:.3f}s"
          f" (gpu: {gpu_train:.3f}s / {gpu_eval:.3f}s)")
    sequential_wall = 0.0
    for seed in seeds:
        _, _, train_t, eval_t, _, _, _ = trainer._run_permutation(seed)
        sequential_wall += train_t + eval_t
    print(f"sequential wall (train+eval, {len(seeds)} streams): {sequential_wall:.3f}s")
    print("OK" if diff.max() < 1e-12 else "DIFFERS (see magnitude above)")


if __name__ == "__main__":
    main()
