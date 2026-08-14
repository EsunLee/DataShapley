#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gshap.config import DataConfig, ExperimentConfig, RuntimeConfig, TrainConfig
from gshap.data import load_or_create_bundle
from gshap.lr_search import read_selected_learning_rate
from gshap.trainer import GShapTrainer


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one G-Shapley group")
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--results-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--learning-rate", type=float)
    result.add_argument("--device", default="auto")
    result.add_argument("--iterations", type=int, default=50)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--test-size", type=int, default=5_000)
    result.add_argument("--lr-val-size", type=int, default=5_000)
    result.add_argument("--feature-file", default="ucf_train_feat_704459x512.npy")
    result.add_argument("--label-file", default="ucf_train_label_704459x1.npy")
    result.add_argument("--keep-marginals", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    data = DataConfig(feature_file=args.feature_file, label_file=args.label_file,
                      test_size=args.test_size, lr_val_size=args.lr_val_size)
    train = TrainConfig(batch_size=args.batch_size, iterations=args.iterations,
                        learning_rate=args.learning_rate)
    runtime = RuntimeConfig(device=args.device, keep_marginals=args.keep_marginals)
    config = ExperimentConfig(args.data_dir, args.results_dir, data=data, train=train, runtime=runtime)
    bundle = load_or_create_bundle(args.data_dir, args.results_dir / "split", data)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = read_selected_learning_rate(args.results_dir / "split" / "lr_search.json")
    output = GShapTrainer(bundle, config, learning_rate).run_group(args.seed)
    print(f"Completed group {args.seed}: {output}")


if __name__ == "__main__":
    main()

