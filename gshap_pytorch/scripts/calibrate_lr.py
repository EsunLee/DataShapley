#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gshap.config import DataConfig, ExperimentConfig, RuntimeConfig, TrainConfig
from gshap.data import load_or_create_bundle
from gshap.lr_search import calibrate_learning_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate one-pass SGD learning rate")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--lr-val-size", type=int, default=5_000)
    parser.add_argument("--lr-train-size", type=int, default=100_000)
    parser.add_argument("--feature-file", default="ucf_train_feat_704459x512.npy")
    parser.add_argument("--label-file", default="ucf_train_label_704459x1.npy")
    parser.add_argument("--skip-confirm", action="store_true")
    args = parser.parse_args()

    data = DataConfig(feature_file=args.feature_file, label_file=args.label_file,
                      test_size=args.test_size, lr_val_size=args.lr_val_size,
                      lr_train_size=args.lr_train_size)
    train = TrainConfig(batch_size=args.batch_size)
    config = ExperimentConfig(args.data_dir, args.results_dir, data=data, train=train,
                              runtime=RuntimeConfig(device=args.device))
    split_dir = args.results_dir / "split"
    bundle = load_or_create_bundle(args.data_dir, split_dir, data)
    result = calibrate_learning_rate(bundle, config, split_dir, confirm=not args.skip_confirm)
    print(f"Selected learning rate: {result.learning_rate:.8g}; confirmation AUC={result.confirmation_auc}")


if __name__ == "__main__":
    main()

