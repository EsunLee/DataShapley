#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gshap.data import synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5_000)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    synthetic_dataset(args.output_dir, args.n, args.dim, args.seed)


if __name__ == "__main__":
    main()

