#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gshap.io_utils import atomic_json, atomic_numpy, write_csv


def read_costs(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


TIME_COLUMNS = {
    "wall_train_time", "wall_eval_time", "wall_total_time",
    "gpu_train_time", "gpu_eval_time",
    "wall_train_time_cumulative", "wall_eval_time_cumulative",
    "wall_total_time_cumulative",
}


def round_time_values(costs: list[list[dict[str, str]]]) -> list[list[dict[str, str]]]:
    """Report time columns to two decimal places (deliverable precision).

    FLOPs stay integer-valued and AUC / efficiency_error keep full precision
    (rounding final_auc to 2 dp would break the reported efficiency identity).
    """
    def round_row(row: dict[str, str]) -> dict[str, str]:
        return {key: (f"{float(value):.2f}" if key in TIME_COLUMNS else value)
                for key, value in row.items()}
    return [[round_row(item) for item in group] for group in costs]


def matrix_csv(path: Path, seeds: list[int], costs: list[list[dict[str, str]]], column: str) -> None:
    count = len(costs[0])
    fieldnames = ["seed"] + [f"permutation_{index}" for index in range(1, count + 1)]
    rows = []
    for seed, group in zip(seeds, costs, strict=True):
        row: dict[str, str | int] = {"seed": seed}
        row.update({f"permutation_{index}": group[index - 1][column]
                    for index in range(1, count + 1)})
        rows.append(row)
    write_csv(path, rows, fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed G-Shapley groups")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5922, 13764, 14933])
    args = parser.parse_args()

    arrays = []
    costs = []
    for seed in args.seeds:
        group = args.results_dir / f"group_seed{seed}"
        with (group / "meta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta["status"] != "complete":
            raise RuntimeError(f"Group {seed} is not complete")
        arrays.append(np.load(group / "group_shap.npy"))
        costs.append(read_costs(group / "costs_iteration.csv"))
    if len({len(item) for item in costs}) != 1:
        raise RuntimeError("Groups contain different iteration counts")
    costs = round_time_values(costs)
    values = np.stack(arrays)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    train_indices = np.load(args.results_dir / "split" / "train_indices.npy")
    with (args.results_dir / "split" / "split_meta.json").open("r", encoding="utf-8") as handle:
        n_total = int(json.load(handle)["n_total"])
    mean_all = np.full(n_total, np.nan, dtype=np.float64)
    std_all = np.full(n_total, np.nan, dtype=np.float64)
    mean_all[train_indices] = mean
    std_all[train_indices] = std
    atomic_numpy(args.results_dir / "shap_mean_train.npy", mean)
    atomic_numpy(args.results_dir / "shap_std_train.npy", std)
    atomic_numpy(args.results_dir / "shap_mean_all.npy", mean_all)
    atomic_numpy(args.results_dir / "shap_std_all.npy", std_all)

    summary = []
    for name, array in (("mean", mean), ("std", std)):
        summary.append({"statistic": name, "count": len(array), "mean": float(array.mean()),
                        "std": float(array.std()), "min": float(array.min()),
                        "q25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
                        "q75": float(np.quantile(array, 0.75)), "max": float(array.max())})
    write_csv(args.results_dir / "shap_summary.csv", summary,
              ["statistic", "count", "mean", "std", "min", "q25", "median", "q75", "max"])

    outputs = {
        "flops_3x50_train.csv": "train_flops_cumulative",
        "flops_3x50_total.csv": "total_flops_cumulative",
        "time_3x50_train.csv": "wall_train_time_cumulative",
        "time_3x50_total.csv": "wall_total_time_cumulative",
    }
    for filename, column in outputs.items():
        matrix_csv(args.results_dir / filename, args.seeds, costs, column)
    for seed, group_costs in zip(args.seeds, costs, strict=True):
        write_csv(args.results_dir / f"cost_detail_seed{seed}.csv", group_costs, list(group_costs[0]))
    atomic_json(args.results_dir / "aggregate_meta.json", {
        "seeds": args.seeds, "groups": len(args.seeds), "iterations": len(costs[0]),
        "n_train": len(mean), "n_total": n_total, "std_ddof": 1,
    })
    print(f"Aggregated {len(args.seeds)} groups into {args.results_dir}")


if __name__ == "__main__":
    main()

