from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .data import DatasetBundle
from .device import resolve_device, synchronize
from .flops import permutation_flops
from .io_utils import atomic_json, atomic_numpy, atomic_torch, ensure_dir, sha256_array, write_csv
from .metrics import evaluate_auc
from .model import PlovadDecoder
from .seeds import derive_iteration_seeds, set_seed


@dataclass
class CostRow:
    iteration: int
    iteration_seed: int
    n_batches: int
    train_flops_iteration: int
    eval_flops_iteration: int
    total_flops_iteration: int
    train_flops_cumulative: int
    eval_flops_cumulative: int
    total_flops_cumulative: int
    wall_train_time: float
    wall_eval_time: float
    wall_total_time: float
    gpu_train_time: float
    gpu_eval_time: float
    wall_train_time_cumulative: float
    wall_eval_time_cumulative: float
    wall_total_time_cumulative: float
    initial_auc: float
    final_auc: float
    efficiency_error: float


def _load_cost_rows(path: Path) -> list[CostRow]:
    if not path.exists():
        return []
    integer_fields = {
        "iteration", "iteration_seed", "n_batches", "train_flops_iteration",
        "eval_flops_iteration", "total_flops_iteration", "train_flops_cumulative",
        "eval_flops_cumulative", "total_flops_cumulative",
    }
    rows: list[CostRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            parsed = {key: int(value) if key in integer_fields else float(value)
                      for key, value in raw.items()}
            rows.append(CostRow(**parsed))
    return rows


class GShapTrainer:
    def __init__(self, bundle: DatasetBundle, config: ExperimentConfig, learning_rate: float) -> None:
        self.bundle = bundle
        self.config = config
        self.learning_rate = float(learning_rate)
        self.device = resolve_device(config.runtime.device)
        self.train_x = torch.as_tensor(np.asarray(bundle.features[bundle.train_indices]), device=self.device, dtype=torch.float32)
        self.train_y = torch.as_tensor(np.asarray(bundle.labels[bundle.train_indices]), device=self.device, dtype=torch.float32)
        self.test_x = torch.as_tensor(np.asarray(bundle.features[bundle.test_indices]), device=self.device, dtype=torch.float32)
        self.test_y = torch.as_tensor(np.asarray(bundle.labels[bundle.test_indices]), device=self.device, dtype=torch.float32)
        self.loss_function = nn.BCEWithLogitsLoss()

    def _run_permutation(self, seed: int) -> tuple[np.ndarray, float, float, float, float, float, str]:
        set_seed(seed, self.config.runtime.deterministic)
        model = PlovadDecoder(self.config.model.input_dim, self.config.model.hidden_dim).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.0, weight_decay=0.0)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(len(self.train_x), generator=generator)
        permutation_hash = sha256_array(permutation.numpy())
        marginal = np.zeros(len(permutation), dtype=np.float64)
        previous_auc = self.config.train.baseline_auc
        train_time = 0.0
        evaluation_time = 0.0
        gpu_train_time = 0.0
        gpu_evaluation_time = 0.0
        train_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if self.device.type == "cuda" else None
        eval_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if self.device.type == "cuda" else None

        for start in range(0, len(permutation), self.config.train.batch_size):
            compact_cpu = permutation[start : start + self.config.train.batch_size]
            compact = compact_cpu.to(self.device)
            model.train()
            synchronize(self.device)
            begin = time.perf_counter()
            if train_events:
                train_events[0].record()
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss_function(model(self.train_x[compact]), self.train_y[compact])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for seed {seed}, batch {start}")
            loss.backward()
            optimizer.step()
            if train_events:
                train_events[1].record()
            synchronize(self.device)
            train_time += time.perf_counter() - begin
            if train_events:
                gpu_train_time += train_events[0].elapsed_time(train_events[1]) / 1_000.0

            synchronize(self.device)
            begin = time.perf_counter()
            if eval_events:
                eval_events[0].record()
            current_auc = evaluate_auc(model, self.test_x, self.test_y)
            if eval_events:
                eval_events[1].record()
            synchronize(self.device)
            evaluation_time += time.perf_counter() - begin
            if eval_events:
                gpu_evaluation_time += eval_events[0].elapsed_time(eval_events[1]) / 1_000.0
            positions = compact_cpu.numpy()
            marginal[positions] = (current_auc - previous_auc) / len(positions)
            previous_auc = current_auc

        efficiency_error = abs(float(marginal.sum()) - (previous_auc - self.config.train.baseline_auc))
        if efficiency_error > self.config.train.efficiency_tolerance:
            raise AssertionError(f"Efficiency identity failed: {efficiency_error:.3e}")
        return (marginal, previous_auc, train_time, evaluation_time,
                gpu_train_time, gpu_evaluation_time, permutation_hash)

    def run_group(self, master_seed: int) -> Path:
        output_dir = ensure_dir(self.config.results_dir / f"group_seed{master_seed}")
        costs_path = output_dir / "costs_iteration.csv"
        checkpoint_path = output_dir / "checkpoint.pt"
        seeds = derive_iteration_seeds(master_seed, self.config.train.iterations)
        atomic_numpy(output_dir / "iteration_seeds.npy", seeds)
        rows = _load_cost_rows(costs_path)

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            completed = int(checkpoint["completed_iterations"])
            group_sum = np.asarray(checkpoint["group_sum"], dtype=np.float64)
            if "cost_rows" in checkpoint:
                rows = [CostRow(**item) for item in checkpoint["cost_rows"]]
                if rows:
                    write_csv(costs_path, [asdict(item) for item in rows], list(asdict(rows[0])))
            if completed != len(rows):
                raise RuntimeError("Checkpoint and recoverable cost rows disagree")
        else:
            completed = 0
            group_sum = np.zeros(len(self.train_x), dtype=np.float64)

        train_flops, eval_flops, total_flops = permutation_flops(
            len(self.train_x), len(self.test_x), self.config.train.batch_size,
            self.config.model.input_dim, self.config.model.hidden_dim,
        )
        for iteration_index in range(completed, self.config.train.iterations):
            seed = int(seeds[iteration_index])
            wall_begin = time.perf_counter()
            (marginal, final_auc, train_time, eval_time, gpu_train_time,
             gpu_eval_time, permutation_hash) = self._run_permutation(seed)
            wall_total = time.perf_counter() - wall_begin
            group_sum += marginal
            iteration = iteration_index + 1
            previous = rows[-1] if rows else None
            row = CostRow(
                iteration=iteration, iteration_seed=seed,
                n_batches=math.ceil(len(self.train_x) / self.config.train.batch_size),
                train_flops_iteration=train_flops, eval_flops_iteration=eval_flops,
                total_flops_iteration=total_flops,
                train_flops_cumulative=(previous.train_flops_cumulative if previous else 0) + train_flops,
                eval_flops_cumulative=(previous.eval_flops_cumulative if previous else 0) + eval_flops,
                total_flops_cumulative=(previous.total_flops_cumulative if previous else 0) + total_flops,
                wall_train_time=train_time, wall_eval_time=eval_time, wall_total_time=wall_total,
                gpu_train_time=gpu_train_time, gpu_eval_time=gpu_eval_time,
                wall_train_time_cumulative=(previous.wall_train_time_cumulative if previous else 0.0) + train_time,
                wall_eval_time_cumulative=(previous.wall_eval_time_cumulative if previous else 0.0) + eval_time,
                wall_total_time_cumulative=(previous.wall_total_time_cumulative if previous else 0.0) + wall_total,
                initial_auc=self.config.train.baseline_auc, final_auc=final_auc,
                efficiency_error=abs(float(marginal.sum()) - (final_auc - self.config.train.baseline_auc)),
            )
            rows.append(row)
            if self.config.runtime.keep_marginals:
                atomic_numpy(ensure_dir(output_dir / "marginals") / f"marginal_{iteration:03d}.npy", marginal)
            if self.config.runtime.keep_permutations:
                atomic_json(ensure_dir(output_dir / "permutations") / f"perm_{iteration:03d}.json",
                            {"seed": seed, "sha256": permutation_hash})
            atomic_numpy(output_dir / "group_shap.npy", group_sum / iteration)
            write_csv(costs_path, [asdict(item) for item in rows], list(asdict(row)))
            atomic_numpy(output_dir / "scores_iteration.npy", np.asarray(
                [[item.initial_auc, item.final_auc] for item in rows], dtype=np.float64
            ))
            atomic_torch(checkpoint_path, {"completed_iterations": iteration, "group_sum": group_sum,
                                           "cost_rows": [asdict(item) for item in rows]})
            atomic_json(output_dir / "meta.json", {
                "master_seed": master_seed, "learning_rate": self.learning_rate,
                "completed_iterations": iteration, "target_iterations": self.config.train.iterations,
                "status": "complete" if iteration == self.config.train.iterations else "running",
                "config": self.config.to_dict(),
            })
        return output_dir
