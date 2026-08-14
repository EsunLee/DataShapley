#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G-Shapley groups serially or one per GPU")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5922, 13764, 14933])
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()
    if args.parallel and len(args.devices) < len(args.seeds):
        raise SystemExit("Parallel mode requires at least one distinct device per seed")
    if args.parallel and len(set(args.devices[:len(args.seeds)])) != len(args.seeds):
        raise SystemExit("Parallel mode requires distinct devices so each group owns one GPU")

    commands: list[list[str]] = []
    for index, seed in enumerate(args.seeds):
        device = args.devices[index] if args.parallel else args.devices[0]
        command = [sys.executable, str(ROOT / "scripts" / "run_group.py"),
                   "--data-dir", str(args.data_dir), "--results-dir", str(args.results_dir),
                   "--seed", str(seed), "--device", device, "--iterations", str(args.iterations)]
        if args.learning_rate is not None:
            command.extend(("--learning-rate", str(args.learning_rate)))
        commands.append(command)
    if args.parallel:
        processes = [subprocess.Popen(command) for command in commands]
        failures = [process.wait() for process in processes]
        if any(failures):
            raise SystemExit(f"One or more groups failed: {failures}")
    else:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
