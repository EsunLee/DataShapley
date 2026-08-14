#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gshap.flops import forward_flops
from gshap.model import PlovadDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()
    import torch
    from thop import profile
    model = PlovadDecoder(args.input_dim, args.hidden_dim)
    macs, parameters = profile(model, inputs=(torch.zeros(1, args.input_dim),), verbose=False)
    print(json.dumps({"theoretical_weight_flops": forward_flops(args.input_dim, args.hidden_dim),
                      "thop_macs": int(macs), "thop_macs_times_2": int(2 * macs),
                      "parameters": int(parameters)}, indent=2))


if __name__ == "__main__":
    main()

