# -*- coding: utf-8 -*-
"""Generate repeated mean, scale, and permutation drift datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate repeated labeled drift datasets")
    parser.add_argument("--source", required=True)
    parser.add_argument("--stable-intervals", required=True)
    parser.add_argument("--output", required=True, help="new batch output directory")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--min-length", type=int, default=512)
    parser.add_argument("--time-column", default="date")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    generator = Path(__file__).with_name("inject_synthetic_drift.py")
    for repetition in range(args.repetitions):
        for method in ("mean", "scale", "permutation"):
            command = [
                sys.executable,
                str(generator),
                "--source", args.source,
                "--stable-intervals", args.stable_intervals,
                "--method", method,
                "--min-length", str(args.min_length),
                "--time-column", args.time_column,
                "--seed", str(args.seed + repetition),
                "--output", str(destination / f"{method}-{repetition:03d}"),
            ]
            subprocess.run(command, check=True)
    print(f"generated={3 * args.repetitions} output={destination.resolve()}")


if __name__ == "__main__":
    main()
