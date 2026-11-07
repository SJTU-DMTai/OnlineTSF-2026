"""Standalone entry for the current greedy Oracle; no selector training required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onlinetsf.decision.__main__ import main


if __name__ == "__main__":
    main(["oracle", *sys.argv[1:]])
