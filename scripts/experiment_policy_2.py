"""Command-line entry point; reusable implementation lives in onlinetsf."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onlinetsf.update_study.policy_experiment import main


if __name__ == "__main__":
    main()
