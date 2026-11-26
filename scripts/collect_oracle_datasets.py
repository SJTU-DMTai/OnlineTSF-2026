"""Collect Beam and fixed-32 oracle training data on ETTm1 history (20%-40%)."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onlinetsf.decision.collection import collect_experiment
from onlinetsf.decision.fixed_collection import collect_fixed_oracle
from onlinetsf.decision.oracle import evaluate_oracle
from onlinetsf.decision.preparation import prepare_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=("dsof", "onenet", "fsnet"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="logical device after CUDA_VISIBLE_DEVICES mapping")
    parser.add_argument("--workers", type=int, help="override each YAML's search workers")
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must be distinct")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("workers must be positive")
    config_root = Path(__file__).resolve().parents[1] / "configs"
    args.output.mkdir(parents=True, exist_ok=False)
    for method in args.methods:
        config_path = config_root / f"selector_ettm1_{method}_manual.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["split"] = {"offline_fraction": .2, "train_end_fraction": .4, "test_end_fraction": 1.}
        config["forecast"]["device"] = args.device
        if args.workers is not None:
            config["search"]["workers"] = args.workers
        config["collection"].update(mode="reuse", train_states=None)
        config["collection"].pop("h_act", None)
        config["collection"].pop("h_tail", None)
        root = args.output / method
        root.mkdir()
        (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"=== {method}: prepare -> Beam search -> Beam data -> fixed-32 data ===", flush=True)
        prepare_experiment(config, config_root, root / "prepared")
        evaluate_oracle(root / "prepared", root / "beam_oracle", split="history", profile=True)
        collect_experiment(root / "prepared", root / "beam_oracle", root / "beam_data", mode="reuse", profile=True)
        collect_fixed_oracle(root / "prepared", root / "fixed32_data", window=32,
                             workers=config["search"].get("workers", 1), profile=True)
        print(f"=== {method}: collection complete: {root} ===", flush=True)


if __name__ == "__main__":
    main()
