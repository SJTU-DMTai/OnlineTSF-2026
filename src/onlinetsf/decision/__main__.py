"""Collect, train and evaluate selectors using the current update action space."""

import argparse
from pathlib import Path

import yaml

from .experiment import collect_experiment, evaluate_experiment, train_experiment


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("selector_experiment.yaml"))
        command.add_argument("--output", type=Path, required=True, help="new output directory")
        command.add_argument("--strategy", help="override the forecasting strategy in profiles.yaml")
        command.add_argument("--actions", type=Path, help="override the candidate action YAML")
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, required=True, help="completed collection directory")
    train.add_argument("--config", type=Path, help="optional experiment YAML whose selectors section overrides training")
    train.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--models", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--devices", nargs="+", help="parallel OneNet evaluation on physical GPUs, e.g. cuda:0 cuda:2")
    evaluate.add_argument("--workers-per-device", nargs="+", type=int, help="concurrent policies per device, e.g. 1 2")
    oracle = commands.add_parser("oracle", help="evaluate only the greedy hindsight Oracle")
    oracle.add_argument("--data", type=Path, required=True, help="completed collection directory")
    oracle.add_argument("--output", type=Path, required=True)
    oracle.add_argument("--actions", type=Path, help="candidate YAML; defaults to the collected action catalog")
    oracle.add_argument("--steps", type=int, help="optional limit on test forecast origins")
    args = parser.parse_args(argv)
    if args.command in {"run", "collect"}:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.strategy:
            config["forecast"]["strategy"] = args.strategy
        if args.actions:
            config["actions"] = str(args.actions.resolve())
        if args.command == "collect":
            collect_experiment(config, args.config.resolve().parent, args.output)
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            collect_experiment(config, args.config.resolve().parent, args.output / "data")
            train_experiment(args.output / "data", args.output / "models")
            evaluate_experiment(args.output / "data", args.output / "models", args.output / "test")
    elif args.command == "train":
        options = yaml.safe_load(args.config.read_text(encoding="utf-8"))["selectors"] if args.config else None
        train_experiment(args.data, args.output, options)
    elif args.command == "evaluate":
        evaluate_experiment(args.data, args.models, args.output,
                            devices=args.devices, workers_per_device=args.workers_per_device)
    else:
        from .oracle import evaluate_oracle

        evaluate_oracle(args.data, args.output, actions=args.actions, steps=args.steps)


if __name__ == "__main__":
    main()
