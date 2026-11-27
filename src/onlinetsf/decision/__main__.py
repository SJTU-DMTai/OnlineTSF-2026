"""Independent preparation, hindsight search, teacher collection and deployment."""

import argparse
from pathlib import Path

import yaml

from .experiment import collect_experiment, evaluate_experiment, train_experiment
from .oracle import evaluate_oracle
from .oracle_greedy import evaluate_greedy_oracle
from .fixed_collection import collect_fixed_oracle
from .preparation import prepare_experiment


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/selector_experiment.yaml"))
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--compact-output", action="store_true", help="keep models, metadata and aggregate comparisons only")
            command.add_argument("--ema-beta", type=float)
            command.add_argument("--devices", nargs="+")
            command.add_argument("--workers-per-device", nargs="+", type=int)
    for name in ("oracle", "oracle_greedy"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True, help="prepared snapshots directory")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--split", choices=("history", "test", "online"), default="test")
        command.add_argument("--start-row", type=int, help="first raw target row, inclusive")
        command.add_argument("--end-row", type=int, help="last raw target row, exclusive")
        if name == "oracle":
            command.add_argument("--k", nargs="+", type=int, help="coarse-to-fine action lengths, e.g. 64 32 16")
            command.add_argument("--k-max", type=int, help="largest action length for a halving schedule")
            command.add_argument("--k-min", type=int, help="smallest action length for a halving schedule")
            command.add_argument("--beam-width", type=int)
            command.add_argument("--h-tail", nargs="+", type=int, help="one shared lookahead length, or one per round")
            command.add_argument("--workers", type=int)
            command.add_argument("--profile", action="store_true",
                                 help="write buffered wall-clock diagnostics to output/profile.jsonl (no added CUDA sync)")
        else:
            command.add_argument("--ema-beta", type=float)
    collect = commands.add_parser("collect")
    collect.add_argument("--data", type=Path, required=True, help="prepared snapshots directory")
    collect.add_argument("--oracle", type=Path, required=True, help="completed history beam search")
    collect.add_argument("--config", type=Path, help="optional collection/training overrides")
    collect.add_argument("--mode", choices=("reuse", "resample"),
                         help="reuse winning-round search expansions or resample along teacher trajectory")
    collect.add_argument("--anchor-epsilon", type=float,
                         help="nonnegative margin for teacher-anchored cost truncation in reuse mode")
    collect.add_argument("--no-anchor", action="store_true",
                         help="disable teacher-anchored cost truncation in reuse mode")
    collect.add_argument("--workers", type=int, help="worker processes for filling pruned actions or resampling")
    collect.add_argument("--profile", action="store_true",
                         help="write buffered wall-clock diagnostics to output/profile.jsonl (no added CUDA sync)")
    collect.add_argument("--output", type=Path, required=True)
    fixed = commands.add_parser("collect_fixed")
    fixed.add_argument("--data", type=Path, required=True, help="prepared snapshots directory")
    fixed.add_argument("--output", type=Path, required=True)
    fixed.add_argument("--window", type=int, default=32, help="evaluate and execute each action for this many origins")
    fixed.add_argument("--workers", type=int)
    fixed.add_argument("--profile", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, required=True, help="teacher collection directory")
    train.add_argument("--config", type=Path)
    train.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, required=True, help="teacher collection directory")
    evaluate.add_argument("--models", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--devices", nargs="+")
    evaluate.add_argument("--workers-per-device", nargs="+", type=int)
    group = evaluate.add_mutually_exclusive_group()
    group.add_argument("--ema-beta", type=float)
    group.add_argument("--ema-betas", nargs="+", help="EMA grid, e.g. 0 default 0.5; Native/Frozen run once")
    suite = commands.add_parser("suite", help="train and evaluate configured existing collections")
    suite.add_argument("--config", type=Path, required=True)
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--devices", nargs="+")
    suite.add_argument("--workers-per-device", nargs="+", type=int)
    suite.add_argument("--dry-run", action="store_true", help="validate inputs and write the experiment plan without fitting or forecasting")
    for command in (train, evaluate, suite):
        command.add_argument("--compact-output", action="store_true", help="keep models, metadata and aggregate comparisons only")
    suite.add_argument("--collections", nargs="+", help="configured collection ids to run, in the given order")
    args = parser.parse_args(argv)
    if args.command in {"prepare", "run"}:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.command == "prepare":
            prepare_experiment(config, args.config.resolve().parent, args.output)
        else:
            if args.ema_beta is not None:
                config.setdefault("evaluation", {}).pop("ema_betas", None)
                config["evaluation"]["ema_beta"] = args.ema_beta
            args.output.mkdir(parents=True, exist_ok=False)
            prepare_experiment(config, args.config.resolve().parent, args.output / "prepared")
            evaluate_oracle(args.output / "prepared", args.output / "teacher", split="history")
            collect_experiment(args.output / "prepared", args.output / "teacher", args.output / "data")
            train_experiment(args.output / "data", args.output / "models", compact_output=args.compact_output)
            evaluate_experiment(args.output / "data", args.output / "models", args.output / "test",
                                devices=args.devices, workers_per_device=args.workers_per_device, compact_output=args.compact_output)
    elif args.command == "oracle":
        search = {name: getattr(args, name) for name in ("k", "k_max", "k_min", "beam_width", "h_tail", "workers")
                  if getattr(args, name) is not None}
        if args.h_tail is not None and len(args.h_tail) == 1:
            search["h_tail"] = args.h_tail[0]
        evaluate_oracle(args.data, args.output, split=args.split, start_row=args.start_row,
                        end_row=args.end_row, search=search, profile=args.profile)
    elif args.command == "oracle_greedy":
        evaluate_greedy_oracle(args.data, args.output, split=args.split, start_row=args.start_row,
                               end_row=args.end_row, ema_beta=args.ema_beta)
    elif args.command == "collect":
        if args.no_anchor and args.anchor_epsilon is not None:
            raise ValueError("specify either --anchor-epsilon or --no-anchor, not both")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config else None
        kwargs = {}
        if args.mode is not None:
            kwargs["mode"] = args.mode
        if args.no_anchor:
            kwargs["anchor_epsilon"] = None
        elif args.anchor_epsilon is not None:
            kwargs["anchor_epsilon"] = args.anchor_epsilon
        if args.workers is not None:
            kwargs["workers"] = args.workers
        if args.profile:
            kwargs["profile"] = True
        collect_experiment(args.data, args.oracle, args.output, config=config, **kwargs)
    elif args.command == "collect_fixed":
        collect_fixed_oracle(args.data, args.output, window=args.window,
                             workers=args.workers, profile=args.profile)
    elif args.command == "train":
        options = yaml.safe_load(args.config.read_text(encoding="utf-8"))["selectors"] if args.config else None
        train_experiment(args.data, args.output, options, compact_output=args.compact_output)
    elif args.command == "suite":
        from .suite import run_suite

        run_suite(args.config, args.output, devices=args.devices,
                  workers_per_device=args.workers_per_device, dry_run=args.dry_run,
                  compact_output=args.compact_output, collection_ids=args.collections)
    else:
        betas = None if args.ema_betas is None else [None if value == "default" else float(value) for value in args.ema_betas]
        evaluate_experiment(args.data, args.models, args.output, devices=args.devices,
                            workers_per_device=args.workers_per_device, ema_beta=args.ema_beta, ema_betas=betas,
                            compact_output=args.compact_output)


if __name__ == "__main__":
    main()
