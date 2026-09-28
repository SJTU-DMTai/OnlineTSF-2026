# -*- coding: utf-8 -*-
"""Data selection, initialization, and reporting shared by experiments A--D."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from operator import itemgetter
from statistics import mean, stdev

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onlinetsf.__main__ import _build_backbone, _build_method, _run_offline_training
from onlinetsf.config import load_config
from onlinetsf.data import load_benchmark_dataset
from onlinetsf.methods import UnderCaliMethod


STRATEGIES = ("dlinear_dsof", "tcn_under_cali", "fsnet", "onenet")


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/labeled_drift")
    parser.add_argument("--profiles", type=Path, default=ROOT / "profiles.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 24])
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--families", nargs="+", help="family glob patterns, e.g. liu2023/sudden sisc/*")
    parser.add_argument("--drift-types", nargs="+", help="mechanism, transition, or mechanism:transition")
    parser.add_argument("--max-per-family", type=int, default=3, help="stratified sample size; 0 selects all")
    parser.add_argument("--max-streams", type=int)
    parser.add_argument("--train-rows", type=int, help="raw prefix length; default 1000 for synthetic SISC, 256 otherwise")
    parser.add_argument("--allow-mixed-prefix", action="store_true", help="allow labeled drift inside training; always flagged")
    parser.add_argument("--min-pre-rows", type=int, default=0)
    parser.add_argument("--min-post-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--uncertainty-epochs", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-online-steps", type=int, help="smoke-test limit; recovery truncation is reported")
    parser.add_argument("--pre-rows", type=int, default=128)
    parser.add_argument("--post-rows", type=int, default=256)
    parser.add_argument("--write-values", action="store_true", help="also write per-horizon, per-target forecasts")
    parser.add_argument("--parameter-metrics", action="store_true", help="measure parameter movement; adds copying overhead")
    return parser


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_stream(path: Path, root: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    with (path.parent / manifest["files"]["labels"]).open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    for label in labels:
        for key in ("start_index", "end_index_exclusive", "new_concept_full_from"):
            label[key] = int(label[key]) if label[key] != "" else None
        label["drift_type"] = f"{label['method']}:{label['transition']}"
        label["affected_variables"] = json.loads(label["affected_variables"])
        start, end, full = label["start_index"], label["end_index_exclusive"], label["new_concept_full_from"]
        if not 0 <= start < end <= manifest["rows"] or (full is not None and not start <= full <= manifest["rows"]):
            raise ValueError(f"invalid drift interval in {path}")
    labels.sort(key=itemgetter("start_index"))
    if len({label["start_index"] for label in labels}) != len(labels):
        raise ValueError(f"simultaneous drift labels require explicit merging: {path}")
    for previous, current in zip(labels, labels[1:]):
        if (previous["new_concept_full_from"] or previous["end_index_exclusive"]) > current["start_index"]:
            raise ValueError(f"overlapping transitions require explicit attribution: {path}")
    source = manifest["source"]
    if source.startswith("Liu"):
        family = "liu2023/" + labels[0]["method"]
    elif source.startswith("Mendeley"):
        family = "mendeley/" + path.parent.name
    elif source == "SISC synthetic series":
        source_path = manifest["source_path"].replace("\\", "/")
        family = "sisc/" + source_path.split("/Series Sinteticas/")[1].rsplit("/", 1)[0]
    elif source == "SISC real financial return series":
        family = "sisc/real"
    else:
        family = path.parent.parent.name
    relative_id = path.parent.relative_to(root).as_posix()
    return {"stream_id": path.parent.name if relative_id == "." else relative_id,
            "path": path.parent, "family": family, "manifest": manifest, "labels": labels}


def phase_at(raw_index: int, labels: list[dict], pre_rows: int = 128, post_rows: int = 256) -> dict:
    """Exclusive analysis attribution; this metadata never reaches ordinary policies."""
    previous = None
    following = None
    for label in labels:
        if label["start_index"] <= raw_index:
            previous = label
        else:
            following = label
            break
    phase = "stable"
    label = previous or following
    if previous is not None:
        full = previous["new_concept_full_from"]
        if full is None:
            # Reference annotations do not establish an exact end of transition.
            phase = "reference_post"
        elif raw_index < full:
            phase = "transition"
        elif raw_index - full < min(32, post_rows):
            phase = "post_early"
        elif raw_index - full < min(128, post_rows):
            phase = "post_middle"
        elif raw_index - full < post_rows:
            phase = "post_late"
    if phase in {"stable", "reference_post"} and following is not None and following["start_index"] - raw_index <= pre_rows:
        phase, label = "pre_drift", following
    return {"phase": phase, "event_id": label["event_id"] if label else "",
            "mechanism": label["method"] if label else "none",
            "transition": label["transition"] if label else "none",
            "drift_type": label["drift_type"] if label else "none"}


def initialize_experiment(args) -> list[dict]:
    positive = [args.context_length, args.epochs, args.batch_size, args.threads, args.pre_rows, args.post_rows, *args.horizons]
    optional = [args.train_rows, args.max_streams, args.max_online_steps, args.uncertainty_epochs]
    if any(value <= 0 for value in positive) or any(value is not None and value <= 0 for value in optional):
        raise ValueError("lengths, epochs, batch size, threads and optional limits must be positive")
    if min(args.max_per_family, args.min_pre_rows, args.min_post_rows) < 0:
        raise ValueError("sampling and eligibility limits must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.horizons)) != len(args.horizons) or len(set(args.strategies)) != len(args.strategies):
        raise ValueError("seeds, horizons and strategies must not contain duplicates")
    profiles = yaml.safe_load(args.profiles.read_text(encoding="utf-8"))
    unknown = set(args.strategies).difference(profiles["strategies"])
    if unknown:
        raise ValueError(f"unknown strategy profiles: {sorted(unknown)}")
    torch.set_num_threads(args.threads)
    groups = defaultdict(list)
    inventory = []
    for path in sorted(args.data_root.rglob("manifest.json")):
        stream = read_stream(path, args.data_root)
        family, manifest, labels = stream["family"], stream["manifest"], stream["labels"]
        if args.families and not any(fnmatch.fnmatchcase(family, pattern) for pattern in args.families):
            continue
        if args.drift_types and not any(value in args.drift_types for label in labels for value in (label["method"], label["transition"], label["drift_type"])):
            continue
        prefix = args.train_rows or (1000 if manifest["source"] == "SISC synthetic series" else 256)
        continuous = bool(labels and labels[0]["start_index"] == 0 and labels[0]["new_concept_full_from"] == manifest["rows"])
        reference = manifest.get("label_status") == "reference_not_exact_ground_truth"
        mixed = bool(labels and labels[0]["start_index"] < prefix)
        reason = ""
        if prefix < args.context_length + max(args.horizons) or prefix + max(args.horizons) > manifest["rows"]:
            reason = "insufficient_training_or_online_rows"
        elif mixed and not (continuous or reference or args.allow_mixed_prefix):
            reason = "drift_inside_offline_prefix"
        elif args.min_pre_rows or args.min_post_rows:
            eligible = False
            for position, label in enumerate(labels):
                next_start = labels[position + 1]["start_index"] if position + 1 < len(labels) else manifest["rows"]
                full = label["new_concept_full_from"]
                if label["start_index"] - prefix >= args.min_pre_rows and (args.min_post_rows == 0 or full is not None and next_start - full >= args.min_post_rows):
                    eligible = True
            if not eligible:
                reason = "no_event_satisfies_pre_post_lengths"
        stream.update(train_rows=prefix, mixed_prefix=mixed, reference_labels=reference)
        record = {"stream_id": stream["stream_id"], "family": family, "rows": manifest["rows"],
                  "train_rows": prefix, "mixed_prefix": mixed, "reference_labels": reference,
                  "drift_types": ";".join(sorted({label["drift_type"] for label in labels})),
                  "status": reason or "eligible"}
        inventory.append(record)
        if not reason:
            groups[family].append(stream)
    generator = random.Random(args.selection_seed)
    selected = []
    for family in sorted(groups):
        candidates = groups[family]
        if args.max_per_family and len(candidates) > args.max_per_family:
            candidates = generator.sample(candidates, args.max_per_family)
        selected.extend(sorted(candidates, key=itemgetter("stream_id")))
    if args.max_streams:
        selected = selected[:args.max_streams]
    if not selected:
        raise ValueError("no eligible streams; inspect filters, prefix length, and data root")
    args.output.mkdir(parents=True, exist_ok=False)
    chosen = {stream["stream_id"] for stream in selected}
    for record in inventory:
        if record["status"] == "eligible":
            record["status"] = "selected" if record["stream_id"] in chosen else "not_sampled"
    write_rows(args.output / "inventory.csv", inventory)
    write_json(args.output / "settings.json", vars(args))
    return selected


def prepared_runs(args, streams):
    """Yield one trained initialization; each policy must clone it before use."""
    for number, stream in enumerate(streams):
        for strategy in args.strategies:
            for horizon in args.horizons:
                for seed in args.seeds:
                    directory = args.output / f"stream-{number:04d}" / strategy / f"h{horizon}-seed{seed}"
                    directory.mkdir(parents=True)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    manifest = stream["manifest"]
                    dataset = load_benchmark_dataset(
                        "labeled", stream["path"] / manifest["files"]["data"], args.context_length, horizon,
                        time_column=manifest["time_column"], target_columns=manifest["target_columns"],
                        feature_columns=manifest["feature_columns"], include_time_features=False,
                    )
                    if dataset.values.shape[0] != manifest["rows"]:
                        raise ValueError(f"manifest row count differs from CSV: {stream['stream_id']}")
                    train_windows = stream["train_rows"] - args.context_length - horizon + 1
                    # The half-window offset prevents floating-point floor errors in the existing trainer.
                    ratio = (train_windows + 0.5) / len(dataset)
                    config = {"seed": seed, "profiles": str(args.profiles.resolve()),
                              "selection": {"strategy": strategy, "detector": "none"},
                              "data": {"name": "labeled", "path": str((stream["path"] / manifest["files"]["data"]).resolve()),
                                       "context_length": args.context_length, "horizon": horizon, "stride": 1,
                                       "time_column": manifest["time_column"], "target_columns": manifest["target_columns"]},
                              "offline": {"train_ratio": ratio, "epochs": args.epochs, "batch_size": args.batch_size},
                              "online": {"feedback_delay": horizon, "device": args.device}}
                    profiles = yaml.safe_load(args.profiles.read_text(encoding="utf-8"))
                    if profiles["strategies"][strategy]["method"]["name"] == "dsof":
                        config["online"]["feedback_delay"] = list(range(1, horizon + 1))
                    config_path = directory / "config.yaml"
                    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                    config = load_config(config_path)
                    if args.uncertainty_epochs is not None and config["method"]["name"] == "under_cali":
                        config["method"]["uncertainty_epochs"] = args.uncertainty_epochs
                    dataset.standardize(train_windows)
                    config["data"].update(num_features=dataset.num_features, target_names=list(dataset.target_names))
                    model = _build_backbone(config, dataset.num_features, dataset.num_targets, dataset.target_indices)
                    method = _build_method(config, model)
                    trained = _run_offline_training(dataset, method, config["offline"])
                    assert trained == train_windows
                    if isinstance(method, UnderCaliMethod):
                        method.pretrain_uncertainty(dataset, train_windows, args.batch_size)
                    start = stream["train_rows"] - args.context_length
                    stop = min(len(dataset), start + args.max_online_steps) if args.max_online_steps else len(dataset)
                    metadata = {"stream_id": stream["stream_id"], "family": stream["family"], "strategy": strategy,
                                "horizon": horizon, "seed": seed, "train_rows": stream["train_rows"],
                                "mixed_prefix": stream["mixed_prefix"], "reference_labels": stream["reference_labels"]}
                    write_json(directory / "resolved.json", {**metadata, "config": config, "start": start, "stop": stop,
                                                             "train_windows": train_windows, "labels": stream["labels"]})
                    print(f"initialized {stream['stream_id']} {strategy} H={horizon} seed={seed}", flush=True)
                    yield stream, dataset, method, start, stop, directory, metadata


def aggregate_metrics(rows: list[dict], path: Path, group_fields: tuple[str, ...], metrics: tuple[str, ...]) -> None:
    """Macro-average seeds within streams before treating streams as replicates."""
    groups = {}
    for row in rows:
        group = groups.setdefault(tuple(row[field] for field in group_fields), {})
        group.setdefault(row["stream_id"], []).append(row)
    result = []
    for key, streams in sorted(groups.items()):
        record = dict(zip(group_fields, key))
        record["streams"] = len(streams)
        record["records"] = sum(len(stream_rows) for stream_rows in streams.values())
        for metric in metrics:
            values = []
            for stream_rows in streams.values():
                available = [row[metric] for row in stream_rows if row.get(metric) is not None]
                if available:
                    values.append(mean(available))
            record[metric + "_mean"] = mean(values) if values else None
            record[metric + "_std_across_streams"] = stdev(values) if len(values) > 1 else None
        result.append(record)
    write_rows(path, result)
