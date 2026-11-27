"""Run the ordinary train/evaluate pipeline over named existing collections."""

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml

from .experiment import train_experiment, evaluate_experiment, write_action_usage
from .model_specs import expand_model_specs
from .preparation import save_json
from .selector import resolve_ema_beta


def run_suite(config_path: Path, output: Path, *, devices: list[str] | None = None,
              workers_per_device: list[int] | None = None, dry_run: bool = False,
              compact_output: bool = False, collection_ids: list[str] | None = None) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = expand_model_specs(config["selectors"])
    collections = config.get("collections")
    if not isinstance(collections, list) or not collections:
        raise ValueError("suite.collections must list named collected data directories")
    if collection_ids is not None:
        by_id = {item["id"]: item for item in collections}
        if len(set(collection_ids)) != len(collection_ids) or not collection_ids or set(collection_ids) - by_id.keys():
            raise ValueError("--collections requires unique configured collection ids")
        collections = [by_id[name] for name in collection_ids]
    evaluation = config.get("evaluation", {})
    requested = evaluation.get("ema_betas", [evaluation.get("ema_beta")])
    if not isinstance(requested, list) or not requested:
        raise ValueError("evaluation.ema_betas must be a nonempty list")
    plans, identifiers = [], set()
    for collection in collections:
        if set(collection) != {"id", "data"}:
            raise ValueError("each collection requires id and data")
        name = collection["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in identifiers:
            raise ValueError("collection ids must be unique path-safe names")
        identifiers.add(name)
        directory = (config_path.resolve().parent / collection["data"]).resolve()
        required = [directory / "manifest.json", directory / "training.npz"]
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{name}: missing or empty required input: {path}")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest.get("kind") != "collection":
            raise ValueError(f"incomplete collection: {directory}")
        with np.load(directory / "training.npz") as data:
            needed = {"states", "maes", "advantages", "origins", "label_ends"}
            missing = needed - set(data.files)
            if missing:
                raise ValueError(f"{name}: training.npz missing arrays: {sorted(missing)}")
            origins, label_ends = data["origins"], data["label_ends"]
            shape = data["maes"].shape
            if (shape != (len(origins), len(manifest["catalog"]["actions"]))
                    or data["advantages"].shape != shape
                    or data["states"].shape != (len(origins), len(manifest["feature_names"]))):
                raise ValueError(f"{name}: inconsistent training array shapes")
            if not len(origins) or not np.isfinite(data["maes"]).all() or not np.isfinite(data["advantages"]).all():
                raise ValueError("collection requires nonempty finite action costs")
            if any(job["loss"] == "top1_bpr" or job["label_mode"] == "hard" for job in models):
                if "teacher_actions" not in data:
                    raise ValueError("hard-label and BPR experiments require recorded teacher_actions")
        context, horizon = manifest["config"]["data"]["context_length"], manifest["config"]["data"]["horizon"]
        if (not np.array_equal(label_ends, origins + manifest["h_act"] + manifest["h_tail"] + horizon - 1)
                or np.any(origins + context < manifest["offline_end"])
                or np.any(label_ends + context > manifest["train_end"])):
            raise ValueError("collection labels cross their historical data boundaries")
        prepared_dir = (directory / manifest["prepared_data"]).resolve()
        required.extend([prepared_dir / "manifest.json", prepared_dir / "online_start.pt"])
        for path in required[2:]:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{name}: missing or empty required input: {path}")
        prepared = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
        if prepared["status"] != "complete" or not (prepared_dir / "online_start.pt").is_file():
            raise ValueError("collection requires complete forecasting snapshots")
        for key in ("offline_end", "train_end", "test_end", "test_start", "test_stop", "catalog"):
            if manifest[key] != prepared[key]:
                raise ValueError(f"collection and prepared snapshots disagree on {key}")
        if "prepared_sha256" in manifest and manifest["prepared_sha256"] != hashlib.sha256((prepared_dir / "manifest.json").read_bytes()).hexdigest():
            raise ValueError("prepared manifest changed after collection")
        seed = config["selectors"].get("seed", manifest["config"].get("seed", 0))
        forecast_seed = manifest["config"].get("seed", 0)
        if prepared["config"].get("seed", 0) != forecast_seed:
            raise ValueError(f"{name}: collection and snapshot forecasting seeds differ")
        betas = list(dict.fromkeys(resolve_ema_beta(value, manifest["tau"]) for value in requested))
        plans.append(dict(id=name, data=str(directory), states=shape[0], actions=shape[1],
                          seed=seed, forecast_seed=forecast_seed, required_files=[str(path) for path in required],
                          training_target_rows=[manifest["offline_end"], manifest["train_end"]],
                          test_target_rows=[manifest["train_end"], manifest["test_end"]],
                          h_act=manifest["h_act"], h_tail=manifest["h_tail"], ema_betas=betas,
                          models=len(models), online_trajectories=2 + len(models) * len(betas)))
        print(f"[OK] {name}: states={shape[0]}, actions={shape[1]}, selector_seed={seed}, forecast_seed={forecast_seed}; data={directory}", flush=True)
    plan = dict(status="planned" if dry_run else "running", config=str(config_path.resolve()),
                selectors=config["selectors"], models=models, collections=plans, results=[], compact_output=compact_output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", plan)
    if dry_run:
        print(f"Preflight passed: {len(plans)} collections; all required input files are present and training arrays validated. No training or online evaluation executed.", flush=True)
        return plan
    rows = []
    try:
        for item in plans:
            root = output / item["id"]
            train_experiment(Path(item["data"]), root / "models", config["selectors"], compact_output=compact_output)
            evaluate_experiment(Path(item["data"]), root / "models", root / "test",
                                devices=devices, workers_per_device=workers_per_device,
                                ema_betas=requested, compact_output=compact_output)
            result = json.loads((root / "test/manifest.json").read_text(encoding="utf-8"))
            rows.extend({"collection": item["id"], **row} for row in result["results"])
            plan["results"].append(dict(id=item["id"], status="complete", trajectories=len(result["results"])))
            save_json(output / "manifest.json", plan)
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = list(dict.fromkeys(key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))))
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        write_action_usage(output / "action_usage.csv", rows)
        plan["status"] = "complete"
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", plan)
    return plan
