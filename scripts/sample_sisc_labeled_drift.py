"""Select a small, reproducible SISC subset with exact drift labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/labeled_drift/sisc")
    parser.add_argument("--output", type=Path, default=ROOT / "data/labeled_drift_sisc_sampled")
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.per_type <= 0 or not source.is_dir():
        parser.error("per-type must be positive and source must be a directory")
    if output.exists() or source in output.parents:
        parser.error("output must not exist or be inside source")

    candidates: dict[str, dict[str, list[tuple[Path, dict]]]] = {}
    for manifest_path in sorted(source.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source"] != "SISC synthetic series":
            continue
        with (manifest_path.parent / manifest["files"]["labels"]).open(encoding="utf-8", newline="") as handle:
            drift_types = {f"{row['method']}:{row['transition']}" for row in csv.DictReader(handle)}
        if len(drift_types) != 1:
            raise ValueError(f"expected one drift type per SISC series: {manifest_path}")
        drift_type = drift_types.pop()
        family = manifest_path.parent.relative_to(source).parent.as_posix()
        candidates.setdefault(drift_type, {}).setdefault(family, []).append((manifest_path, manifest))

    rng = random.Random(args.seed)
    selected = []
    for drift_type, family_groups in sorted(candidates.items()):
        families = sorted(family_groups)
        rng.shuffle(families)
        for family in families:
            rng.shuffle(family_groups[family])
        type_selected = 0
        while type_selected < args.per_type:
            available = [family for family in families if family_groups[family]]
            if not available:
                raise ValueError(f"only {type_selected} series available for {drift_type}")
            for family in available:
                manifest_path, manifest = family_groups[family].pop()
                selected.append((drift_type, family, manifest_path, manifest))
                type_selected += 1
                if type_selected == args.per_type:
                    break

    output.mkdir(parents=True)
    rows = []
    for drift_type, family, manifest_path, manifest in selected:
        relative = manifest_path.parent.relative_to(source)
        destination = output / "sisc" / relative
        destination.mkdir(parents=True)
        shutil.copy2(manifest_path, destination / "manifest.json")
        for filename in manifest["files"].values():
            shutil.copy2(manifest_path.parent / filename, destination / filename)
        rows.append({"drift_type": drift_type, "family": f"sisc/{family}",
                     "stream_id": (Path("sisc") / relative).as_posix(), "rows": manifest["rows"],
                     "source": str(manifest_path.parent), "seed": args.seed})

    with (output / "selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"selected {len(rows)} SISC series into {output}")


if __name__ == "__main__":
    main()
