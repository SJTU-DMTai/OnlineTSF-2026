"""Paired future benefits, hindsight action rankings, and exploratory state associations."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from .reporting import write_csv


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 4:
        return None
    xs, ys = zip(*pairs, strict=True)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def analyze(root: Path) -> None:
    summaries, states, outcomes = [], {}, []
    identity = ("dataset", "method", "horizon", "seed", "probe_id")
    # The same entry point accepts one run or a parent containing GPU shards.
    # policies.json identifies job roots and avoids counting shard summaries twice.
    for marker in sorted(root.rglob("policies.json")):
        directory = marker.parent
        if (directory / "summary.csv").exists():
            summaries.extend(read_csv(directory / "summary.csv"))
        if (directory / "probe_states.csv").exists():
            for row in read_csv(directory / "probe_states.csv"):
                states[tuple(row[key] for key in identity)] = row
        if (directory / "probe_outcomes.csv").exists():
            outcomes.extend(read_csv(directory / "probe_outcomes.csv"))
    grouped = defaultdict(list)
    for row in outcomes:
        grouped[tuple(row[key] for key in identity) + (row["lookahead"],)].append(row)
    benefits, best = [], []
    relationships = defaultdict(list)
    drift_relationships = defaultdict(list)
    patterns = defaultdict(list)
    offline_features = {"event_rows", "event_peak_votes", "event_consensus_rows"}
    excluded = set(identity) | {"kind", "index", "clock", "event_id"} | offline_features
    for key, rows in grouped.items():
        reference = next(row for row in rows if row["policy"] == "native")
        hold = next((row for row in rows if row["policy"] == "frozen"), None)
        winner = min(rows, key=lambda row: float(row["mae"]))
        best.append({name: reference[name] for name in (*identity, "kind", "lookahead")} |
                    {"hindsight_best_policy": winner["policy"], "best_mae": winner["mae"],
                     "native_mae": reference["mae"], "native_regret": float(reference["mae"]) - float(winner["mae"])})
        state = states[key[:-1]]
        for row in rows:
            native_mae = float(reference["mae"])
            benefit = native_mae - float(row["mae"])
            benefits.append(dict(row, benefit_vs_native=benefit,
                                 relative_benefit_vs_native=benefit / native_mae if native_mae else None,
                                 benefit_vs_frozen=float(hold["mae"]) - float(row["mae"]) if hold else None))
            pattern_key = tuple(row[name] for name in ("dataset", "method", "horizon", "seed", "lookahead", "kind", "policy"))
            patterns[pattern_key].append(benefit)
            for feature, value in state.items():
                if feature in offline_features and value != "" and row["kind"] != "stable":
                    relation_key = tuple(row[name] for name in ("dataset", "method", "horizon", "seed", "lookahead", "policy")) + (feature,)
                    drift_relationships[relation_key].append((float(value), benefit))
                if feature in excluded or value == "":
                    continue
                relation_key = tuple(row[name] for name in ("dataset", "method", "horizon", "seed", "lookahead", "policy")) + (feature,)
                relationships[relation_key].append((float(value), benefit))
    relation_rows = []
    for key, pairs in sorted(relationships.items()):
        correlation = pearson(pairs)
        if correlation is not None:
            relation_rows.append(dict(zip(("dataset", "method", "horizon", "seed", "lookahead", "policy", "feature"), key, strict=True),
                                      probes=len(pairs), pearson_benefit_vs_native=correlation))
    pattern_rows = []
    for key, values in sorted(patterns.items()):
        pattern_rows.append(dict(zip(("dataset", "method", "horizon", "seed", "lookahead", "kind", "policy"), key, strict=True),
                                 probes=len(values), mean_benefit_vs_native=sum(values) / len(values),
                                 beneficial_fraction=sum(value > 0 for value in values) / len(values)))
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "paired_benefits.csv", benefits)
    write_csv(root / "best_actions.csv", best)
    write_csv(root / "relationships.csv", relation_rows)
    drift_rows = []
    for key, pairs in sorted(drift_relationships.items()):
        correlation = pearson(pairs)
        if correlation is not None:
            drift_rows.append(dict(zip(("dataset", "method", "horizon", "seed", "lookahead", "policy", "feature"), key, strict=True),
                                  probes=len(pairs), pearson_benefit_vs_native=correlation))
    write_csv(root / "drift_relationships.csv", drift_rows)
    write_csv(root / "pattern_summary.csv", pattern_rows)
    (root / "analysis_notes.md").write_text(
        "# 更新策略探索结果\n\n"
        f"全流结果 {len(summaries)} 行；同状态分支比较 {len(benefits)} 行。\n\n"
        "收益为同一探针上 native MAE 减去候选 MAE，正值表示候选更好。"
        "lookahead 是连续发出的预测窗口数，H>1 时目标重叠；不能将这些窗口当作独立重复实验。\n\n"
        "best_actions 是有限候选集合中使用未来标签选择的事后较优动作，仅用于诊断。"
        "relationships 是按数据集、方法、H、seed、评价范围和动作分组的探索性 Pearson 相关；"
        "不含事件长度、位置或伪标签等事后特征，不是因果效应或显著性检验。"
        "探针来自原生轨迹，不能直接保证控制器部署后的收益。应在独立时间段和随机种子验证规律。\n\n"
        "drift_relationships 单列伪标签事件长度、票数等事后特征；这些特征不可直接用作在线控制器输入。"
        "同一事件的多个探针不独立。共识标签描述观测分布变化，不能单独证明条件预测关系发生概念漂移。\n\n"
        "分支成本列对应整个 branch_steps 区块，不能解释成每个短 lookahead 的成本。"
        "误差处于离线前缀固定 z-score 尺度，未使用漂移检测器的 sigmoid 变换。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
