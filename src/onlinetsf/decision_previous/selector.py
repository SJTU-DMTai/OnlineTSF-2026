"""Inference for the JSON regression trees produced by the lightweight study."""

from __future__ import annotations

import math
import struct


def select_action(selector: dict, state: dict) -> tuple[str, dict[str, float]]:
    """Match sklearn float32 thresholds without sklearn/pandas at runtime."""
    raw = [state.get(name) for name in selector["features"]]
    missing = [value is None or not math.isfinite(value) for value in raw]
    values = [median if absent else value for value, absent, median in zip(raw, missing, selector["medians"], strict=True)]
    design = [struct.unpack("f", struct.pack("f", value))[0] for value in values + missing]
    scores = {}
    for action in selector["actions"]:
        score = 0.0
        if action != "native":
            tree = selector["trees"][action]
            node = 0
            while tree["left"][node] != -1:
                node = tree["left"][node] if design[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
            score = tree["value"][node]
        scores[action] = score
    return max(scores, key=scores.get), scores
