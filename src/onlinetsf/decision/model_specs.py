"""Resolve configured selector variants into unique, reproducible training jobs."""

import math
import re

from .selector import ESTIMATORS, FORMULATIONS, FORMULATION_PARAMETERS
from .ranking import RANKING_PAIRS


def expand_model_specs(options: dict) -> list[dict]:
    if "formulations" in options or "estimators" in options:
        raise ValueError("replace selectors.formulations/estimators with selectors.models")
    seed = options.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("selectors.seed must be an integer in [0, 2**32)")
    parameters = options.get("parameters", {})
    if set(parameters) - set(ESTIMATORS):
        raise ValueError("unknown estimator parameter section")
    formulation_parameters = options.get("formulation_parameters", {})
    if set(formulation_parameters) - set(FORMULATIONS):
        raise ValueError("unknown formulation parameter section")
    if any(set(values) - set(ESTIMATORS) for values in formulation_parameters.values()):
        raise ValueError("unknown estimator in formulation parameters")
    ranking = options.get("ranking", {})
    if set(ranking) - {"r_max"}:
        raise ValueError("unknown selector ranking option")
    r_max = ranking.get("r_max", 1.)
    if isinstance(r_max, bool) or not isinstance(r_max, (int, float)) or not math.isfinite(r_max) or r_max <= 0:
        raise ValueError("ranking r_max must be finite and positive")
    if set(options.get("classification", {})) - {"temperature", "temperature_candidates", "cv_splits"}:
        raise ValueError("unknown classification temperature option")
    models = options.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("selectors.models must be a nonempty list")
    variants = options.get("variants", {})
    if set(variants) - set(ESTIMATORS):
        raise ValueError("unknown estimator in selectors.variants")
    jobs, identifiers = [], set()
    for model in models:
        if (not isinstance(model, dict) or not {"formulation", "estimator"} <= model.keys()
                or model.keys() - {"formulation", "estimator", "loss", "label_mode", "variants", "parameters"}):
            raise ValueError("each selectors.models entry requires formulation and estimator")
        formulation, estimator = model["formulation"], model["estimator"]
        loss, label_mode = model.get("loss", "default"), model.get("label_mode", "soft")
        if formulation not in FORMULATIONS or estimator not in ESTIMATORS:
            raise ValueError("unknown selector formulation or estimator")
        if loss not in {"default", "top1_bpr"}:
            raise ValueError(f"unknown selector loss: {loss}")
        if loss == "top1_bpr" and (formulation, estimator) not in RANKING_PAIRS:
            raise ValueError("top1_bpr supports only MLP selectors")
        if label_mode not in {"soft", "hard"} or (label_mode == "hard" and (formulation != "classification" or loss != "default")):
            raise ValueError("hard labels apply only to ordinary classification; pure BPR has no label mode")
        selected = model.get("variants", [None])
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
            raise ValueError("model variants must be a nonempty unique list")
        for variant in selected:
            if variant is not None and (not isinstance(variant, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", variant)):
                raise ValueError("variant names must contain only letters, digits, underscores or hyphens")
            if variant is not None and variant not in variants.get(estimator, {}):
                raise ValueError(f"undefined {estimator} variant: {variant}")
            effective_parameters = {
                **FORMULATION_PARAMETERS.get(formulation, {}).get(estimator, {}),
                **parameters.get(estimator, {}),
                **formulation_parameters.get(formulation, {}).get(estimator, {}),
                **(variants[estimator][variant] if variant is not None else {}),
                **model.get("parameters", {})}
            name = estimator + ("_hard" if label_mode == "hard" else "")
            if loss != "default":
                name += "_" + loss
            if variant is not None:
                name += "_" + variant
            directory = f"{formulation}/{name}"
            if directory in identifiers:
                raise ValueError(f"duplicate selector output: {directory}")
            identifiers.add(directory)
            jobs.append(dict(formulation=formulation, estimator=estimator, loss=loss,
                             label_mode=label_mode, variant=variant, parameters=effective_parameters, directory=directory))
    return jobs
