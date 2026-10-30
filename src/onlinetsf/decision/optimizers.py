"""Give native units independent rates without replacing optimizer statistics."""

from collections.abc import Mapping

from torch import nn
from torch.optim import Optimizer


class OptimizerRates:
    def __init__(self, specifications: list[tuple[Optimizer, dict[str, list[nn.Parameter]]]]) -> None:
        self.bindings: list[tuple[str, dict, float]] = []
        for optimizer, units in specifications:
            owners = {id(parameter): name for name, parameters in units.items() for parameter in parameters}
            groups = []
            for original in optimizer.param_groups:
                split: dict[str, list[nn.Parameter]] = {}
                for parameter in original["params"]:
                    split.setdefault(owners[id(parameter)], []).append(parameter)
                for name, parameters in split.items():
                    group = {**original, "params": parameters}
                    groups.append(group)
                    self.bindings.append((name, group, original["lr"]))
            optimizer.param_groups[:] = groups

    def apply(self, scales: Mapping[str, float]) -> None:
        for name, group, native in self.bindings:
            group["lr"] = native * scales.get(name, 1.0)
