"""Method-declared update units and action legality, independent of scheduling."""

from dataclasses import dataclass

from .actions import UpdateAction


@dataclass(frozen=True)
class UpdateUnit:
    name: str
    opportunity: str
    has_learning_rate: bool = False


@dataclass(frozen=True)
class UpdateCapabilities:
    units: tuple[UpdateUnit, ...]

    @property
    def unit_names(self) -> frozenset[str]:
        return frozenset(unit.name for unit in self.units)

    def native_action(self) -> UpdateAction:
        return UpdateAction(units=self.unit_names)

    def validate(self, action: UpdateAction) -> None:
        if action.units - self.unit_names:
            raise ValueError(f"unknown update units: {sorted(action.units - self.unit_names)}")
        allowed_rates = {unit.name for unit in self.units if unit.has_learning_rate} & action.units
        if action.learning_rate_scales.keys() - allowed_rates:
            raise ValueError("learning rate scales specified for units without a step size")


class UnitSchedule:
    """Enable selected update units on each native update opportunity."""

    def __init__(self) -> None:
        self.active: frozenset[str] = frozenset()
        self.counts: dict[str, int] = {}

    def activate(self, action: UpdateAction) -> None:
        self.active = action.units

    def due(self, name: str) -> bool:
        if name not in self.active:
            return False
        self.counts[name] = self.counts.get(name, 0) + 1
        return True

    def adaptation_due(self) -> bool:
        """FSNet calls this only on forwards that natively advance memory."""
        return self.due("adaptation_memory")
