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

    def native_action(self, duration: int = 1) -> UpdateAction:
        return UpdateAction(units=self.unit_names, duration=duration)

    def validate(self, action: UpdateAction) -> None:
        if action.units - self.unit_names:
            raise ValueError(f"unknown update units: {sorted(action.units - self.unit_names)}")
        allowed_rates = {unit.name for unit in self.units if unit.has_learning_rate} & action.units
        if action.learning_rate_scales.keys() - allowed_rates:
            raise ValueError("learning rate scales specified for units without a step size")


class UnitSchedule:
    """Count each unit's native legal opportunities, independently of duration."""

    def __init__(self) -> None:
        self.periods: dict[str, int] = {}
        self.counts: dict[str, int] = {}

    def activate(self, action: UpdateAction) -> None:
        periods = {name: action.periods.get(name, 1) for name in action.units}
        self.counts = {name: self.counts.get(name, 0) if self.periods.get(name) == period else 0
                       for name, period in periods.items()}
        self.periods = periods

    def due(self, name: str) -> bool:
        if name not in self.periods:
            return False
        count = self.counts[name]
        self.counts[name] += 1
        return count % self.periods[name] == 0

    def adaptation_due(self) -> bool:
        """FSNet calls this only on forwards that natively advance memory."""
        return self.due("adaptation_memory")
