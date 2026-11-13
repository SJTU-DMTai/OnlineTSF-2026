"""Translate declared units to each method's existing training mechanisms."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..methods import DSOFMethod, FSNetMethod, OneNetMethod
from ..online import MethodFeedback
from .capabilities import UpdateCapabilities, UpdateUnit, UnitSchedule
from .optimizers import OptimizerRates
from .batches import FeedbackBatches


@dataclass
class PreparedFeedback:
    feedback: MethodFeedback
    opportunities: tuple[str, ...]
    expert_predictions: tuple[Tensor, Tensor] | None = None


class OneNetAdapter:
    capabilities = UpdateCapabilities(
        (UpdateUnit("time", "complete_feedback", True),
         UpdateUnit("variable", "complete_feedback", True),
         UpdateUnit("decision", "complete_feedback", True),
         UpdateUnit("long", "complete_feedback", True),
         UpdateUnit("short", "complete_feedback")))

    def __init__(self, method: OneNetMethod) -> None:
        self.method = method
        self.batches = FeedbackBatches()
        self.rates = OptimizerRates([
            (method.expert_optimizer, {"time": list(method.model.cross_time.parameters()),
                                       "variable": list(method.model.cross_variable.parameters())}),
            (method.decision_optimizer, {"decision": list(method.model.decision.parameters())})])
        self.native_long_rate = method.weight_learning_rate

    def configure(self, learning_rate_scales) -> None:
        self.rates.apply(learning_rate_scales)
        self.method.weight_learning_rate = self.native_long_rate * learning_rate_scales.get("long", 1.0)

    def prepare(self, feedback: MethodFeedback, *, training: bool) -> PreparedFeedback:
        if not feedback.observed_mask.all():
            raise ValueError("OneNet requires complete-horizon feedback")
        predictions = self.method._pending_predictions.popleft()
        return PreparedFeedback(feedback, ("time", "variable", "decision", "long", "short"), predictions)

    def apply(self, prepared: PreparedFeedback, due: frozenset[str], schedule: UnitSchedule) -> float | None:
        self.batches.append(prepared, (name for name in prepared.opportunities if name in schedule.active))
        return self.apply_batches(self.batches.take(name for name in prepared.opportunities if name in due), schedule)

    def apply_batches(self, batches, schedule: UnitSchedule) -> float | None:
        method = self.method
        method.last_gradient_steps = method.last_egd_steps = 0
        losses = []
        for units, items in batches:
            context = torch.stack([item.feedback.context for item in items]).to(method.device)
            target = torch.stack([item.feedback.target for item in items]).to(method.device)
            cross_time = torch.stack([item.expert_predictions[0] for item in items]).to(method.device)
            cross_variable = torch.stack([item.expert_predictions[1] for item in items]).to(method.device)
            long_rate = method.weight_learning_rate
            method.weight_learning_rate = long_rate * len(items)
            try:
                for _ in range(method.n_inner):
                    method._update(context, target, cross_time, cross_variable, units & {"time", "variable", "decision"},
                                   update_long="long" in units, refresh_short="short" in units)
            finally:
                method.weight_learning_rate = long_rate
            prediction = torch.stack([item.feedback.prediction for item in items]).to(method.device)
            losses.append(method.loss_fn(prediction, target).detach().item())
        return sum(losses) / len(losses) if losses else None

    def predict(self, context: Tensor, schedule: UnitSchedule) -> Tensor:
        return self.method.predict(context)


class FSNetAdapter:
    capabilities = UpdateCapabilities(
        (UpdateUnit("base", "complete_feedback", True),
         UpdateUnit("adapter", "complete_feedback", True),
         UpdateUnit("head", "complete_feedback", True),
         UpdateUnit("gradient_memory", "complete_feedback"),
         UpdateUnit("adaptation_memory", "prediction_and_native_inner_forward")))

    def __init__(self, method: FSNetMethod) -> None:
        self.method = method
        self.batches = FeedbackBatches()
        self.rates = OptimizerRates([(method.optimizer, method.parameter_groups())])

    def configure(self, learning_rate_scales) -> None:
        self.rates.apply(learning_rate_scales)

    def prepare(self, feedback: MethodFeedback, *, training: bool) -> PreparedFeedback:
        if not feedback.observed_mask.all():
            raise ValueError("FSNet requires complete-horizon feedback")
        return PreparedFeedback(feedback, ("base", "adapter", "head", "gradient_memory"))

    def apply(self, prepared: PreparedFeedback, due: frozenset[str], schedule: UnitSchedule) -> float | None:
        self.batches.append(prepared, (name for name in prepared.opportunities if name in schedule.active))
        return self.apply_batches(self.batches.take(name for name in prepared.opportunities if name in due), schedule)

    def apply_batches(self, batches, schedule: UnitSchedule) -> float | None:
        method = self.method
        gradient_steps = 0
        losses = []
        for units, items in batches:
            context = torch.stack([item.feedback.context for item in items])
            target = torch.stack([item.feedback.target for item in items])
            losses.append(method.update_batch(context, target, units & {"base", "adapter", "head"},
                                               gradient_memory="gradient_memory" in units,
                                               memory_schedule=schedule.adaptation_due))
            gradient_steps += method.last_gradient_steps
        method.last_gradient_steps, method.last_egd_steps = gradient_steps, 0
        return sum(losses) / len(losses) if losses else None

    def predict(self, context: Tensor, schedule: UnitSchedule) -> Tensor:
        return self.method.predict(context, advance_state=schedule.adaptation_due())


class DSOFAdapter:
    capabilities = UpdateCapabilities(
        (UpdateUnit("teacher", "native_replay_trigger", True),
         UpdateUnit("student", "native_replay_trigger", True),
         UpdateUnit("online", "first_step_feedback", True)))

    def __init__(self, method: DSOFMethod) -> None:
        self.method = method
        self.batches = FeedbackBatches()
        self.rates = OptimizerRates([
            (method.teacher_optimizer, {"teacher": list(method.model.teacher.parameters())}),
            (method.student_optimizer, {"student": list(method.model.student.parameters())}),
            (method.online_optimizer, {"online": list(method.model.student.parameters())})])

    def configure(self, learning_rate_scales) -> None:
        self.rates.apply(learning_rate_scales)

    def prepare(self, feedback: MethodFeedback, *, training: bool) -> PreparedFeedback:
        method = self.method
        method._record_forecast_feedback(feedback)
        # Native replay needs complete windows even during frozen intervals.
        # Retain only outstanding partial labels and the method's bounded replay.
        values, observed = method._partial.pop(feedback.index, (torch.zeros_like(feedback.target),
                                                               torch.zeros_like(feedback.observed_mask)))
        values[feedback.observed_mask] = feedback.target[feedback.observed_mask]
        observed |= feedback.observed_mask
        opportunities = []
        if observed.all():
            method.replay.append((feedback.context.detach().cpu().clone(), values.detach().cpu().clone()))
            if training and len(method.replay) >= method.replay_batch_size:
                method._complete_count += 1
                if (method._complete_count - 1) % method.replay_frequency == 0:
                    opportunities.extend(("teacher", "student"))
        else:
            method._partial[feedback.index] = values, observed
        if feedback.observed_mask[0].all() and training:
            opportunities.append("online")
        return PreparedFeedback(feedback, tuple(opportunities))

    def apply(self, prepared: PreparedFeedback, due: frozenset[str], schedule: UnitSchedule) -> float | None:
        method = self.method
        method.last_gradient_steps = 0
        losses = []
        if due & {"teacher", "student"}:
            for _ in range(method.replay_epochs):
                indices = torch.randperm(len(method.replay))[:method.replay_batch_size].tolist()
                context = torch.stack([method.replay[index][0] for index in indices]).to(method.device)
                target = torch.stack([method.replay[index][1] for index in indices]).to(method.device)
                losses.append(method._supervised_update(context, target, due & {"teacher", "student"}))
        if "online" in prepared.opportunities and "online" in schedule.active:
            self.batches.append(prepared, ("online",))
        gradient_steps = method.last_gradient_steps
        online_loss = self.apply_batches(self.batches.take(("online",) if "online" in due else ()), schedule)
        method.last_gradient_steps += gradient_steps
        if online_loss is not None:
            losses.append(online_loss)
        return sum(losses) / len(losses) if losses else None

    def apply_batches(self, batches, schedule: UnitSchedule) -> float | None:
        method = self.method
        method.last_gradient_steps = 0
        losses = []
        for _, items in batches:
            context = torch.stack([item.feedback.context for item in items])
            current = torch.stack([item.feedback.current_context for item in items])
            first_target = torch.stack([item.feedback.target[:1] for item in items])
            losses.append(method._fast_update_batch(context, current, first_target))
        return sum(losses) / len(losses) if losses else None

    def predict(self, context: Tensor, schedule: UnitSchedule) -> Tensor:
        return self.method.predict(context)


def build_adapter(method) -> OneNetAdapter | FSNetAdapter | DSOFAdapter:
    if isinstance(method, OneNetMethod):
        return OneNetAdapter(method)
    if isinstance(method, FSNetMethod):
        return FSNetAdapter(method)
    if isinstance(method, DSOFMethod):
        return DSOFAdapter(method)
    raise TypeError("update actions support OneNetMethod, FSNetMethod and DSOFMethod")
