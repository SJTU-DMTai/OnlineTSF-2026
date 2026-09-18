# -*- coding: utf-8 -*-
"""Prequential scheduling, delayed feedback, and metrics for online forecasting."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class MethodFeedback:
    """One label arrival delivered to a method-specific online learner."""

    index: int
    available_at: int
    context: Tensor
    target: Tensor
    observed_mask: Tensor
    prediction: Tensor


class OnlineMethod(Protocol):
    """Minimal lifecycle implemented by method-specific online learners."""

    def predict(self, context: Tensor) -> Tensor:
        """Return one forecast with shape ``[horizon, targets]``."""

    def on_feedback(self, feedback: MethodFeedback) -> float | None:
        """Consume newly observed target values and return an update loss."""


@dataclass(frozen=True)
class ForecastEmission:
    """One forecast emitted before its corresponding label values are used."""

    index: int
    available_at: int | tuple[int, ...]
    prediction: Tensor


@dataclass(frozen=True)
class FeedbackEvent:
    """Metrics and update outcome for one complete or partial label arrival."""

    index: int
    available_at: int
    observed_values: int
    mae: float
    mse: float
    cumulative_mae: float
    cumulative_mse: float
    adaptation_loss: float | None
    prediction: Tensor | None = None
    target: Tensor | None = None
    observed_mask: Tensor | None = None
    features: Tensor | None = None


@dataclass(frozen=True)
class OnlineStep:
    """The forecast emitted at one stream index and feedback delivered there."""

    emission: ForecastEmission
    feedback: tuple[FeedbackEvent, ...]


@dataclass(frozen=True)
class OnlineMetrics:
    """Aggregate prequential metrics for all observed forecast values."""

    forecasts_emitted: int
    feedback_received: int
    values_scored: int
    mae: float | None
    mse: float | None
    adaptation_steps: int
    mean_adaptation_loss: float | None
    setup_seconds: float | None = None
    offline_training_seconds: float | None = None
    online_evaluation_seconds: float | None = None
    drift_detection_seconds: float | None = None
    total_seconds: float | None = None


@dataclass(frozen=True)
class OnlineRun:
    """Result of replaying an ordered sequence of forecasting windows."""

    metrics: OnlineMetrics
    events: tuple[FeedbackEvent, ...]


@dataclass
class _PendingFeedback:
    """One set of target coordinates that becomes observable at one index."""

    index: int
    available_at: int
    order: int
    context: Tensor
    target: Tensor
    prediction: Tensor
    observed_mask: Tensor


class OnlineExecutor:
    """Schedule method-neutral prequential forecasting with delayed labels.

    ``feedback_delay`` can be a scalar, making the complete forecast target
    available after that many emitted windows, or a sequence with one delay per
    horizon step.  The latter emits partial feedback events whose
    ``observed_mask`` identifies exactly which target values are available.
    Methods decide how to train from that feedback; this executor only controls
    timing, prequential scoring, metrics, and checkpointable pending state.
    """

    def __init__(
        self,
        method: OnlineMethod,
        *,
        feedback_delay: int | Sequence[int],
        keep_predictions: bool = False,
    ) -> None:
        if isinstance(feedback_delay, int):
            if feedback_delay < 0:
                raise ValueError("feedback_delay must be non-negative")
            self.feedback_delay: int | tuple[int, ...] = feedback_delay
        else:
            delays = tuple(feedback_delay)
            if not delays or any(not isinstance(delay, int) or delay < 0 for delay in delays):
                raise ValueError("feedback_delay sequence must contain non-negative integers")
            self.feedback_delay = delays

        self.method = method
        self.keep_predictions = keep_predictions
        self._pending: list[tuple[int, int, _PendingFeedback]] = []
        self._events: list[FeedbackEvent] = []
        self._last_index: int | None = None
        self._next_pending_order = 0
        self._forecasts_emitted = 0
        self._feedback_received = 0
        self._values_scored = 0
        self._absolute_error_sum = 0.0
        self._squared_error_sum = 0.0
        self._adaptation_steps = 0
        self._adaptation_loss_sum = 0.0

    @property
    def events(self) -> tuple[FeedbackEvent, ...]:
        """Feedback events in label-arrival order."""

        return tuple(self._events)

    @property
    def metrics(self) -> OnlineMetrics:
        """Return aggregate prequential metrics without mutating state."""

        if self._values_scored == 0:
            mae = None
            mse = None
        else:
            mae = self._absolute_error_sum / self._values_scored
            mse = self._squared_error_sum / self._values_scored
        mean_adaptation_loss = (
            self._adaptation_loss_sum / self._adaptation_steps
            if self._adaptation_steps
            else None
        )
        return OnlineMetrics(
            forecasts_emitted=self._forecasts_emitted,
            feedback_received=self._feedback_received,
            values_scored=self._values_scored,
            mae=mae,
            mse=mse,
            adaptation_steps=self._adaptation_steps,
            mean_adaptation_loss=mean_adaptation_loss,
        )

    def step(self, index: int, context: Tensor, target: Tensor) -> OnlineStep:
        """Deliver due labels, forecast, then schedule this forecast's labels."""

        if self._last_index is not None and index <= self._last_index:
            raise ValueError("stream indices must be strictly increasing")
        if context.ndim != 2 or target.ndim != 2:
            raise ValueError("context and target must have shapes [time, features]")

        feedback = list(self._deliver_through(index))
        prediction = self.method.predict(context.detach().clone()).detach().to(device="cpu").clone()
        if prediction.shape != target.shape:
            raise ValueError(
                "method prediction and target shapes differ: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )

        emission_available_at = self._schedule_feedback(index, context, target, prediction)
        self._last_index = index
        self._forecasts_emitted += 1
        emission = ForecastEmission(
            index=index,
            available_at=emission_available_at,
            prediction=prediction.clone(),
        )
        feedback.extend(self._deliver_through(index))
        return OnlineStep(emission=emission, feedback=tuple(feedback))

    def flush(self) -> tuple[FeedbackEvent, ...]:
        """Deliver every remaining label event in chronological order."""

        feedback: list[FeedbackEvent] = []
        while self._pending:
            feedback.extend(self._deliver_through(self._pending[0][0]))
        return tuple(feedback)

    def run_dataset(
        self,
        dataset: Sequence[tuple[Tensor, Tensor]],
        *,
        start: int = 0,
        stop: int | None = None,
        flush: bool = True,
    ) -> OnlineRun:
        """Replay an ordered slice of a sliding-window dataset."""

        stop = len(dataset) if stop is None else stop
        if start < 0 or stop < start or stop > len(dataset):
            raise ValueError("start and stop must select a valid dataset slice")

        first_event = len(self._events)
        for index in range(start, stop):
            context, target = dataset[index]
            self.step(index, context, target)
        if flush:
            self.flush()
        return OnlineRun(metrics=self.metrics, events=tuple(self._events[first_event:]))

    def state_dict(self) -> dict[str, Any]:
        """Return scheduling, metrics, and pending feedback for checkpointing."""

        return {
            "feedback_delay": self.feedback_delay,
            "keep_predictions": self.keep_predictions,
            "last_index": self._last_index,
            "next_pending_order": self._next_pending_order,
            "pending": [
                {
                    "index": item.index,
                    "available_at": item.available_at,
                    "order": item.order,
                    "context": item.context.clone(),
                    "target": item.target.clone(),
                    "prediction": item.prediction.clone(),
                    "observed_mask": item.observed_mask.clone(),
                }
                for _, _, item in sorted(self._pending)
            ],
            "events": [self._event_state(event) for event in self._events],
            "metrics": {
                "forecasts_emitted": self._forecasts_emitted,
                "feedback_received": self._feedback_received,
                "values_scored": self._values_scored,
                "absolute_error_sum": self._absolute_error_sum,
                "squared_error_sum": self._squared_error_sum,
                "adaptation_steps": self._adaptation_steps,
                "adaptation_loss_sum": self._adaptation_loss_sum,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""

        saved_delay = state["feedback_delay"]
        if isinstance(saved_delay, list):
            saved_delay = tuple(saved_delay)
        if saved_delay != self.feedback_delay:
            raise ValueError("checkpoint feedback_delay differs from this executor")
        if state["keep_predictions"] != self.keep_predictions:
            raise ValueError("checkpoint keep_predictions differs from this executor")

        self._last_index = state["last_index"]
        self._next_pending_order = state["next_pending_order"]
        self._pending = []
        for item in state["pending"]:
            pending = _PendingFeedback(
                index=item["index"],
                available_at=item["available_at"],
                order=item["order"],
                context=item["context"].detach().to(device="cpu").clone(),
                target=item["target"].detach().to(device="cpu").clone(),
                prediction=item["prediction"].detach().to(device="cpu").clone(),
                observed_mask=item["observed_mask"].detach().to(device="cpu").clone(),
            )
            heapq.heappush(self._pending, (pending.available_at, pending.order, pending))
        self._events = [self._event_from_state(event) for event in state["events"]]
        metrics = state["metrics"]
        self._forecasts_emitted = metrics["forecasts_emitted"]
        self._feedback_received = metrics["feedback_received"]
        self._values_scored = metrics["values_scored"]
        self._absolute_error_sum = metrics["absolute_error_sum"]
        self._squared_error_sum = metrics["squared_error_sum"]
        self._adaptation_steps = metrics["adaptation_steps"]
        self._adaptation_loss_sum = metrics["adaptation_loss_sum"]

    def _schedule_feedback(
        self,
        index: int,
        context: Tensor,
        target: Tensor,
        prediction: Tensor,
    ) -> int | tuple[int, ...]:
        if isinstance(self.feedback_delay, int):
            delays = (self.feedback_delay,)
            masks = (torch.ones_like(target, dtype=torch.bool),)
            emission_available_at: int | tuple[int, ...] = index + self.feedback_delay
        else:
            if len(self.feedback_delay) != target.shape[0]:
                raise ValueError("feedback_delay sequence length must equal the forecast horizon")
            by_delay: dict[int, Tensor] = {}
            for horizon_index, delay in enumerate(self.feedback_delay):
                if delay not in by_delay:
                    by_delay[delay] = torch.zeros_like(target, dtype=torch.bool)
                by_delay[delay][horizon_index] = True
            delays = tuple(sorted(by_delay))
            masks = tuple(by_delay[delay] for delay in delays)
            emission_available_at = tuple(index + delay for delay in self.feedback_delay)

        stored_context = context.detach().to(device="cpu").clone()
        stored_target = target.detach().to(device="cpu").clone()
        for delay, observed_mask in zip(delays, masks, strict=True):
            pending = _PendingFeedback(
                index=index,
                available_at=index + delay,
                order=self._next_pending_order,
                context=stored_context,
                target=stored_target,
                prediction=prediction.clone(),
                observed_mask=observed_mask.detach().to(device="cpu").clone(),
            )
            heapq.heappush(self._pending, (pending.available_at, pending.order, pending))
            self._next_pending_order += 1
        return emission_available_at

    def _deliver_through(self, index: int) -> tuple[FeedbackEvent, ...]:
        feedback: list[FeedbackEvent] = []
        while self._pending and self._pending[0][0] <= index:
            _, _, pending = heapq.heappop(self._pending)
            feedback.append(self._consume_feedback(pending))
        return tuple(feedback)

    def _consume_feedback(self, pending: _PendingFeedback) -> FeedbackEvent:
        observed_target = pending.target.masked_fill(~pending.observed_mask, float("nan"))
        error = pending.prediction[pending.observed_mask] - pending.target[pending.observed_mask]
        absolute_error = error.abs()
        squared_error = error.square()
        observed_values = error.numel()
        self._absolute_error_sum += absolute_error.sum().item()
        self._squared_error_sum += squared_error.sum().item()
        self._values_scored += observed_values
        self._feedback_received += 1

        adaptation_loss = self.method.on_feedback(
            MethodFeedback(
                index=pending.index,
                available_at=pending.available_at,
                context=pending.context.clone(),
                target=observed_target,
                observed_mask=pending.observed_mask.clone(),
                prediction=pending.prediction.clone(),
            )
        )
        if adaptation_loss is not None:
            self._adaptation_steps += 1
            self._adaptation_loss_sum += adaptation_loss
        metrics = self.metrics
        event = FeedbackEvent(
            index=pending.index,
            available_at=pending.available_at,
            observed_values=observed_values,
            mae=absolute_error.mean().item(),
            mse=squared_error.mean().item(),
            cumulative_mae=metrics.mae if metrics.mae is not None else 0.0,
            cumulative_mse=metrics.mse if metrics.mse is not None else 0.0,
            adaptation_loss=adaptation_loss,
            prediction=pending.prediction if self.keep_predictions else None,
            target=observed_target if self.keep_predictions else None,
            observed_mask=pending.observed_mask if self.keep_predictions else None,
            features=pending.context[-1] if self.keep_predictions else None,
        )
        self._events.append(event)
        return event

    @staticmethod
    def _event_state(event: FeedbackEvent) -> dict[str, Any]:
        state = asdict(event)
        for name in ("prediction", "target", "observed_mask"):
            value = state[name]
            if value is not None:
                state[name] = value.clone()
        return state

    @staticmethod
    def _event_from_state(state: dict[str, Any]) -> FeedbackEvent:
        return FeedbackEvent(
            index=state["index"],
            available_at=state["available_at"],
            observed_values=state["observed_values"],
            mae=state["mae"],
            mse=state["mse"],
            cumulative_mae=state["cumulative_mae"],
            cumulative_mse=state["cumulative_mse"],
            adaptation_loss=state["adaptation_loss"],
            prediction=state["prediction"],
            target=state["target"],
            observed_mask=state["observed_mask"],
            features=state.get("features"),
        )
