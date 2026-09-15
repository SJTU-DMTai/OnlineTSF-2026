# -*- coding: utf-8 -*-
"""Prequential execution for online time-series forecasting.

The executor separates forecasting from feedback.  A forecast is always scored
against the parameters that produced it; an optional optimizer update happens
only when that forecast's target becomes available.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


LossFunction = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class MethodFeedback:
    """The immutable sample delivered to an online method."""

    index: int
    available_at: int
    context: Tensor
    target: Tensor
    prediction: Tensor


class OnlineMethod(Protocol):
    """Minimal lifecycle implemented by method-specific online learners."""

    def predict(self, context: Tensor) -> Tensor:
        """Return one forecast with shape ``[horizon, targets]``."""

    def on_feedback(self, feedback: MethodFeedback) -> float | None:
        """Consume an available label and return the update loss, if any."""


@dataclass(frozen=True)
class ForecastEmission:
    """One forecast emitted before its corresponding target is used."""

    index: int
    available_at: int
    prediction: Tensor


@dataclass(frozen=True)
class FeedbackEvent:
    """Metrics and update outcome once a forecast receives feedback."""

    index: int
    available_at: int
    mae: float
    mse: float
    cumulative_mae: float
    cumulative_mse: float
    adaptation_loss: float | None
    prediction: Tensor | None = None
    target: Tensor | None = None


@dataclass(frozen=True)
class OnlineStep:
    """The forecast emitted at one stream index and feedback delivered there."""

    emission: ForecastEmission
    feedback: tuple[FeedbackEvent, ...]


@dataclass(frozen=True)
class OnlineMetrics:
    """Aggregate prequential metrics for all feedback received so far."""

    forecasts_emitted: int
    feedback_received: int
    values_scored: int
    mae: float | None
    mse: float | None
    adaptation_steps: int
    mean_adaptation_loss: float | None


@dataclass(frozen=True)
class OnlineRun:
    """Result of replaying an ordered sequence of forecasting windows."""

    metrics: OnlineMetrics
    events: tuple[FeedbackEvent, ...]


@dataclass
class _PendingForecast:
    index: int
    available_at: int
    context: Tensor
    target: Tensor
    prediction: Tensor


class OnlineExecutor:
    """Run a model under delayed-feedback, prequential evaluation.

    ``feedback_delay`` is measured in emitted forecasting windows.  A value of
    zero implements immediate feedback (the protocol used by the FSNet paper).
    For a sliding-window forecast of an entire horizon, use the horizon as the
    delay: the final target value then becomes observable before the forecast at
    ``index + horizon`` is made.

    Pass either a PyTorch ``model`` or an ``OnlineMethod``.  A plain model can
    use the optional optimizer-based OGD path.  A method owns its prediction and
    feedback updates, while the executor retains responsibility for timing,
    scoring, and the pending-feedback queue.  Every event is scored before its
    feedback update.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        feedback_delay: int,
        method: OnlineMethod | None = None,
        optimizer: Optimizer | None = None,
        loss_fn: LossFunction | None = None,
        update_steps: int = 1,
        device: torch.device | str | None = None,
        keep_predictions: bool = False,
    ) -> None:
        if feedback_delay < 0:
            raise ValueError("feedback_delay must be non-negative")
        if (model is None) == (method is None):
            raise ValueError("provide exactly one of model or method")
        if method is not None and (optimizer is not None or loss_fn is not None):
            raise ValueError("an online method owns its optimizer and loss function")
        if method is not None and update_steps != 1:
            raise ValueError("an online method owns its update schedule")
        if update_steps <= 0:
            raise ValueError("update_steps must be positive")
        if optimizer is None and loss_fn is not None:
            raise ValueError("loss_fn requires an optimizer")
        if optimizer is not None and loss_fn is None:
            raise ValueError("optimizer requires loss_fn")

        self.model = model
        self.method = method
        self.feedback_delay = feedback_delay
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.update_steps = update_steps
        self.keep_predictions = keep_predictions
        self.device = (
            torch.device(device)
            if device is not None
            else self._model_device() if model is not None else None
        )

        self._pending: deque[_PendingForecast] = deque()
        self._events: list[FeedbackEvent] = []
        self._last_index: int | None = None
        self._forecasts_emitted = 0
        self._feedback_received = 0
        self._values_scored = 0
        self._absolute_error_sum = 0.0
        self._squared_error_sum = 0.0
        self._adaptation_steps = 0
        self._adaptation_loss_sum = 0.0

    def _model_device(self) -> torch.device:
        assert self.model is not None
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.model.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    @property
    def events(self) -> tuple[FeedbackEvent, ...]:
        """Feedback events in target-arrival order."""

        return tuple(self._events)

    @property
    def metrics(self) -> OnlineMetrics:
        """A snapshot of aggregate metrics without mutating executor state."""

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
        """Deliver due feedback, emit one forecast, and schedule its feedback."""

        if self._last_index is not None and index <= self._last_index:
            raise ValueError("stream indices must be strictly increasing")
        if context.ndim != 2 or target.ndim != 2:
            raise ValueError("context and target must have shapes [time, features]")

        feedback = list(self._deliver_through(index))
        prediction = self._predict(context)
        if prediction.shape != target.shape:
            raise ValueError(
                "model prediction and target shapes differ: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )

        available_at = index + self.feedback_delay
        self._pending.append(
            _PendingForecast(
                index=index,
                available_at=available_at,
                context=context.detach().to(device="cpu").clone(),
                target=target.detach().to(device="cpu").clone(),
                prediction=prediction.clone(),
            )
        )
        self._last_index = index
        self._forecasts_emitted += 1
        emission = ForecastEmission(index=index, available_at=available_at, prediction=prediction.clone())

        if self.feedback_delay == 0:
            feedback.extend(self._deliver_through(index))
        return OnlineStep(emission=emission, feedback=tuple(feedback))

    def flush(self) -> tuple[FeedbackEvent, ...]:
        """Deliver all remaining delayed feedback in chronological order.

        This is useful after replaying a finite offline dataset.  It does not
        alter any forecast already emitted, but it lets final metrics include
        every target in the replay.
        """

        feedback: list[FeedbackEvent] = []
        while self._pending:
            feedback.extend(self._deliver_through(self._pending[0].available_at))
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
        """Return executor state needed to resume a delayed-feedback stream.

        Model or method state is intentionally separate: save the underlying
        model and optimizer through their normal ``state_dict`` methods in the
        same checkpoint.
        """

        return {
            "feedback_delay": self.feedback_delay,
            "update_steps": self.update_steps,
            "keep_predictions": self.keep_predictions,
            "last_index": self._last_index,
            "pending": [
                {
                    "index": item.index,
                    "available_at": item.available_at,
                    "context": item.context.clone(),
                    "target": item.target.clone(),
                    "prediction": item.prediction.clone(),
                }
                for item in self._pending
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

        if state["feedback_delay"] != self.feedback_delay:
            raise ValueError("checkpoint feedback_delay differs from this executor")
        if state["update_steps"] != self.update_steps:
            raise ValueError("checkpoint update_steps differs from this executor")
        if state["keep_predictions"] != self.keep_predictions:
            raise ValueError("checkpoint keep_predictions differs from this executor")

        self._last_index = state["last_index"]
        self._pending = deque(
            _PendingForecast(
                index=item["index"],
                available_at=item["available_at"],
                context=item["context"].detach().to(device="cpu").clone(),
                target=item["target"].detach().to(device="cpu").clone(),
                prediction=item["prediction"].detach().to(device="cpu").clone(),
            )
            for item in state["pending"]
        )
        self._events = [self._event_from_state(event) for event in state["events"]]
        metrics = state["metrics"]
        self._forecasts_emitted = metrics["forecasts_emitted"]
        self._feedback_received = metrics["feedback_received"]
        self._values_scored = metrics["values_scored"]
        self._absolute_error_sum = metrics["absolute_error_sum"]
        self._squared_error_sum = metrics["squared_error_sum"]
        self._adaptation_steps = metrics["adaptation_steps"]
        self._adaptation_loss_sum = metrics["adaptation_loss_sum"]

    def _predict(self, context: Tensor) -> Tensor:
        if self.method is not None:
            prediction = self.method.predict(context.detach().clone())
        else:
            assert self.model is not None and self.device is not None
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(context.unsqueeze(0).to(self.device)).squeeze(0)
        return prediction.detach().to(device="cpu").clone()

    def _deliver_through(self, index: int) -> tuple[FeedbackEvent, ...]:
        feedback: list[FeedbackEvent] = []
        while self._pending and self._pending[0].available_at <= index:
            feedback.append(self._consume_feedback(self._pending.popleft()))
        return tuple(feedback)

    def _consume_feedback(self, pending: _PendingForecast) -> FeedbackEvent:
        error = pending.prediction - pending.target
        absolute_error = error.abs()
        squared_error = error.square()
        self._absolute_error_sum += absolute_error.sum().item()
        self._squared_error_sum += squared_error.sum().item()
        self._values_scored += error.numel()
        self._feedback_received += 1

        adaptation_loss = self._adapt(pending)
        metrics = self.metrics
        event = FeedbackEvent(
            index=pending.index,
            available_at=pending.available_at,
            mae=absolute_error.mean().item(),
            mse=squared_error.mean().item(),
            cumulative_mae=metrics.mae if metrics.mae is not None else 0.0,
            cumulative_mse=metrics.mse if metrics.mse is not None else 0.0,
            adaptation_loss=adaptation_loss,
            prediction=pending.prediction if self.keep_predictions else None,
            target=pending.target if self.keep_predictions else None,
        )
        self._events.append(event)
        return event

    def _adapt(self, pending: _PendingForecast) -> float | None:
        if self.method is not None:
            loss = self.method.on_feedback(
                MethodFeedback(
                    index=pending.index,
                    available_at=pending.available_at,
                    context=pending.context.clone(),
                    target=pending.target.clone(),
                    prediction=pending.prediction.clone(),
                )
            )
            if loss is not None:
                self._adaptation_steps += 1
                self._adaptation_loss_sum += loss
            return loss

        if self.optimizer is None or self.loss_fn is None:
            return None

        assert self.model is not None and self.device is not None
        self.model.train()
        context = pending.context.unsqueeze(0).to(self.device)
        target = pending.target.unsqueeze(0).to(self.device)
        losses: list[float] = []
        for _ in range(self.update_steps):
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(context)
            loss = self.loss_fn(prediction, target)
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().item())

        self._adaptation_steps += self.update_steps
        self._adaptation_loss_sum += sum(losses)
        return sum(losses) / len(losses)

    @staticmethod
    def _event_state(event: FeedbackEvent) -> dict[str, Any]:
        state = asdict(event)
        for name in ("prediction", "target"):
            value = state[name]
            if value is not None:
                state[name] = value.clone()
        return state

    @staticmethod
    def _event_from_state(state: dict[str, Any]) -> FeedbackEvent:
        return FeedbackEvent(
            index=state["index"],
            available_at=state["available_at"],
            mae=state["mae"],
            mse=state["mse"],
            cumulative_mae=state["cumulative_mae"],
            cumulative_mse=state["cumulative_mse"],
            adaptation_loss=state["adaptation_loss"],
            prediction=state["prediction"],
            target=state["target"],
        )
