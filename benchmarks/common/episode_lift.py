"""Dependency-light target-object lift tracking for direct evaluations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EpisodeLiftTracker:
    """Track a sustained target-object height increase within one episode."""

    target_object: str
    initial_height_m: float
    height_threshold_m: float = 0.03
    hold_steps_required: int = 5

    def __post_init__(self) -> None:
        if not self.target_object:
            raise ValueError("target_object must be non-empty")
        for name, value in (
            ("initial_height_m", self.initial_height_m),
            ("height_threshold_m", self.height_threshold_m),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.height_threshold_m <= 0.0:
            raise ValueError("height_threshold_m must be positive")
        if isinstance(self.hold_steps_required, bool) or self.hold_steps_required <= 0:
            raise ValueError("hold_steps_required must be a positive integer")

        self.max_height_m = float(self.initial_height_m)
        self.max_height_delta_m = 0.0
        self.above_threshold_steps = 0
        self.consecutive_above_threshold_steps = 0
        self.max_consecutive_above_threshold_steps = 0
        self.first_above_threshold_step: int | None = None
        self.lift_confirmed_step: int | None = None

    @property
    def lifted(self) -> bool:
        return self.lift_confirmed_step is not None

    def update(
        self,
        *,
        step_index: int,
        object_height_m: float,
        contact_latched: bool,
    ) -> dict:
        """Consume one post-step height sample and return trace-ready state."""

        if step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not math.isfinite(object_height_m):
            raise ValueError(
                f"object_height_m must be finite, got {object_height_m!r}"
            )

        object_height_m = float(object_height_m)
        height_delta_m = object_height_m - float(self.initial_height_m)
        self.max_height_m = max(self.max_height_m, object_height_m)
        self.max_height_delta_m = max(self.max_height_delta_m, height_delta_m)

        above_threshold = bool(
            contact_latched and height_delta_m >= self.height_threshold_m
        )
        if above_threshold:
            self.above_threshold_steps += 1
            self.consecutive_above_threshold_steps += 1
            if self.first_above_threshold_step is None:
                self.first_above_threshold_step = int(step_index)
        else:
            self.consecutive_above_threshold_steps = 0

        self.max_consecutive_above_threshold_steps = max(
            self.max_consecutive_above_threshold_steps,
            self.consecutive_above_threshold_steps,
        )
        if (
            self.lift_confirmed_step is None
            and self.consecutive_above_threshold_steps >= self.hold_steps_required
        ):
            self.lift_confirmed_step = int(step_index)

        return {
            "target_object_height_m": object_height_m,
            "target_object_lift_delta_m": height_delta_m,
            "target_object_above_lift_threshold": above_threshold,
            "target_object_lift_hold_count": int(
                self.consecutive_above_threshold_steps
            ),
            "target_object_lifted": self.lifted,
        }

    def summary(self) -> dict:
        """Return the structured episode-level lift result."""

        return {
            "enabled": True,
            "target_object": self.target_object,
            "initial_height_m": float(self.initial_height_m),
            "max_height_m": float(self.max_height_m),
            "max_height_delta_m": float(self.max_height_delta_m),
            "height_threshold_m": float(self.height_threshold_m),
            "hold_steps_required": int(self.hold_steps_required),
            "above_threshold_steps": int(self.above_threshold_steps),
            "max_consecutive_above_threshold_steps": int(
                self.max_consecutive_above_threshold_steps
            ),
            "first_above_threshold_step": self.first_above_threshold_step,
            "lift_confirmed_step": self.lift_confirmed_step,
            "lifted": self.lifted,
        }
