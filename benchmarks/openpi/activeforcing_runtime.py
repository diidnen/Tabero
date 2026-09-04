"""Small, dependency-light ActiveForcing runtime boundary.

ActiveForcing selects one scalar physical force setpoint.  This module keeps
that boundary separate from OpenPI action decoding and delegates execution to
Tabero's native ``ForcePositionAction``.  It intentionally does not import
IsaacLab, Torch, or the ActiveForcing training runner, so the interface can be
tested offline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Protocol


PROTOCOL_VERSION = "ACTIVEFORCING_FULL_CLAIM_V2_UTILITY"
TELEMETRY_SCHEMA_VERSION = 1


def _coerce_scalar(value: Any, *, field_name: str) -> float:
    """Convert a genuine scalar model output to a finite non-negative float."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a numeric scalar, not bool")
    if isinstance(value, Real):
        result = float(value)
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        if getattr(tensor, "ndim", 0) != 0:
            raise TypeError(f"{field_name} must be a scalar, got ndim={tensor.ndim}")
        result = float(tensor.item())
    elif hasattr(value, "ndim") and hasattr(value, "item"):
        if getattr(value, "ndim", 0) != 0:
            raise TypeError(f"{field_name} must be a scalar, got ndim={value.ndim}")
        result = float(value.item())
    else:
        raise TypeError(f"{field_name} must be a numeric scalar, got {type(value).__name__}")
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative, got {result!r}")
    return result


@dataclass(frozen=True)
class ActiveForcingDecision:
    """The only value crossing the ActiveForcing model/executor boundary."""

    target_force_n: float
    source: str = "model"
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_force_n",
            _coerce_scalar(self.target_force_n, field_name="target_force_n"),
        )
        if not self.source:
            raise ValueError("decision source must be non-empty")

    @classmethod
    def from_model_output(cls, output: Any) -> "ActiveForcingDecision":
        """Normalize a model result while requiring an explicit scalar field."""

        if isinstance(output, cls):
            return output
        payload = output
        if isinstance(output, Mapping) and isinstance(output.get("activeforcing"), Mapping):
            payload = output["activeforcing"]
        if isinstance(payload, Mapping):
            if "target_force_n" not in payload:
                raise KeyError("ActiveForcing model output must contain target_force_n")
            source = str(payload.get("source", "model"))
            return cls(payload["target_force_n"], source=source)
        if hasattr(payload, "target_force_n"):
            return cls(getattr(payload, "target_force_n"))
        raise TypeError("ActiveForcing model output must be a mapping/object with target_force_n")


class ActiveForcingModel(Protocol):
    """Protocol implemented by a runtime ActiveForcing model/provider."""

    def predict_target_force_n(
        self, *, inference_response: Any, context: Mapping[str, Any]
    ) -> Any:
        """Return a model output containing ``target_force_n: float``."""


class ResponseTargetForceModel:
    """Read the scalar decision emitted alongside an OpenPI response.

    The OpenPI action chunk remains untouched.  A compatible policy server can
    return either ``{"target_force_n": 4.0, "actions": ...}`` or
    ``{"activeforcing": {"target_force_n": 4.0}, "actions": ...}``.
    """

    def predict_target_force_n(
        self, *, inference_response: Any, context: Mapping[str, Any]
    ) -> Any:
        del context
        return ActiveForcingDecision.from_model_output(inference_response)


class JsonTargetForceModel:
    """Explicit offline smoke provider; never used unless a path is supplied."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        parsed = ActiveForcingDecision.from_model_output(payload)
        self.decision = ActiveForcingDecision(
            parsed.target_force_n, source=f"json:{self.path}"
        )

    def predict_target_force_n(
        self, *, inference_response: Any, context: Mapping[str, Any]
    ) -> ActiveForcingDecision:
        del inference_response, context
        return self.decision


def select_expected_utility_force(
    candidates_n: list[float] | tuple[float, ...],
    success_probabilities: list[float] | tuple[float, ...],
    *,
    fmax_task_n: float,
) -> ActiveForcingDecision:
    """Select a force using the authoritative expected-utility definition."""

    if len(candidates_n) == 0 or len(candidates_n) != len(success_probabilities):
        raise ValueError("candidates_n and success_probabilities must have equal non-zero length")
    fmax = _coerce_scalar(fmax_task_n, field_name="fmax_task_n")
    if fmax <= 0.0:
        raise ValueError("fmax_task_n must be positive")
    scored: list[tuple[float, float, float]] = []
    for force, probability in zip(candidates_n, success_probabilities):
        f = _coerce_scalar(force, field_name="candidate_force_n")
        p = _coerce_scalar(probability, field_name="success_probability")
        if p > 1.0:
            raise ValueError("success_probability must be in [0, 1]")
        utility = p * ((fmax - f) / fmax) + (1.0 - p) * -1.0
        scored.append((utility, f, p))
    # Max utility; on ties choose the lower force as required by protocol.
    utility, force, _ = max(scored, key=lambda item: (item[0], -item[1]))
    return ActiveForcingDecision(
        target_force_n=force,
        source=f"expected_utility:{utility:.9g}",
    )


def validate_native_13d_action(action: Any) -> None:
    """Fail closed unless the native hybrid action has the expected 13D shape."""

    shape = getattr(action, "shape", None)
    if shape is None or len(shape) not in (1, 2) or int(shape[-1]) != 13:
        raise ValueError(f"expected native Tabero hybrid action shape (13,) or (N,13), got {shape}")


@dataclass
class ActiveForcingExecutor:
    """Delegate scalar force execution to one native Tabero action term."""

    action_term: Any
    enabled: bool = False
    force_tolerance_n: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.force_tolerance_n)) or self.force_tolerance_n < 0.0:
            raise ValueError("force_tolerance_n must be finite and non-negative")
        if self.enabled:
            for name in ("set_target_squeeze_force_n", "clear_target_squeeze_force_n", "get_force_status"):
                if not callable(getattr(self.action_term, name, None)):
                    raise TypeError(f"enabled ActiveForcing requires native ForcePositionAction.{name}()")
        self._decision: ActiveForcingDecision | None = None
        self._decision_count = 0

    @classmethod
    def from_env(cls, env: Any, *, enabled: bool, force_tolerance_n: float = 0.25) -> "ActiveForcingExecutor":
        term = env.action_manager.get_term("arm_action")
        return cls(term, enabled=enabled, force_tolerance_n=force_tolerance_n)

    @property
    def decision(self) -> ActiveForcingDecision | None:
        return self._decision

    def begin_episode(self) -> None:
        self._decision = None
        self._decision_count = 0
        if self.enabled:
            self.action_term.clear_target_squeeze_force_n()

    def consume_model_output(self, output: Any) -> ActiveForcingDecision | None:
        if not self.enabled:
            return None
        decision = ActiveForcingDecision.from_model_output(output)
        # set_target_squeeze_force_n resets native persistent correction state;
        # do not call it again when the model repeats the same scalar.
        if self._decision is None or decision.target_force_n != self._decision.target_force_n:
            self.action_term.set_target_squeeze_force_n(decision.target_force_n)
            self._decision_count += 1
            self._decision = decision
        return self._decision

    def validate_action(self, action: Any) -> None:
        if self.enabled:
            validate_native_13d_action(action)

    def prepare_action(self, action: Any) -> Any:
        """Validate and return the native action without changing any slice."""
        self.validate_action(action)
        return action

    def telemetry(self) -> dict[str, Any]:
        status = dict(self.action_term.get_force_status()) if callable(getattr(self.action_term, "get_force_status", None)) else {}
        active = bool(self.enabled and self._decision is not None and status.get("mode") == "newton")
        measured = status.get("measured_force_n")
        target = self._decision.target_force_n if self._decision is not None else None
        bilateral = status.get("bilateral_contact")
        error = None
        within_tolerance = None
        if target is not None and measured is not None:
            error = float(measured) - target
            within_tolerance = abs(error) <= float(self.force_tolerance_n)
        if bilateral is True:
            contact_status = "bilateral"
        elif bilateral is False:
            contact_status = "lost_or_insufficient"
        else:
            contact_status = "unknown"
        return {
            "activeforcing_protocol_version": PROTOCOL_VERSION,
            "activeforcing_telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "F_des_N": target,
            "F_meas_N": measured,
            "F_cmd": target,
            "force_error_N": error,
            "within_force_tolerance": within_tolerance,
            "execution_state": status.get("status", "WAITING_FOR_MODEL_DECISION" if self.enabled else "DISABLED"),
            "force_loop_active": active,
            "contact_status": contact_status,
            "contact_bilateral": bilateral,
            "native_force_status": status,
            "decision_count": self._decision_count,
        }

    def summary(self, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "model_decision": None if self._decision is None else self._decision.target_force_n,
            "decision_count": self._decision_count,
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "steps": len(rows),
            "force_loop_active_steps": sum(bool(row.get("force_loop_active")) for row in rows),
            "contact_lost_steps": sum(row.get("contact_status") == "lost_or_insufficient" for row in rows),
            "telemetry_rows": [dict(row) for row in rows],
        }


__all__ = [
    "ActiveForcingDecision",
    "ActiveForcingExecutor",
    "ActiveForcingModel",
    "JsonTargetForceModel",
    "PROTOCOL_VERSION",
    "ResponseTargetForceModel",
    "select_expected_utility_force",
    "validate_native_13d_action",
]
