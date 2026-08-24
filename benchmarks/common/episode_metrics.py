from __future__ import annotations

"""Pure helpers for per-episode force and step evaluation metrics."""

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from benchmarks.common.metrics import (
    compute_contact_force_metrics_from_13d,
    compute_contact_force_metrics_from_lr_forces,
    compute_topk_mean,
)


FORCE_METRIC_KEYS = (
    "squeeze_avg_pred",
    "squeeze_avg_meas",
    "squeeze_max_pred",
    "squeeze_max_meas",
    "ap_avg_pred",
    "ap_avg_meas",
    "ap_max_pred",
    "ap_max_meas",
)

TRAJECTORY_FORCE_STATUSES = frozenset(
    {
        "complete",
        "insufficient_samples",
        "no_valid_samples",
        "grasp_not_started",
        "unavailable",
        "error",
    }
)


def _scalar_bool(value: Any, *, field_name: str) -> bool:
    """Convert one tensor/array/scalar grasp observation to bool."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"{field_name} must contain exactly one environment value, got shape "
            f"{array.shape}"
        )
    return bool(array.reshape(-1)[0])


class TrajectoryForceTracker:
    """Track the measured squeeze samples used by RLinf's success force bonus.

    The grasp observation is latched per source. A sample is valid only after a
    grasp has started, both fingers exceed ``contact_epsilon_n`` in force norm,
    and the raw two-finger squeeze exceeds the same threshold. The force input
    is the current, unfiltered gripper-local ``(left, right) x (x, y, z)`` force.
    """

    def __init__(
        self,
        grasp_term_names: Sequence[str],
        *,
        contact_epsilon_n: float,
        min_valid_samples: int,
    ) -> None:
        names = tuple(str(name) for name in grasp_term_names)
        if not names or any(not name.startswith("grasp_") for name in names):
            raise ValueError("grasp_term_names must contain at least one grasp_<n> term")
        if len(set(names)) != len(names):
            raise ValueError("grasp_term_names must not contain duplicates")
        if not math.isfinite(contact_epsilon_n) or contact_epsilon_n < 0.0:
            raise ValueError("contact_epsilon_n must be finite and non-negative")
        if type(min_valid_samples) is not int or min_valid_samples < 1:
            raise ValueError("min_valid_samples must be a positive integer")

        self.grasp_term_names = names
        self.contact_epsilon_n = float(contact_epsilon_n)
        self.min_valid_samples = int(min_valid_samples)
        self.reset()

    def reset(self) -> None:
        self._grasp_started_by_source = {
            name: False for name in self.grasp_term_names
        }
        self._grasp_start_step: int | None = None
        self._force_sum = 0.0
        self._valid_samples = 0

    def update(
        self,
        *,
        step_index: int,
        grasp_observations: dict[str, Any],
        force_lr_raw: Any,
    ) -> dict[str, Any]:
        """Consume one post-``env.step`` state and return trace-ready fields."""

        if type(step_index) is not int or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if not isinstance(grasp_observations, dict):
            raise ValueError("subtask_terms observations must be a dictionary")

        grasp_observed = False
        active_terms: list[str] = []
        for name in self.grasp_term_names:
            if name not in grasp_observations:
                raise ValueError(f"missing grasp observation {name!r}")
            grasped = _scalar_bool(grasp_observations[name], field_name=name)
            grasp_observed |= grasped
            if grasped:
                active_terms.append(name)
                self._grasp_started_by_source[name] = True
        if grasp_observed and self._grasp_start_step is None:
            self._grasp_start_step = int(step_index)

        force_lr = np.asarray(force_lr_raw, dtype=np.float64)
        if force_lr.shape != (2, 3):
            raise ValueError(
                "raw gripper force must have shape (2, 3), got "
                f"{force_lr.shape}"
            )
        if not np.all(np.isfinite(force_lr)):
            raise ValueError("raw gripper force contains non-finite values")

        finger_norms = np.linalg.norm(force_lr, axis=-1)
        raw_squeeze = 2.0 * min(abs(force_lr[0, 2]), abs(force_lr[1, 2]))
        grasp_started = any(self._grasp_started_by_source.values())
        both_fingers_contact = bool(
            np.all(finger_norms > self.contact_epsilon_n)
        )
        valid_step = bool(
            grasp_started
            and both_fingers_contact
            and raw_squeeze > self.contact_epsilon_n
        )
        if valid_step:
            self._force_sum += float(raw_squeeze)
            self._valid_samples += 1

        return {
            "reward_grasp_observed": bool(grasp_observed),
            "reward_grasp_terms_active": active_terms,
            "reward_grasp_started": bool(grasp_started),
            "reward_force_valid_step": bool(valid_step),
            "reward_squeeze_meas_raw": float(raw_squeeze),
        }

    def summary(self) -> dict[str, Any]:
        grasp_started = any(self._grasp_started_by_source.values())
        if not grasp_started:
            status = "grasp_not_started"
        elif self._valid_samples == 0:
            status = "no_valid_samples"
        elif self._valid_samples < self.min_valid_samples:
            status = "insufficient_samples"
        else:
            status = "complete"
        mean_force = (
            self._force_sum / float(self._valid_samples)
            if self._valid_samples > 0
            else None
        )
        return {
            "trajectory_mean_measured_squeeze": mean_force,
            "trajectory_force_valid_samples": int(self._valid_samples),
            "trajectory_force_status": status,
            "trajectory_force_error": None,
            "grasp_started": bool(grasp_started),
            "grasp_start_step": self._grasp_start_step,
        }


def summarize_force_tracking_metrics(
    *,
    expected_steps: int,
    reward_valid_steps: Sequence[Any],
    predicted_squeeze_values: Sequence[Any],
    effective_squeeze_target_values: Sequence[Any],
    measured_squeeze_raw_values: Sequence[Any],
    gripper_pred_values: Sequence[Any],
    gripper_cmd_values: Sequence[Any],
    gripper_closed_limit_m: float,
    gripper_open_limit_m: float,
    saturation_epsilon_m: float = 1.0e-8,
) -> dict[str, Any]:
    """Summarize force-target tracking on a strictly aligned step population.

    The legacy ``squeeze_avg_pred`` and ``squeeze_avg_meas`` fields intentionally
    retain their historical whole-trajectory semantics.  This additive summary
    instead compares raw model target, controller-effective target, and raw
    measured squeeze on the exact reward-valid steps.  It also reports when the
    final gripper position command is clamped at either position limit.
    """

    if type(expected_steps) is not int or expected_steps < 0:
        raise ValueError("expected_steps must be a non-negative integer")
    sequences = {
        "reward_valid_steps": reward_valid_steps,
        "predicted_squeeze_values": predicted_squeeze_values,
        "effective_squeeze_target_values": effective_squeeze_target_values,
        "measured_squeeze_raw_values": measured_squeeze_raw_values,
        "gripper_pred_values": gripper_pred_values,
        "gripper_cmd_values": gripper_cmd_values,
    }
    mismatched = {
        name: len(values)
        for name, values in sequences.items()
        if len(values) != expected_steps
    }
    if mismatched:
        raise ValueError(
            "force-tracking sequences must match expected_steps; got "
            f"expected_steps={expected_steps}, lengths={mismatched}"
        )
    for name, value in (
        ("gripper_closed_limit_m", gripper_closed_limit_m),
        ("gripper_open_limit_m", gripper_open_limit_m),
        ("saturation_epsilon_m", saturation_epsilon_m),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    closed_limit = float(gripper_closed_limit_m)
    open_limit = float(gripper_open_limit_m)
    saturation_epsilon = float(saturation_epsilon_m)
    if open_limit <= closed_limit:
        raise ValueError("gripper_open_limit_m must exceed gripper_closed_limit_m")
    if saturation_epsilon < 0.0:
        raise ValueError("saturation_epsilon_m must be non-negative")

    available_rows: list[tuple[bool, float, float, float, float, float]] = []
    unavailable_steps = 0
    reward_valid_total = 0
    for index in range(expected_steps):
        valid_step = bool(reward_valid_steps[index])
        reward_valid_total += int(valid_step)
        numeric_values: list[float] = []
        for values in (
            predicted_squeeze_values,
            effective_squeeze_target_values,
            measured_squeeze_raw_values,
            gripper_pred_values,
            gripper_cmd_values,
        ):
            value = values[index]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                break
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                break
            numeric_values.append(numeric_value)
        if len(numeric_values) != 5:
            unavailable_steps += 1
            continue
        available_rows.append((valid_step, *numeric_values))

    valid_rows = [row for row in available_rows if row[0]]
    lower_saturated = [
        row for row in available_rows if row[5] <= closed_limit + saturation_epsilon
    ]
    upper_saturated = [
        row for row in available_rows if row[5] >= open_limit - saturation_epsilon
    ]
    valid_lower_saturated = [
        row for row in valid_rows if row[5] <= closed_limit + saturation_epsilon
    ]
    valid_upper_saturated = [
        row for row in valid_rows if row[5] >= open_limit - saturation_epsilon
    ]

    def _mean(values: Sequence[float]) -> float | None:
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None

    def _rmse(values: Sequence[float]) -> float | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return float(np.sqrt(np.mean(np.square(array))))

    predicted_valid = [row[1] for row in valid_rows]
    effective_valid = [row[2] for row in valid_rows]
    measured_valid = [row[3] for row in valid_rows]
    model_errors = [
        predicted - measured
        for predicted, measured in zip(predicted_valid, measured_valid)
    ]
    effective_errors = [
        effective - measured
        for effective, measured in zip(effective_valid, measured_valid)
    ]
    available_steps = len(available_rows)
    valid_available_steps = len(valid_rows)
    if expected_steps == 0 or available_steps == 0:
        status = "unavailable"
    elif available_steps == expected_steps and valid_available_steps == reward_valid_total:
        status = "complete"
    else:
        status = "partial"

    return {
        "status": status,
        "semantics": {
            "population": "reward_force_valid_step",
            "predicted": "raw_model_squeeze_target",
            "effective_target": "controller_target_after_feed_forward_or_override",
            "measured": "raw_unfiltered_gripper_local_squeeze",
        },
        "sample_counts": {
            "env_steps": int(expected_steps),
            "available_steps": int(available_steps),
            "unavailable_steps": int(unavailable_steps),
            "reward_valid_steps": int(reward_valid_total),
            "reward_valid_available_steps": int(valid_available_steps),
        },
        "reward_valid": {
            "mean_predicted_squeeze_n": _mean(predicted_valid),
            "mean_effective_squeeze_target_n": _mean(effective_valid),
            "mean_measured_squeeze_raw_n": _mean(measured_valid),
            "mean_model_target_error_n": _mean(model_errors),
            "mean_effective_target_error_n": _mean(effective_errors),
            "effective_target_mae_n": _mean([abs(value) for value in effective_errors]),
            "effective_target_rmse_n": _rmse(effective_errors),
        },
        "gripper_position_command": {
            "closed_limit_m": closed_limit,
            "open_limit_m": open_limit,
            "saturation_epsilon_m": saturation_epsilon,
            "lower_saturation_steps": int(len(lower_saturated)),
            "upper_saturation_steps": int(len(upper_saturated)),
            "lower_saturation_ratio": (
                float(len(lower_saturated)) / float(available_steps)
                if available_steps
                else None
            ),
            "upper_saturation_ratio": (
                float(len(upper_saturated)) / float(available_steps)
                if available_steps
                else None
            ),
            "reward_valid_lower_saturation_steps": int(
                len(valid_lower_saturated)
            ),
            "reward_valid_upper_saturation_steps": int(
                len(valid_upper_saturated)
            ),
            "reward_valid_lower_saturation_ratio": (
                float(len(valid_lower_saturated)) / float(valid_available_steps)
                if valid_available_steps
                else None
            ),
            "reward_valid_upper_saturation_ratio": (
                float(len(valid_upper_saturated)) / float(valid_available_steps)
                if valid_available_steps
                else None
            ),
        },
    }


def resolve_episode_termination(
    *,
    info: Any,
    terminated: bool,
    truncated: bool,
) -> tuple[str, str | None]:
    """Resolve a stable evaluator end reason from IsaacLab termination logs.

    IsaacLab auto-resets a terminated environment before ``env.step`` returns,
    but it preserves the triggering manager term under
    ``info['log']['Episode_Termination/<term>']``.  Damage terms take precedence
    over generic termination labels so a broken object cannot be reported as an
    unexplained failure.
    """

    if truncated:
        return "truncated", None
    if not terminated:
        return "running", None

    log = info.get("log", {}) if isinstance(info, dict) else {}
    if not isinstance(log, dict):
        return "terminated", None

    active_terms: list[str] = []
    prefix = "Episode_Termination/"
    for key, value in log.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        try:
            if hasattr(value, "item"):
                value = value.item()
            active = float(value) > 0.0
        except (TypeError, ValueError):
            active = bool(value)
        if active:
            active_terms.append(key[len(prefix) :])

    damage_terms = sorted(term for term in active_terms if term.startswith("object_damage_"))
    if damage_terms:
        return "object_damage", damage_terms[0]
    if active_terms:
        return "terminated", sorted(active_terms)[0]
    return "terminated", None


def extract_object_damage_details(
    *, info: Any, terminal_term: str | None, env_index: int = 0
) -> dict[str, Any] | None:
    """Extract a JSON-safe damage trigger snapshot from IsaacLab extras."""

    prefix = "object_damage_"
    if not terminal_term or not terminal_term.startswith(prefix):
        return None
    object_name = terminal_term[len(prefix) :]
    damage_root = info.get("object_damage", {}) if isinstance(info, dict) else {}
    if not isinstance(damage_root, dict):
        return None
    raw = damage_root.get(object_name)
    if not isinstance(raw, dict):
        return None

    def scalar(value: Any) -> Any:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[env_index]
            if hasattr(value, "item"):
                value = value.item()
        except (IndexError, TypeError, ValueError):
            return None
        return value

    def optional_float(value: Any) -> float | None:
        value = scalar(value)
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    max_squeeze_force = optional_float(raw.get("max_squeeze_force"))
    consecutive_count = scalar(raw.get("consecutive_count"))
    measured_squeeze_force = optional_float(raw.get("measured_squeeze_force"))
    if max_squeeze_force is None or consecutive_count is None or measured_squeeze_force is None:
        return None

    return {
        "object_name": object_name,
        "mode": str(raw.get("mode", "fixed")),
        "mass_kg": optional_float(raw.get("mass_kg")),
        "gravity_m_s2": optional_float(raw.get("gravity_m_s2")),
        "gripper_static_friction": optional_float(raw.get("gripper_static_friction")),
        "object_static_friction": optional_float(raw.get("object_static_friction")),
        "effective_static_friction": optional_float(raw.get("effective_static_friction")),
        "tolerance_factor": optional_float(raw.get("tolerance_factor")),
        "max_squeeze_force": max_squeeze_force,
        "consecutive_frames": int(raw["consecutive_frames"]),
        "consecutive_count": int(consecutive_count),
        "measured_squeeze_force": measured_squeeze_force,
    }


def extract_damage_threshold_snapshot(
    *, info: Any, env_index: int = 0
) -> dict[str, Any] | None:
    """Extract the reset-time damage-threshold calculation for one LIBERO object."""

    root = info.get("object_damage_threshold", {}) if isinstance(info, dict) else {}
    if not isinstance(root, dict) or not root:
        return None

    object_name = sorted(root)[0]
    raw = root.get(object_name)
    if not isinstance(raw, dict):
        return None

    def scalar(value: Any) -> Any:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[env_index]
            if hasattr(value, "item"):
                value = value.item()
        except (IndexError, TypeError, ValueError):
            return None
        return value

    def optional_float(value: Any) -> float | None:
        value = scalar(value)
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    max_squeeze_force = optional_float(raw.get("max_squeeze_force"))
    if max_squeeze_force is None:
        return None
    return {
        "object_name": object_name,
        "mode": str(raw.get("mode", "fixed")),
        "mass_kg": optional_float(raw.get("mass_kg")),
        "gravity_m_s2": optional_float(raw.get("gravity_m_s2")),
        "gripper_static_friction": optional_float(raw.get("gripper_static_friction")),
        "object_static_friction": optional_float(raw.get("object_static_friction")),
        "effective_static_friction": optional_float(raw.get("effective_static_friction")),
        "tolerance_factor": optional_float(raw.get("tolerance_factor")),
        "max_squeeze_force": max_squeeze_force,
        "consecutive_frames": int(raw.get("consecutive_frames", 4)),
    }


def extract_friction_snapshot(*, info: Any, env_index: int = 0) -> dict[str, Any] | None:
    """Extract the currently active gripper/object friction pair from reset extras."""

    root = info.get("physics_friction", {}) if isinstance(info, dict) else {}
    if not isinstance(root, dict) or not root:
        return None

    def scalar(value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[env_index]
            if hasattr(value, "item"):
                value = value.item()
            result = float(value)
        except (IndexError, TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def pair(value: Any) -> dict[str, float] | None:
        if not isinstance(value, dict):
            return None
        static_friction = scalar(value.get("static_friction"))
        dynamic_friction = scalar(value.get("dynamic_friction"))
        if static_friction is None or dynamic_friction is None:
            return None
        return {
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
        }

    gripper = pair(root.get("gripper"))
    objects = {
        name: parsed
        for name, raw in sorted(root.items())
        if name != "gripper" and (parsed := pair(raw)) is not None
    }
    if gripper is None and not objects:
        return None
    return {"gripper": gripper, "objects": objects}


def summarize_friction_statistics(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize configured friction samples over all recorded episodes."""

    scopes: dict[str, dict[str, list[float]]] = {}
    for episode in episodes:
        friction = episode.get("friction")
        if not isinstance(friction, dict):
            continue
        gripper = friction.get("gripper")
        if isinstance(gripper, dict):
            scopes.setdefault("gripper", {"static_friction": [], "dynamic_friction": []})
            for key in ("static_friction", "dynamic_friction"):
                value = gripper.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    scopes["gripper"][key].append(float(value))
        objects = friction.get("objects", {})
        if not isinstance(objects, dict):
            continue
        for object_name, pair in objects.items():
            if not isinstance(object_name, str) or not isinstance(pair, dict):
                continue
            scope_name = f"object:{object_name}"
            scopes.setdefault(scope_name, {"static_friction": [], "dynamic_friction": []})
            for key in ("static_friction", "dynamic_friction"):
                value = pair.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    scopes[scope_name][key].append(float(value))

    result: dict[str, Any] = {}
    for scope_name, fields in sorted(scopes.items()):
        scope_summary: dict[str, Any] = {}
        for field_name, values in fields.items():
            scope_summary[field_name] = {
                "count": len(values),
                "mean": _mean_or_none(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        result[scope_name] = scope_summary
    return result


def summarize_damage_threshold_statistics(
    episodes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize reset-time mass, effective friction, and damage threshold by object."""

    fields = ("mass_kg", "effective_static_friction", "max_squeeze_force")
    objects: dict[str, dict[str, list[float]]] = {}
    for episode in episodes:
        snapshot = episode.get("damage_threshold")
        if not isinstance(snapshot, dict):
            continue
        object_name = snapshot.get("object_name")
        if not isinstance(object_name, str):
            continue
        values = objects.setdefault(object_name, {field: [] for field in fields})
        for field in fields:
            value = snapshot.get(field)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                values[field].append(float(value))

    result: dict[str, Any] = {}
    for object_name, values_by_field in sorted(objects.items()):
        result[object_name] = {
            field: {
                "count": len(values),
                "mean": _mean_or_none(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for field, values in values_by_field.items()
        }
    return result


def _float_values(values: Iterable[float]) -> list[float]:
    finite_values: list[float] = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite_values.append(value)
    return finite_values


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _topk_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(compute_topk_mean(values, frac=0.05))


def _stack_lr(values: Sequence[Any]) -> np.ndarray | None:
    if len(values) == 0:
        return None
    vectors: list[np.ndarray] = []
    for value in values:
        try:
            vector = np.asarray(value, dtype=np.float32).reshape(3)
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(vector)):
            vectors.append(vector)
    if not vectors:
        return None
    return np.stack(vectors, axis=0)


def summarize_episode_force_metrics(
    *,
    force_mode: str,
    env_steps: int,
    actions_13d: Sequence[Any] = (),
    squeeze_pred_values: Iterable[float] = (),
    squeeze_meas_values: Iterable[float] = (),
    ap_pred_values: Iterable[float] = (),
    ap_meas_values: Iterable[float] = (),
    fL_meas_values: Sequence[Any] = (),
    fR_meas_values: Sequence[Any] = (),
) -> dict[str, Any]:
    """Summarize one episode while preserving the legacy eight force metrics.

    ``force_mode`` is one of ``full`` (hybrid/tactile), ``measured_only``
    (binary), or ``not_applicable`` (diffik/osc).
    """

    if force_mode not in {"full", "measured_only", "not_applicable"}:
        raise ValueError(f"unsupported force_mode: {force_mode}")
    if env_steps < 0:
        raise ValueError(f"env_steps must be non-negative, got {env_steps}")

    squeeze_pred = _float_values(squeeze_pred_values)
    squeeze_meas = _float_values(squeeze_meas_values)
    ap_pred = _float_values(ap_pred_values)
    ap_meas = _float_values(ap_meas_values)

    action_array: np.ndarray | None = None
    if len(actions_13d) > 0:
        valid_actions: list[np.ndarray] = []
        for action in actions_13d:
            try:
                action_array_item = np.asarray(action, dtype=np.float32).reshape(13)
            except (TypeError, ValueError):
                continue
            if np.all(np.isfinite(action_array_item)):
                valid_actions.append(action_array_item)
        if valid_actions:
            action_array = np.stack(valid_actions)

    pred_metrics = None
    if action_array is not None:
        pred_metrics = compute_contact_force_metrics_from_13d(action_array)

    fL_meas = _stack_lr(fL_meas_values)
    fR_meas = _stack_lr(fR_meas_values)
    meas_metrics = None
    if fL_meas is not None and fR_meas is not None and fL_meas.shape == fR_meas.shape:
        meas_metrics = compute_contact_force_metrics_from_lr_forces(fL_meas, fR_meas)

    force_metrics: dict[str, float | None] = {key: None for key in FORCE_METRIC_KEYS}
    if force_mode == "full":
        force_metrics.update(
            {
                "squeeze_avg_pred": _mean_or_none(squeeze_pred),
                "squeeze_avg_meas": _mean_or_none(squeeze_meas),
                "squeeze_max_pred": None if pred_metrics is None else float(pred_metrics.squeeze_max),
                "squeeze_max_meas": _topk_or_none(squeeze_meas),
                "ap_avg_pred": None if pred_metrics is None else float(pred_metrics.external_norm_mean),
                "ap_avg_meas": _mean_or_none(ap_meas),
                "ap_max_pred": None if pred_metrics is None else float(pred_metrics.external_norm_max),
                "ap_max_meas": _topk_or_none(ap_meas),
            }
        )
    elif force_mode == "measured_only":
        # Preserve the legacy binary-mode convention: prediction-style aggregate
        # slots mirror the measured contact metrics because no force is predicted.
        force_metrics.update(
            {
                "squeeze_avg_pred": _mean_or_none(squeeze_pred),
                "squeeze_avg_meas": _mean_or_none(squeeze_meas),
                "squeeze_max_pred": None if meas_metrics is None else float(meas_metrics.squeeze_max),
                "squeeze_max_meas": None if meas_metrics is None else float(meas_metrics.squeeze_max),
                "ap_avg_pred": None if meas_metrics is None else float(meas_metrics.external_norm_mean),
                "ap_avg_meas": None if meas_metrics is None else float(meas_metrics.external_norm_mean),
                "ap_max_pred": None if meas_metrics is None else float(meas_metrics.external_norm_max),
                "ap_max_meas": None if meas_metrics is None else float(meas_metrics.external_norm_max),
            }
        )

    predicted_action_steps = 0 if pred_metrics is None else int(pred_metrics.num_steps)
    measured_force_steps = 0 if meas_metrics is None else int(meas_metrics.num_steps)
    force_samples = {
        "predicted_action_steps": predicted_action_steps,
        "measured_force_steps": measured_force_steps,
        "squeeze_pred_steps": len(squeeze_pred),
        "squeeze_meas_steps": len(squeeze_meas),
        "ap_pred_steps": len(ap_pred),
        "ap_meas_steps": len(ap_meas),
        "predicted_contact_steps": None if pred_metrics is None else int(pred_metrics.num_contact_steps),
        "measured_contact_steps": None if meas_metrics is None else int(meas_metrics.num_contact_steps),
        "predicted_contact_ratio": None if pred_metrics is None else float(pred_metrics.contact_ratio),
        "measured_contact_ratio": None if meas_metrics is None else float(meas_metrics.contact_ratio),
    }

    if force_mode == "not_applicable":
        force_status = "not_applicable"
        force_samples["coverage_ratio"] = None
    else:
        if force_mode == "full":
            required_counts = (
                predicted_action_steps,
                measured_force_steps,
                len(squeeze_pred),
                len(squeeze_meas),
                len(ap_pred),
                len(ap_meas),
            )
        else:
            required_counts = (measured_force_steps, len(squeeze_meas), len(ap_meas))
        minimum_count = min(required_counts, default=0)
        coverage_ratio = 1.0 if env_steps == 0 else min(1.0, float(minimum_count) / float(env_steps))
        force_samples["coverage_ratio"] = coverage_ratio
        force_status = "complete" if all(count == env_steps for count in required_counts) else "partial"

    return {
        "force_status": force_status,
        "force_samples": force_samples,
        **force_metrics,
    }


def aggregate_success_force_metrics(
    episodes: Sequence[dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Average each legacy metric independently over successful episodes."""

    aggregates: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for key in FORCE_METRIC_KEYS:
        values: list[float] = []
        for episode in episodes:
            if not episode.get("success"):
                continue
            value = episode.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            value = float(value)
            if math.isfinite(value):
                values.append(value)
        aggregates[key] = _mean_or_none(values)
        counts[key] = len(values)
    return aggregates, counts


def _numeric_summary(values: Sequence[float]) -> dict[str, int | float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "mean": _mean_or_none(finite),
        "median": float(np.median(finite)) if finite else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def summarize_trajectory_force_metrics(
    episodes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate only reward-eligible trajectory means while retaining coverage."""

    eligible_all: list[float] = []
    eligible_success: list[float] = []
    valid_samples_all: list[float] = []
    valid_samples_success: list[float] = []
    successful_total = 0
    successful_ineligible = 0

    for episode in episodes:
        success = episode.get("success") is True
        successful_total += int(success)
        sample_count = episode.get("trajectory_force_valid_samples")
        if type(sample_count) is int and sample_count >= 0:
            valid_samples_all.append(float(sample_count))
            if success:
                valid_samples_success.append(float(sample_count))

        value = episode.get("trajectory_mean_measured_squeeze")
        eligible = (
            episode.get("trajectory_force_status") == "complete"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
        if eligible:
            eligible_all.append(float(value))
            if success:
                eligible_success.append(float(value))
        elif success:
            successful_ineligible += 1

    return {
        "success_trajectory_mean_measured_squeeze": _mean_or_none(
            eligible_success
        ),
        "success_trajectory_force_episode_count": len(eligible_success),
        "all_eligible_trajectory_mean_measured_squeeze": _mean_or_none(
            eligible_all
        ),
        "all_eligible_trajectory_force_episode_count": len(eligible_all),
        "successful_trajectory_force_total_episodes": successful_total,
        "successful_trajectory_force_ineligible_episodes": successful_ineligible,
        "trajectory_force_valid_sample_statistics": {
            "all": _numeric_summary(valid_samples_all),
            "successful": _numeric_summary(valid_samples_success),
        },
    }


def _summarize_step_group(episodes: Sequence[dict[str, Any]]) -> dict[str, int | float | None]:
    summary: dict[str, int | float | None] = {"episodes": len(episodes)}
    for field in ("env_steps", "inference_chunks"):
        values = [episode.get(field) for episode in episodes]
        valid = [int(value) for value in values if type(value) is int and value >= 0]
        summary[f"{field}_total"] = sum(valid)
        summary[f"{field}_mean"] = float(np.mean(valid)) if valid else None
        summary[f"{field}_min"] = min(valid) if valid else None
        summary[f"{field}_max"] = max(valid) if valid else None
    return summary


def summarize_step_statistics(episodes: Sequence[dict[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    """Return all/successful/failed step summaries."""

    return {
        "all": _summarize_step_group(episodes),
        "successful": _summarize_step_group([episode for episode in episodes if episode.get("success") is True]),
        "failed": _summarize_step_group([episode for episode in episodes if episode.get("success") is False]),
    }


def validate_episode_records(
    episodes: Sequence[dict[str, Any]],
    *,
    expected_count: int,
    replan_steps: int,
    max_inference_chunks: int | None = None,
) -> list[str]:
    """Return completeness warnings without changing evaluation success status."""

    warnings: list[str] = []
    indices = [episode.get("experiment_index") for episode in episodes]
    valid_indices = [index for index in indices if type(index) is int]
    if len(episodes) != expected_count:
        warnings.append(f"expected {expected_count} episode records, found {len(episodes)}")
    if len(valid_indices) != len(indices):
        warnings.append("one or more episode records have a non-integer experiment_index")
    if len(set(valid_indices)) != len(valid_indices):
        warnings.append("duplicate experiment_index values in episode records")
    if set(valid_indices) != set(range(expected_count)):
        warnings.append("episode records do not cover experiment_index 0..N-1 exactly")

    for episode in episodes:
        index = episode.get("experiment_index")
        env_steps = episode.get("env_steps")
        inference_chunks = episode.get("inference_chunks")
        label = f"episode {index}"
        if type(episode.get("success")) is not bool:
            warnings.append(f"{label} has invalid success={episode.get('success')!r}")
        if not isinstance(episode.get("end_reason"), str) or not episode.get("end_reason"):
            warnings.append(f"{label} has invalid end_reason={episode.get('end_reason')!r}")
        hdf5_episode_index = episode.get("hdf5_episode_index")
        if hdf5_episode_index is not None and type(hdf5_episode_index) is not int:
            warnings.append(
                f"{label} has invalid hdf5_episode_index={hdf5_episode_index!r}"
            )
        if type(env_steps) is not int or env_steps <= 0:
            warnings.append(f"{label} has invalid env_steps={env_steps!r}")
            continue
        if type(inference_chunks) is not int or inference_chunks <= 0:
            warnings.append(f"{label} has invalid inference_chunks={inference_chunks!r}")
            continue
        if max_inference_chunks is not None and inference_chunks > max_inference_chunks:
            warnings.append(
                f"{label} inference_chunks={inference_chunks} exceeds maximum {max_inference_chunks}"
            )
        minimum_steps = (inference_chunks - 1) * replan_steps + 1
        maximum_steps = inference_chunks * replan_steps
        if not minimum_steps <= env_steps <= maximum_steps:
            warnings.append(
                f"{label} env_steps={env_steps} is outside [{minimum_steps}, {maximum_steps}] "
                f"for inference_chunks={inference_chunks}"
            )
        force_status = episode.get("force_status")
        if force_status not in {"complete", "partial", "not_applicable"}:
            warnings.append(f"{label} has invalid force_status={force_status!r}")
        elif force_status == "partial":
            warnings.append(f"{label} has partial force coverage")
        force_samples = episode.get("force_samples")
        if not isinstance(force_samples, dict):
            warnings.append(f"{label} has invalid force_samples")
        for key in FORCE_METRIC_KEYS:
            value = episode.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                warnings.append(f"{label} has invalid {key}={value!r}")
        trajectory_status = episode.get("trajectory_force_status")
        if trajectory_status not in TRAJECTORY_FORCE_STATUSES:
            warnings.append(
                f"{label} has invalid trajectory_force_status={trajectory_status!r}"
            )
        trajectory_samples = episode.get("trajectory_force_valid_samples")
        if type(trajectory_samples) is not int or trajectory_samples < 0:
            warnings.append(
                f"{label} has invalid trajectory_force_valid_samples="
                f"{trajectory_samples!r}"
            )
        trajectory_mean = episode.get("trajectory_mean_measured_squeeze")
        if trajectory_mean is not None and (
            isinstance(trajectory_mean, bool)
            or not isinstance(trajectory_mean, (int, float))
            or not math.isfinite(float(trajectory_mean))
        ):
            warnings.append(
                f"{label} has invalid trajectory_mean_measured_squeeze="
                f"{trajectory_mean!r}"
            )
        if trajectory_status == "complete" and trajectory_mean is None:
            warnings.append(f"{label} complete trajectory force metric has no mean")
        if type(episode.get("grasp_started")) is not bool:
            warnings.append(
                f"{label} has invalid grasp_started={episode.get('grasp_started')!r}"
            )
        grasp_start_step = episode.get("grasp_start_step")
        if grasp_start_step is not None and (
            type(grasp_start_step) is not int or grasp_start_step < 0
        ):
            warnings.append(
                f"{label} has invalid grasp_start_step={grasp_start_step!r}"
            )
    return warnings


__all__ = [
    "FORCE_METRIC_KEYS",
    "TRAJECTORY_FORCE_STATUSES",
    "TrajectoryForceTracker",
    "aggregate_success_force_metrics",
    "summarize_trajectory_force_metrics",
    "summarize_episode_force_metrics",
    "summarize_step_statistics",
    "validate_episode_records",
]
