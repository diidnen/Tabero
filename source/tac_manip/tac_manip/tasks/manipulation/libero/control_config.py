"""Parse optional per-task control overrides for LIBERO environments.

This module intentionally has no IsaacLab imports so the JSON contract can be
validated in lightweight unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal


@dataclass(frozen=True)
class TargetContactSqueezeOverrideConfig:
    """Latch a fixed squeeze target after touching one configured object."""

    target_object: str
    single_finger_normal_force_n: float
    contact_threshold_n: float
    activation: Literal["first_contact_latched"]

    @property
    def squeeze_force_n(self) -> float:
        """Return the two-finger squeeze target used by the controller."""

        return 2.0 * self.single_finger_normal_force_n


def _finite_number(
    value: Any,
    field_name: str,
    *,
    strictly_positive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value!r}.")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero, got {value!r}.")
    if not strictly_positive and result < 0.0:
        raise ValueError(
            f"{field_name} must be greater than or equal to zero, got {value!r}."
        )
    return result


def parse_target_contact_squeeze_override(
    task_info: dict[str, Any],
) -> TargetContactSqueezeOverrideConfig | None:
    """Parse ``task.control_overrides.target_contact_squeeze``.

    A missing block, or a block with ``enabled=false``, is a strict no-op.
    Enabled overrides are deliberately narrow: the target must be one of the
    task's objects of interest and the only supported activation mode latches
    after the first object-filtered finger contact.
    """

    root = task_info.get("control_overrides")
    if root is None:
        return None
    if not isinstance(root, dict):
        raise TypeError("task.control_overrides must be an object.")

    unknown_root_keys = sorted(set(root) - {"target_contact_squeeze"})
    if unknown_root_keys:
        raise ValueError(
            "task.control_overrides has unsupported keys: "
            f"{unknown_root_keys}."
        )

    value = root.get("target_contact_squeeze")
    if value is None:
        return None
    field_name = "task.control_overrides.target_contact_squeeze"
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object.")

    supported_keys = {
        "enabled",
        "target_object",
        "single_finger_normal_force_n",
        "contact_threshold_n",
        "activation",
    }
    unknown_keys = sorted(set(value) - supported_keys)
    if unknown_keys:
        raise ValueError(f"{field_name} has unsupported keys: {unknown_keys}.")

    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{field_name}.enabled must be a boolean, got {enabled!r}.")
    if not enabled:
        return None

    target_object = value.get("target_object")
    if not isinstance(target_object, str) or not target_object.strip():
        raise TypeError(f"{field_name}.target_object must be a non-empty string.")
    target_object = target_object.strip()

    task_objects = task_info.get("objects", {})
    if not isinstance(task_objects, dict):
        raise TypeError("task.objects must be an object.")
    if target_object not in task_objects:
        raise ValueError(
            f"{field_name}.target_object references unknown task object "
            f"{target_object!r}."
        )

    objects_of_interest = task_info.get("obj_of_interest", [])
    if not isinstance(objects_of_interest, list):
        raise TypeError("task.obj_of_interest must be a list.")
    if target_object not in objects_of_interest:
        raise ValueError(
            f"{field_name}.target_object must belong to task.obj_of_interest, "
            f"got {target_object!r}."
        )

    if "single_finger_normal_force_n" not in value:
        raise ValueError(f"{field_name}.single_finger_normal_force_n is required.")
    single_finger_force = _finite_number(
        value["single_finger_normal_force_n"],
        f"{field_name}.single_finger_normal_force_n",
        strictly_positive=True,
    )
    contact_threshold = _finite_number(
        value.get("contact_threshold_n", 0.1),
        f"{field_name}.contact_threshold_n",
        strictly_positive=False,
    )

    activation = value.get("activation", "first_contact_latched")
    if activation != "first_contact_latched":
        raise ValueError(
            f"{field_name}.activation must be 'first_contact_latched', "
            f"got {activation!r}."
        )

    return TargetContactSqueezeOverrideConfig(
        target_object=target_object,
        single_finger_normal_force_n=single_finger_force,
        contact_threshold_n=contact_threshold,
        activation=activation,
    )
