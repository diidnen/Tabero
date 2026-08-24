# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parse optional per-task physics overrides for LIBERO assets.

This module deliberately has no IsaacLab imports so its configuration contract can
be validated in lightweight unit tests and tools.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal


@dataclass(frozen=True)
class FixedMassConfig:
    """Use one fixed rigid-body mass for every environment clone."""

    mass_kg: float


@dataclass(frozen=True)
class UniformMassConfig:
    """Sample an absolute rigid-body mass from a uniform interval."""

    minimum_kg: float
    maximum_kg: float
    apply_on: Literal["startup", "reset"]


ObjectMassConfig = FixedMassConfig | UniformMassConfig


@dataclass(frozen=True)
class FixedFrictionConfig:
    """Use one static/dynamic friction pair for the selected collision shapes."""

    static_friction: float
    dynamic_friction: float


@dataclass(frozen=True)
class UniformFrictionConfig:
    """Choose a pre-sampled uniform friction bucket at startup or reset."""

    minimum_static_friction: float
    maximum_static_friction: float
    minimum_dynamic_friction: float
    maximum_dynamic_friction: float
    apply_on: Literal["startup", "reset"]
    num_buckets: int = 64


FrictionConfig = FixedFrictionConfig | UniformFrictionConfig


@dataclass(frozen=True)
class FixedDamageThresholdConfig:
    """Use one manually configured squeeze-force damage threshold."""

    max_squeeze_force: float


@dataclass(frozen=True)
class MassFrictionDamageThresholdConfig:
    """Derive the squeeze-force limit from reset-time mass and static friction."""

    tolerance_factor: float = 1.1


DamageThresholdConfig = FixedDamageThresholdConfig | MassFrictionDamageThresholdConfig


@dataclass(frozen=True)
class ObjectDamageConfig:
    """Terminate when measured object-specific squeeze stays above a limit."""

    threshold: DamageThresholdConfig
    consecutive_frames: int = 4


def compute_mass_friction_damage_threshold(
    mass_kg: Any,
    gravity_m_s2: Any,
    gripper_static_friction: Any,
    object_static_friction: Any,
    tolerance_factor: Any = 1.1,
) -> Any:
    """Compute ``m*g/mean(mu_gripper, mu_object)*k`` for scalars or tensors."""

    effective_friction = 0.5 * (
        gripper_static_friction + object_static_friction
    )
    return mass_kg * gravity_m_s2 / effective_friction * tolerance_factor


def _positive_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{field_name} must be finite and greater than zero, got {value!r}."
        )
    return result


def _nonnegative_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(
            f"{field_name} must be finite and greater than or equal to zero, got {value!r}."
        )
    return result


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero, got {value!r}.")
    return value


def _parse_mass_config(value: Any, field_name: str) -> ObjectMassConfig:
    if isinstance(value, Real) and not isinstance(value, bool):
        return FixedMassConfig(mass_kg=_positive_finite_number(value, field_name))

    if not isinstance(value, dict):
        raise TypeError(
            f"{field_name} must be a positive number or an object, got {value!r}."
        )

    supported_keys = {"distribution", "range", "apply_on"}
    unknown_keys = sorted(set(value) - supported_keys)
    if unknown_keys:
        raise ValueError(f"{field_name} has unsupported keys: {unknown_keys}.")

    distribution = value.get("distribution", "uniform")
    if distribution != "uniform":
        raise ValueError(
            f"{field_name}.distribution must be 'uniform', got {distribution!r}."
        )

    mass_range = value.get("range")
    if not isinstance(mass_range, (list, tuple)) or len(mass_range) != 2:
        raise TypeError(f"{field_name}.range must contain exactly [min_kg, max_kg].")
    minimum_kg = _positive_finite_number(mass_range[0], f"{field_name}.range[0]")
    maximum_kg = _positive_finite_number(mass_range[1], f"{field_name}.range[1]")
    if maximum_kg < minimum_kg:
        raise ValueError(
            f"{field_name}.range maximum must be greater than or equal to its minimum."
        )

    apply_on = value.get("apply_on", "startup")
    if apply_on not in ("startup", "reset"):
        raise ValueError(
            f"{field_name}.apply_on must be 'startup' or 'reset', got {apply_on!r}."
        )

    return UniformMassConfig(
        minimum_kg=minimum_kg,
        maximum_kg=maximum_kg,
        apply_on=apply_on,
    )


def _friction_range(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{field_name} must contain exactly [minimum, maximum].")
    minimum = _nonnegative_finite_number(value[0], f"{field_name}[0]")
    maximum = _nonnegative_finite_number(value[1], f"{field_name}[1]")
    if maximum < minimum:
        raise ValueError(
            f"{field_name} maximum must be greater than or equal to its minimum."
        )
    return minimum, maximum


def _parse_friction_config(value: Any, field_name: str) -> FrictionConfig:
    if isinstance(value, Real) and not isinstance(value, bool):
        coefficient = _nonnegative_finite_number(value, field_name)
        return FixedFrictionConfig(
            static_friction=coefficient,
            dynamic_friction=coefficient,
        )

    if not isinstance(value, dict):
        raise TypeError(
            f"{field_name} must be a non-negative number or an object, got {value!r}."
        )

    is_uniform = "distribution" in value or any(
        key in value for key in ("static_range", "dynamic_range", "apply_on", "num_buckets")
    )
    if not is_uniform:
        supported_keys = {"static", "dynamic"}
        unknown_keys = sorted(set(value) - supported_keys)
        if unknown_keys:
            raise ValueError(f"{field_name} has unsupported keys: {unknown_keys}.")
        if "static" not in value:
            raise ValueError(f"{field_name}.static is required.")
        static_friction = _nonnegative_finite_number(
            value["static"], f"{field_name}.static"
        )
        dynamic_friction = _nonnegative_finite_number(
            value.get("dynamic", static_friction), f"{field_name}.dynamic"
        )
        if dynamic_friction > static_friction:
            raise ValueError(
                f"{field_name}.dynamic must be less than or equal to {field_name}.static."
            )
        return FixedFrictionConfig(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        )

    supported_keys = {
        "distribution",
        "static_range",
        "dynamic_range",
        "apply_on",
        "num_buckets",
    }
    unknown_keys = sorted(set(value) - supported_keys)
    if unknown_keys:
        raise ValueError(f"{field_name} has unsupported keys: {unknown_keys}.")
    distribution = value.get("distribution", "uniform")
    if distribution != "uniform":
        raise ValueError(
            f"{field_name}.distribution must be 'uniform', got {distribution!r}."
        )
    if "static_range" not in value:
        raise ValueError(f"{field_name}.static_range is required.")
    static_minimum, static_maximum = _friction_range(
        value["static_range"], f"{field_name}.static_range"
    )
    dynamic_minimum, dynamic_maximum = _friction_range(
        value.get("dynamic_range", value["static_range"]),
        f"{field_name}.dynamic_range",
    )
    if dynamic_minimum > static_minimum:
        raise ValueError(
            f"{field_name}.dynamic_range[0] must be less than or equal to "
            f"{field_name}.static_range[0] so every sampled bucket can satisfy dynamic <= static."
        )

    apply_on = value.get("apply_on", "reset")
    if apply_on not in ("startup", "reset"):
        raise ValueError(
            f"{field_name}.apply_on must be 'startup' or 'reset', got {apply_on!r}."
        )
    num_buckets = _positive_integer(value.get("num_buckets", 64), f"{field_name}.num_buckets")
    if num_buckets > 4096:
        raise ValueError(f"{field_name}.num_buckets must not exceed 4096.")

    return UniformFrictionConfig(
        minimum_static_friction=static_minimum,
        maximum_static_friction=static_maximum,
        minimum_dynamic_friction=dynamic_minimum,
        maximum_dynamic_friction=dynamic_maximum,
        apply_on=apply_on,
        num_buckets=num_buckets,
    )


def parse_object_mass_configs(task_info: dict[str, Any]) -> dict[str, ObjectMassConfig]:
    """Parse ``task.physics.objects.<name>.mass_kg`` entries.

    Missing ``physics`` configuration is a strict no-op, preserving the mass
    authored in the source USD (or the simulator-derived mass when none is
    authored).
    """

    physics = task_info.get("physics")
    if physics is None:
        return {}
    if not isinstance(physics, dict):
        raise TypeError("task.physics must be an object.")

    supported_physics_keys = {"objects", "gripper"}
    unknown_physics_keys = sorted(set(physics) - supported_physics_keys)
    if unknown_physics_keys:
        raise ValueError(f"task.physics has unsupported keys: {unknown_physics_keys}.")

    physics_objects = physics.get("objects", {})
    if not isinstance(physics_objects, dict):
        raise TypeError("task.physics.objects must be an object.")

    task_objects = task_info.get("objects", {})
    if not isinstance(task_objects, dict):
        raise TypeError("task.objects must be an object.")

    configs: dict[str, ObjectMassConfig] = {}
    for object_name, object_physics in physics_objects.items():
        if object_name not in task_objects:
            raise ValueError(
                f"task.physics.objects references unknown task object {object_name!r}."
            )
        if not isinstance(object_physics, dict):
            raise TypeError(f"task.physics.objects.{object_name} must be an object.")

        mass_value = object_physics.get("mass_kg")
        if mass_value is None:
            continue
        configs[object_name] = _parse_mass_config(
            mass_value,
            f"task.physics.objects.{object_name}.mass_kg",
        )

    return configs


def parse_gripper_friction_config(task_info: dict[str, Any]) -> FrictionConfig | None:
    """Parse ``task.physics.gripper.friction`` for the two Franka finger bodies."""

    physics = task_info.get("physics")
    if physics is None:
        return None
    if not isinstance(physics, dict):
        raise TypeError("task.physics must be an object.")
    supported_physics_keys = {"objects", "gripper"}
    unknown_physics_keys = sorted(set(physics) - supported_physics_keys)
    if unknown_physics_keys:
        raise ValueError(f"task.physics has unsupported keys: {unknown_physics_keys}.")

    gripper = physics.get("gripper")
    if gripper is None:
        return None
    if not isinstance(gripper, dict):
        raise TypeError("task.physics.gripper must be an object.")
    unknown_gripper_keys = sorted(set(gripper) - {"friction"})
    if unknown_gripper_keys:
        raise ValueError(
            f"task.physics.gripper has unsupported keys: {unknown_gripper_keys}."
        )
    friction = gripper.get("friction")
    if friction is None:
        return None
    return _parse_friction_config(friction, "task.physics.gripper.friction")


def parse_object_friction_configs(task_info: dict[str, Any]) -> dict[str, FrictionConfig]:
    """Parse optional friction settings for named rigid objects in one task."""

    physics = task_info.get("physics")
    if physics is None:
        return {}
    if not isinstance(physics, dict):
        raise TypeError("task.physics must be an object.")
    supported_physics_keys = {"objects", "gripper"}
    unknown_physics_keys = sorted(set(physics) - supported_physics_keys)
    if unknown_physics_keys:
        raise ValueError(f"task.physics has unsupported keys: {unknown_physics_keys}.")

    physics_objects = physics.get("objects", {})
    if not isinstance(physics_objects, dict):
        raise TypeError("task.physics.objects must be an object.")
    task_objects = task_info.get("objects", {})
    if not isinstance(task_objects, dict):
        raise TypeError("task.objects must be an object.")

    configs: dict[str, FrictionConfig] = {}
    for object_name, object_physics in physics_objects.items():
        if object_name not in task_objects:
            raise ValueError(
                f"task.physics.objects references unknown task object {object_name!r}."
            )
        if not isinstance(object_physics, dict):
            raise TypeError(f"task.physics.objects.{object_name} must be an object.")
        friction = object_physics.get("friction")
        if friction is None:
            continue
        configs[object_name] = _parse_friction_config(
            friction, f"task.physics.objects.{object_name}.friction"
        )
    return configs


def parse_object_damage_configs(task_info: dict[str, Any]) -> dict[str, ObjectDamageConfig]:
    """Parse per-object damage terminals from a LIBERO task entry.

    The supported contract is::

        physics.objects.<object_name>.damage = {
            "max_squeeze_force": <positive finite number>,
            "consecutive_frames": <positive integer, default 4>,
        }

    or::

        physics.objects.<object_name>.damage = {
            "threshold": {
                "mode": "mass_friction",
                "tolerance_factor": <positive finite number, default 1.1>,
            },
            "consecutive_frames": <positive integer, default 4>,
        }

    Damage may only be configured for an object listed in ``obj_of_interest``.
    Missing damage configuration is a strict no-op.
    """

    physics = task_info.get("physics")
    if physics is None:
        return {}
    if not isinstance(physics, dict):
        raise TypeError("task.physics must be an object.")

    physics_objects = physics.get("objects", {})
    if not isinstance(physics_objects, dict):
        raise TypeError("task.physics.objects must be an object.")

    task_objects = task_info.get("objects", {})
    if not isinstance(task_objects, dict):
        raise TypeError("task.objects must be an object.")
    objects_of_interest = task_info.get("obj_of_interest", [])
    if not isinstance(objects_of_interest, list):
        raise TypeError("task.obj_of_interest must be a list.")

    configs: dict[str, ObjectDamageConfig] = {}
    for object_name, object_physics in physics_objects.items():
        if object_name not in task_objects:
            raise ValueError(
                f"task.physics.objects references unknown task object {object_name!r}."
            )
        if not isinstance(object_physics, dict):
            raise TypeError(f"task.physics.objects.{object_name} must be an object.")

        damage_value = object_physics.get("damage")
        if damage_value is None:
            continue
        field_name = f"task.physics.objects.{object_name}.damage"
        if object_name not in objects_of_interest:
            raise ValueError(
                f"{field_name} may only be configured for an object in task.obj_of_interest."
            )
        if not isinstance(damage_value, dict):
            raise TypeError(f"{field_name} must be an object.")

        supported_keys = {"max_squeeze_force", "threshold", "consecutive_frames"}
        unknown_keys = sorted(set(damage_value) - supported_keys)
        if unknown_keys:
            raise ValueError(f"{field_name} has unsupported keys: {unknown_keys}.")

        has_fixed_threshold = "max_squeeze_force" in damage_value
        has_derived_threshold = "threshold" in damage_value
        if has_fixed_threshold == has_derived_threshold:
            raise ValueError(
                f"{field_name} must define exactly one of max_squeeze_force or threshold."
            )

        if has_fixed_threshold:
            threshold: DamageThresholdConfig = FixedDamageThresholdConfig(
                max_squeeze_force=_positive_finite_number(
                    damage_value["max_squeeze_force"],
                    f"{field_name}.max_squeeze_force",
                )
            )
        else:
            threshold_value = damage_value["threshold"]
            threshold_field = f"{field_name}.threshold"
            if not isinstance(threshold_value, dict):
                raise TypeError(f"{threshold_field} must be an object.")
            unknown_threshold_keys = sorted(
                set(threshold_value) - {"mode", "tolerance_factor"}
            )
            if unknown_threshold_keys:
                raise ValueError(
                    f"{threshold_field} has unsupported keys: {unknown_threshold_keys}."
                )
            mode = threshold_value.get("mode")
            if mode != "mass_friction":
                raise ValueError(
                    f"{threshold_field}.mode must be 'mass_friction', got {mode!r}."
                )
            threshold = MassFrictionDamageThresholdConfig(
                tolerance_factor=_positive_finite_number(
                    threshold_value.get("tolerance_factor", 1.1),
                    f"{threshold_field}.tolerance_factor",
                )
            )

            gripper_friction = parse_gripper_friction_config(task_info)
            object_friction = parse_object_friction_configs(task_info).get(object_name)
            if gripper_friction is None:
                raise ValueError(
                    f"{field_name} mass_friction mode requires task.physics.gripper.friction."
                )
            if object_friction is None:
                raise ValueError(
                    f"{field_name} mass_friction mode requires "
                    f"task.physics.objects.{object_name}.friction."
                )

            gripper_minimum = (
                gripper_friction.static_friction
                if isinstance(gripper_friction, FixedFrictionConfig)
                else gripper_friction.minimum_static_friction
            )
            object_minimum = (
                object_friction.static_friction
                if isinstance(object_friction, FixedFrictionConfig)
                else object_friction.minimum_static_friction
            )
            if gripper_minimum + object_minimum <= 0.0:
                raise ValueError(
                    f"{field_name} mass_friction mode requires the minimum gripper and "
                    "object static-friction coefficients to have a positive sum."
                )

        configs[object_name] = ObjectDamageConfig(
            threshold=threshold,
            consecutive_frames=_positive_integer(
                damage_value.get("consecutive_frames", 4),
                f"{field_name}.consecutive_frames",
            ),
        )

    return configs
