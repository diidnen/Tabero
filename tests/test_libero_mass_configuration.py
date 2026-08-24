import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "source/tac_manip/tac_manip/tasks/manipulation/libero/physics_config.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "tabero_libero_physics_config", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


physics_config = _load_module()


def _task_info(mass_kg=None):
    task = {
        "obj_of_interest": ["tomato_sauce_1"],
        "objects": {
            "tomato_sauce_1": {
                "type": "tomato_sauce",
                "scale": [0.01, 0.01, 0.01],
            }
        }
    }
    if mass_kg is not None:
        task["physics"] = {
            "objects": {
                "tomato_sauce_1": {
                    "mass_kg": mass_kg,
                }
            }
        }
    return task


def test_missing_physics_configuration_preserves_usd_mass():
    assert physics_config.parse_object_mass_configs(_task_info()) == {}


def test_current_libero_object_suite_remains_a_no_op():
    suite_path = ROOT / "benchmarks/datasets/libero/config/libero_object.json"
    suite = json.loads(suite_path.read_text())

    assert len(suite["tasks"]) == 10
    assert all(
        physics_config.parse_object_mass_configs(task) == {} for task in suite["tasks"]
    )
    assert all(
        physics_config.parse_object_damage_configs(task) == {} for task in suite["tasks"]
    )
    assert all(
        physics_config.parse_gripper_friction_config(task) is None
        for task in suite["tasks"]
    )
    assert all(
        physics_config.parse_object_friction_configs(task) == {}
        for task in suite["tasks"]
    )


def test_fixed_mass_configuration():
    configs = physics_config.parse_object_mass_configs(_task_info(0.25))

    assert configs == {"tomato_sauce_1": physics_config.FixedMassConfig(mass_kg=0.25)}


@pytest.mark.parametrize("apply_on", ["startup", "reset"])
def test_uniform_mass_configuration(apply_on):
    configs = physics_config.parse_object_mass_configs(
        _task_info(
            {
                "distribution": "uniform",
                "range": [0.08, 0.12],
                "apply_on": apply_on,
            }
        )
    )

    assert configs == {
        "tomato_sauce_1": physics_config.UniformMassConfig(
            minimum_kg=0.08,
            maximum_kg=0.12,
            apply_on=apply_on,
        )
    }


@pytest.mark.parametrize("mass_kg", [True, 0.0, -0.1, float("inf")])
def test_invalid_fixed_mass_is_rejected(mass_kg):
    with pytest.raises((TypeError, ValueError)):
        physics_config.parse_object_mass_configs(_task_info(mass_kg))


def test_invalid_uniform_mass_range_is_rejected():
    with pytest.raises(ValueError, match="maximum"):
        physics_config.parse_object_mass_configs(
            _task_info(
                {
                    "distribution": "uniform",
                    "range": [0.2, 0.1],
                    "apply_on": "startup",
                }
            )
        )


def test_unknown_object_is_rejected():
    task = _task_info()
    task["physics"] = {
        "objects": {
            "not_in_this_task": {
                "mass_kg": 0.1,
            }
        }
    }

    with pytest.raises(ValueError, match="unknown task object"):
        physics_config.parse_object_mass_configs(task)


def test_env_cfg_wires_fixed_and_randomized_mass_paths():
    source = (
        ROOT
        / "source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py"
    ).read_text()

    assert "mass_props=fixed_mass_props" in source
    assert "randomize_rigid_body_mass" in source
    assert '"operation": "abs"' in source
    assert '"recompute_inertia": True' in source


def test_fixed_scalar_friction_configuration():
    task = _task_info()
    task["physics"] = {
        "gripper": {"friction": 0.8},
        "objects": {"tomato_sauce_1": {"friction": 0.6}},
    }

    assert physics_config.parse_gripper_friction_config(task) == (
        physics_config.FixedFrictionConfig(
            static_friction=0.8,
            dynamic_friction=0.8,
        )
    )
    assert physics_config.parse_object_friction_configs(task) == {
        "tomato_sauce_1": physics_config.FixedFrictionConfig(
            static_friction=0.6,
            dynamic_friction=0.6,
        )
    }


def test_fixed_static_dynamic_friction_configuration():
    task = _task_info()
    task["physics"] = {
        "objects": {
            "tomato_sauce_1": {
                "friction": {"static": 0.9, "dynamic": 0.7},
            }
        }
    }

    assert physics_config.parse_object_friction_configs(task) == {
        "tomato_sauce_1": physics_config.FixedFrictionConfig(
            static_friction=0.9,
            dynamic_friction=0.7,
        )
    }


def test_uniform_friction_configuration_defaults():
    task = _task_info()
    task["physics"] = {
        "gripper": {
            "friction": {
                "distribution": "uniform",
                "static_range": [0.8, 1.2],
                "dynamic_range": [0.5, 0.9],
            }
        }
    }

    assert physics_config.parse_gripper_friction_config(task) == (
        physics_config.UniformFrictionConfig(
            minimum_static_friction=0.8,
            maximum_static_friction=1.2,
            minimum_dynamic_friction=0.5,
            maximum_dynamic_friction=0.9,
            apply_on="reset",
            num_buckets=64,
        )
    )


def test_uniform_friction_configuration_explicit_controls():
    task = _task_info()
    task["physics"] = {
        "objects": {
            "tomato_sauce_1": {
                "friction": {
                    "distribution": "uniform",
                    "static_range": [0.4, 0.8],
                    "dynamic_range": [0.3, 0.6],
                    "apply_on": "startup",
                    "num_buckets": 16,
                }
            }
        }
    }

    assert physics_config.parse_object_friction_configs(task) == {
        "tomato_sauce_1": physics_config.UniformFrictionConfig(
            minimum_static_friction=0.4,
            maximum_static_friction=0.8,
            minimum_dynamic_friction=0.3,
            maximum_dynamic_friction=0.6,
            apply_on="startup",
            num_buckets=16,
        )
    }


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan"), True])
def test_invalid_scalar_friction_is_rejected(value):
    task = _task_info()
    task["physics"] = {"gripper": {"friction": value}}
    with pytest.raises((TypeError, ValueError)):
        physics_config.parse_gripper_friction_config(task)


def test_zero_friction_is_valid():
    task = _task_info()
    task["physics"] = {"gripper": {"friction": 0.0}}
    assert physics_config.parse_gripper_friction_config(task) == (
        physics_config.FixedFrictionConfig(0.0, 0.0)
    )


def test_dynamic_friction_above_static_is_rejected():
    task = _task_info()
    task["physics"] = {
        "gripper": {"friction": {"static": 0.5, "dynamic": 0.6}}
    }
    with pytest.raises(ValueError, match="dynamic"):
        physics_config.parse_gripper_friction_config(task)


def test_uniform_friction_requires_feasible_dynamic_range():
    task = _task_info()
    task["physics"] = {
        "gripper": {
            "friction": {
                "distribution": "uniform",
                "static_range": [0.4, 0.8],
                "dynamic_range": [0.5, 0.6],
            }
        }
    }
    with pytest.raises(ValueError, match="dynamic_range"):
        physics_config.parse_gripper_friction_config(task)


def test_unknown_object_friction_is_rejected():
    task = _task_info()
    task["physics"] = {
        "objects": {"missing_1": {"friction": 0.5}},
    }
    with pytest.raises(ValueError, match="unknown task object"):
        physics_config.parse_object_friction_configs(task)


def test_mass_damage_and_friction_can_coexist():
    task = _task_info()
    task["physics"] = {
        "gripper": {"friction": 0.8},
        "objects": {
            "tomato_sauce_1": {
                "mass_kg": 0.1,
                "friction": {"static": 0.7, "dynamic": 0.5},
                "damage": {"max_squeeze_force": 25.0},
            }
        },
    }

    assert physics_config.parse_object_mass_configs(task)
    assert physics_config.parse_object_damage_configs(task)
    assert physics_config.parse_gripper_friction_config(task)
    assert physics_config.parse_object_friction_configs(task)


def test_env_cfg_wires_object_and_gripper_friction_paths():
    source = (
        ROOT
        / "source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py"
    ).read_text()
    event_source = (
        ROOT
        / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/events.py"
    ).read_text()

    assert "parse_gripper_friction_config" in source
    assert "parse_object_friction_configs" in source
    assert 'body_names=["panda_leftfinger", "panda_rightfinger"]' in source
    assert "set_or_randomize_rigid_body_friction" in source
    assert "selected_shape_ids" in event_source
    assert "materials[env_ids_cpu[:, None], shape_ids[None, :], 0]" in event_source
    assert "materials[env_ids_cpu[:, None], shape_ids[None, :], 1]" in event_source


def test_damage_configuration_defaults_to_four_consecutive_frames():
    task = _task_info()
    task["physics"] = {
        "objects": {
            "tomato_sauce_1": {
                "damage": {"max_squeeze_force": 25.0},
            }
        }
    }

    assert physics_config.parse_object_damage_configs(task) == {
        "tomato_sauce_1": physics_config.ObjectDamageConfig(
            threshold=physics_config.FixedDamageThresholdConfig(
                max_squeeze_force=25.0
            ),
            consecutive_frames=4,
        )
    }


def test_damage_configuration_accepts_per_object_frame_override():
    task = _task_info()
    task["physics"] = {
        "objects": {
            "tomato_sauce_1": {
                "damage": {
                    "max_squeeze_force": 40.0,
                    "consecutive_frames": 7,
                },
            }
        }
    }

    assert physics_config.parse_object_damage_configs(task) == {
        "tomato_sauce_1": physics_config.ObjectDamageConfig(
            threshold=physics_config.FixedDamageThresholdConfig(
                max_squeeze_force=40.0
            ),
            consecutive_frames=7,
        )
    }


def test_mass_friction_damage_configuration_defaults_tolerance_factor():
    task = _task_info()
    task["physics"] = {
        "gripper": {"friction": {"static": 0.5, "dynamic": 0.4}},
        "objects": {
            "tomato_sauce_1": {
                "friction": {"static": 0.7, "dynamic": 0.5},
                "damage": {"threshold": {"mode": "mass_friction"}},
            }
        },
    }

    assert physics_config.parse_object_damage_configs(task) == {
        "tomato_sauce_1": physics_config.ObjectDamageConfig(
            threshold=physics_config.MassFrictionDamageThresholdConfig(
                tolerance_factor=1.1
            ),
            consecutive_frames=4,
        )
    }


def test_mass_friction_damage_configuration_accepts_tolerance_override():
    task = _task_info()
    task["physics"] = {
        "gripper": {"friction": 0.5},
        "objects": {
            "tomato_sauce_1": {
                "friction": 0.7,
                "damage": {
                    "threshold": {
                        "mode": "mass_friction",
                        "tolerance_factor": 1.25,
                    },
                    "consecutive_frames": 6,
                },
            }
        },
    }

    config = physics_config.parse_object_damage_configs(task)["tomato_sauce_1"]
    assert config == physics_config.ObjectDamageConfig(
        threshold=physics_config.MassFrictionDamageThresholdConfig(
            tolerance_factor=1.25
        ),
        consecutive_frames=6,
    )


def test_mass_friction_damage_formula():
    threshold = physics_config.compute_mass_friction_damage_threshold(
        mass_kg=1.0,
        gravity_m_s2=9.81,
        gripper_static_friction=0.5,
        object_static_friction=0.7,
        tolerance_factor=1.1,
    )

    assert threshold == pytest.approx(17.985)


@pytest.mark.parametrize(
    ("damage", "error_type"),
    [
        ({}, ValueError),
        ({"max_squeeze_force": 0.0}, ValueError),
        ({"max_squeeze_force": float("inf")}, ValueError),
        ({"max_squeeze_force": 10.0, "consecutive_frames": 0}, ValueError),
        ({"max_squeeze_force": 10.0, "consecutive_frames": True}, TypeError),
        ({"max_squeeze_force": 10.0, "unknown": 1}, ValueError),
        (
            {
                "max_squeeze_force": 10.0,
                "threshold": {"mode": "mass_friction"},
            },
            ValueError,
        ),
        ({"threshold": {"mode": "unknown"}}, ValueError),
        (
            {
                "threshold": {
                    "mode": "mass_friction",
                    "tolerance_factor": 0.0,
                }
            },
            ValueError,
        ),
        ({"threshold": {"mode": "mass_friction", "unknown": 1}}, ValueError),
    ],
)
def test_invalid_damage_configuration_is_rejected(damage, error_type):
    task = _task_info()
    task["physics"] = {
        "objects": {"tomato_sauce_1": {"damage": damage}}
    }
    with pytest.raises(error_type):
        physics_config.parse_object_damage_configs(task)


def test_damage_configuration_rejects_non_target_object():
    task = _task_info()
    task["objects"]["basket_1"] = {"type": "basket", "scale": [1, 1, 1]}
    task["physics"] = {
        "objects": {
            "basket_1": {"damage": {"max_squeeze_force": 25.0}},
        }
    }

    with pytest.raises(ValueError, match="obj_of_interest"):
        physics_config.parse_object_damage_configs(task)


def test_mass_friction_damage_requires_explicit_gripper_friction():
    task = _task_info()
    task["physics"] = {
        "objects": {
            "tomato_sauce_1": {
                "friction": 0.5,
                "damage": {"threshold": {"mode": "mass_friction"}},
            }
        }
    }

    with pytest.raises(ValueError, match="gripper.friction"):
        physics_config.parse_object_damage_configs(task)


def test_mass_friction_damage_requires_explicit_object_friction():
    task = _task_info()
    task["physics"] = {
        "gripper": {"friction": 0.5},
        "objects": {
            "tomato_sauce_1": {
                "damage": {"threshold": {"mode": "mass_friction"}},
            }
        },
    }

    with pytest.raises(ValueError, match="tomato_sauce_1.friction"):
        physics_config.parse_object_damage_configs(task)


def test_mass_friction_damage_rejects_zero_minimum_effective_friction():
    task = _task_info()
    task["physics"] = {
        "gripper": {"friction": 0.0},
        "objects": {
            "tomato_sauce_1": {
                "friction": 0.0,
                "damage": {"threshold": {"mode": "mass_friction"}},
            }
        },
    }

    with pytest.raises(ValueError, match="positive sum"):
        physics_config.parse_object_damage_configs(task)


def test_env_cfg_wires_object_damage_terminal():
    source = (
        ROOT
        / "source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_libero_env_cfg.py"
    ).read_text()
    termination_source = (
        ROOT
        / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/terminations.py"
    ).read_text()

    assert "parse_object_damage_configs" in source
    assert 'f"object_damage_{object_name}"' in source
    assert "contact_grasp_{object_name}" in source
    assert "force_matrix_w" in termination_source
    assert "self._consecutive_counts >= int(consecutive_frames)" in termination_source


def test_firm_damage_profile_matches_20k_report_values():
    profile_path = (
        ROOT
        / "benchmarks/datasets/libero/config_profiles/firm_damage_from_rlinf_sft_20k/libero_object.json"
    )
    suite = json.loads(profile_path.read_text())
    expected = {
        0: ("alphabet_soup_1", 73.7260),
        1: ("cream_cheese_1", 47.1633),
        2: ("salad_dressing_1", 16.4627),
        3: ("bbq_sauce_1", 13.8459),
        5: ("tomato_sauce_1", 54.0869),
        6: ("butter_1", 38.1775),
        7: ("milk_1", 48.9534),
        8: ("chocolate_pudding_1", 58.0237),
        9: ("orange_juice_1", 35.3483),
    }

    for task in suite["tasks"]:
        task_id = task["task_id"]
        damage_configs = physics_config.parse_object_damage_configs(task)
        if task_id not in expected:
            assert damage_configs == {}
            continue
        object_name, threshold = expected[task_id]
        assert damage_configs == {
            object_name: physics_config.ObjectDamageConfig(
                threshold=physics_config.FixedDamageThresholdConfig(
                    max_squeeze_force=threshold
                ),
                consecutive_frames=4,
            )
        }

    task5 = next(task for task in suite["tasks"] if task["task_id"] == 5)
    assert physics_config.parse_object_mass_configs(task5) == {
        "tomato_sauce_1": physics_config.UniformMassConfig(
            minimum_kg=0.08,
            maximum_kg=0.12,
            apply_on="reset",
        )
    }


def test_firm_mass_friction_profile_covers_all_firm_targets():
    profile_path = (
        ROOT
        / "benchmarks/datasets/libero/config_profiles/firm_damage_fixed_friction_05_from_rlinf_sft_20k/libero_object.json"
    )
    suite = json.loads(profile_path.read_text())
    firm_task_ids = {0, 1, 2, 3, 5, 6, 7, 8, 9}

    for task in suite["tasks"]:
        task_id = task["task_id"]
        if task_id not in firm_task_ids:
            assert physics_config.parse_gripper_friction_config(task) is None
            assert physics_config.parse_object_friction_configs(task) == {}
            continue
        target = task["obj_of_interest"][0]
        expected = physics_config.FixedFrictionConfig(
            static_friction=0.5,
            dynamic_friction=0.5,
        )
        assert physics_config.parse_gripper_friction_config(task) == expected
        object_friction = physics_config.parse_object_friction_configs(task)
        if task_id == 0:
            assert object_friction == {
                target: physics_config.UniformFrictionConfig(
                    minimum_static_friction=0.4,
                    maximum_static_friction=0.8,
                    minimum_dynamic_friction=0.3,
                    maximum_dynamic_friction=0.6,
                    apply_on="reset",
                    num_buckets=64,
                )
            }
        else:
            assert object_friction == {target: expected}
        assert physics_config.parse_object_damage_configs(task) == {
            target: physics_config.ObjectDamageConfig(
                threshold=physics_config.MassFrictionDamageThresholdConfig(
                    tolerance_factor=1.1
                ),
                consecutive_frames=4,
            )
        }

    task0 = next(task for task in suite["tasks"] if task["task_id"] == 0)
    assert physics_config.parse_object_mass_configs(task0) == {
        "alphabet_soup_1": physics_config.UniformMassConfig(
            minimum_kg=0.8,
            maximum_kg=1.2,
            apply_on="reset",
        )
    }
    task5 = next(task for task in suite["tasks"] if task["task_id"] == 5)
    assert physics_config.parse_object_mass_configs(task5) == {
        "tomato_sauce_1": physics_config.UniformMassConfig(
            minimum_kg=0.08,
            maximum_kg=0.12,
            apply_on="reset",
        )
    }

    checksum_path = profile_path.with_name("libero_object.json.sha256")
    expected_checksum = checksum_path.read_text().split()[0]
    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == expected_checksum

    threshold_provenance = json.loads(
        profile_path.with_name("threshold_provenance.json").read_text()
    )
    assert threshold_provenance["schema_version"] == 2
    assert threshold_provenance["active_derivation"]["mode"] == "mass_friction"
    assert threshold_provenance["superseded_manual_thresholds"]["values"]["0"][
        "max_squeeze_force"
    ] == 73.726


def test_alltask_uniform_mass_friction_damage_profile_covers_every_target():
    profile_path = (
        ROOT
        / "benchmarks/datasets/libero/config_profiles/"
        "alltask_damage_uniform_mass_05_16_friction_04_08_from_rlinf_sft_20k/"
        "libero_object.json"
    )
    suite = json.loads(profile_path.read_text())
    expected_targets = {
        0: "alphabet_soup_1",
        1: "cream_cheese_1",
        2: "salad_dressing_1",
        3: "bbq_sauce_1",
        4: "ketchup_1",
        5: "tomato_sauce_1",
        6: "butter_1",
        7: "milk_1",
        8: "chocolate_pudding_1",
        9: "orange_juice_1",
    }
    expected_mass = physics_config.UniformMassConfig(
        minimum_kg=0.5,
        maximum_kg=1.6,
        apply_on="reset",
    )
    expected_object_friction = physics_config.UniformFrictionConfig(
        minimum_static_friction=0.4,
        maximum_static_friction=0.8,
        minimum_dynamic_friction=0.3,
        maximum_dynamic_friction=0.6,
        apply_on="reset",
        num_buckets=64,
    )
    expected_gripper_friction = physics_config.FixedFrictionConfig(
        static_friction=0.5,
        dynamic_friction=0.5,
    )

    assert {task["task_id"] for task in suite["tasks"]} == set(expected_targets)
    for task in suite["tasks"]:
        task_id = task["task_id"]
        target = expected_targets[task_id]
        assert task["obj_of_interest"] == [target]
        assert physics_config.parse_object_mass_configs(task) == {
            target: expected_mass
        }
        assert physics_config.parse_object_friction_configs(task) == {
            target: expected_object_friction
        }
        assert (
            physics_config.parse_gripper_friction_config(task)
            == expected_gripper_friction
        )
        assert physics_config.parse_object_damage_configs(task) == {
            target: physics_config.ObjectDamageConfig(
                threshold=physics_config.MassFrictionDamageThresholdConfig(
                    tolerance_factor=1.1
                ),
                consecutive_frames=4,
            )
        }

    checksum_path = profile_path.with_name("libero_object.json.sha256")
    expected_checksum = checksum_path.read_text().split()[0]
    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == expected_checksum

    friction_provenance = json.loads(
        profile_path.with_name("friction_provenance.json").read_text()
    )
    assert friction_provenance["task_ids"] == list(range(10))
    assert friction_provenance["objects"]["static_range"] == [0.4, 0.8]
    assert friction_provenance["objects"]["dynamic_range"] == [0.3, 0.6]

    threshold_provenance = json.loads(
        profile_path.with_name("threshold_provenance.json").read_text()
    )
    assert threshold_provenance["active_derivation"]["mode"] == "mass_friction"
    assert set(threshold_provenance["tasks"]) == {str(task_id) for task_id in range(10)}
    assert threshold_provenance["formal_physical_damage_calibration"] is False


def test_alltask_fixed_high_damage_profile_preserves_uniform_random_physics():
    profile_path = (
        ROOT
        / "benchmarks/datasets/libero/config_profiles/"
        "alltask_fixed_damage_1000000_uniform_mass_05_16_friction_04_08_"
        "from_rlinf_sft_20k/libero_object.json"
    )
    suite = json.loads(profile_path.read_text())
    expected_targets = {
        0: "alphabet_soup_1",
        1: "cream_cheese_1",
        2: "salad_dressing_1",
        3: "bbq_sauce_1",
        4: "ketchup_1",
        5: "tomato_sauce_1",
        6: "butter_1",
        7: "milk_1",
        8: "chocolate_pudding_1",
        9: "orange_juice_1",
    }
    expected_mass = physics_config.UniformMassConfig(
        minimum_kg=0.5,
        maximum_kg=1.6,
        apply_on="reset",
    )
    expected_object_friction = physics_config.UniformFrictionConfig(
        minimum_static_friction=0.4,
        maximum_static_friction=0.8,
        minimum_dynamic_friction=0.3,
        maximum_dynamic_friction=0.6,
        apply_on="reset",
        num_buckets=64,
    )
    expected_gripper_friction = physics_config.FixedFrictionConfig(
        static_friction=0.5,
        dynamic_friction=0.5,
    )
    expected_damage = physics_config.FixedDamageThresholdConfig(
        max_squeeze_force=1_000_000.0
    )

    assert {task["task_id"] for task in suite["tasks"]} == set(expected_targets)
    for task in suite["tasks"]:
        target = expected_targets[task["task_id"]]
        assert task["obj_of_interest"] == [target]
        assert physics_config.parse_object_mass_configs(task) == {
            target: expected_mass
        }
        assert physics_config.parse_object_friction_configs(task) == {
            target: expected_object_friction
        }
        assert (
            physics_config.parse_gripper_friction_config(task)
            == expected_gripper_friction
        )
        assert physics_config.parse_object_damage_configs(task) == {
            target: physics_config.ObjectDamageConfig(
                threshold=expected_damage,
                consecutive_frames=4,
            )
        }

    checksum_path = profile_path.with_name("libero_object.json.sha256")
    expected_checksum = checksum_path.read_text().split()[0]
    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == expected_checksum

    threshold_provenance = json.loads(
        profile_path.with_name("threshold_provenance.json").read_text()
    )
    assert threshold_provenance["active_derivation"] == {
        "mode": "fixed",
        "max_squeeze_force": 1_000_000.0,
        "consecutive_frames": 4,
        "purpose": (
            "Keep the damage terminal installed while making its limit inactive "
            "over the benchmark's observed force range"
        ),
        "validity_rule": (
            "Any object-damage trigger invalidates the no-damage-limit baseline "
            "interpretation"
        ),
    }
    assert set(threshold_provenance["tasks"]) == {
        str(task_id) for task_id in range(10)
    }
    assert threshold_provenance["formal_physical_damage_calibration"] is False
