import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/tac_manip/tac_manip/tasks/manipulation/libero/control_config.py"
)
SPEC = importlib.util.spec_from_file_location("control_config", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
control_config = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = control_config
SPEC.loader.exec_module(control_config)


def _task_info(override=None):
    task = {
        "objects": {"alphabet_soup_1": {}, "basket_1": {}},
        "obj_of_interest": ["alphabet_soup_1"],
    }
    if override is not None:
        task["control_overrides"] = {"target_contact_squeeze": override}
    return task


def test_missing_or_disabled_override_is_a_strict_noop():
    assert control_config.parse_target_contact_squeeze_override(_task_info()) is None
    assert (
        control_config.parse_target_contact_squeeze_override(
            _task_info({"enabled": False})
        )
        is None
    )


def test_enabled_override_parses_13n_as_26n_squeeze():
    parsed = control_config.parse_target_contact_squeeze_override(
        _task_info(
            {
                "enabled": True,
                "target_object": "alphabet_soup_1",
                "single_finger_normal_force_n": 13.0,
                "contact_threshold_n": 0.1,
                "activation": "first_contact_latched",
            }
        )
    )

    assert parsed == control_config.TargetContactSqueezeOverrideConfig(
        target_object="alphabet_soup_1",
        single_finger_normal_force_n=13.0,
        contact_threshold_n=0.1,
        activation="first_contact_latched",
    )
    assert parsed.squeeze_force_n == 26.0


@pytest.mark.parametrize(
    ("patch", "error_type"),
    [
        ({"enabled": 1}, TypeError),
        ({"enabled": True, "target_object": "basket_1", "single_finger_normal_force_n": 13.0}, ValueError),
        ({"enabled": True, "target_object": "missing", "single_finger_normal_force_n": 13.0}, ValueError),
        ({"enabled": True, "target_object": "alphabet_soup_1", "single_finger_normal_force_n": 0.0}, ValueError),
        ({"enabled": True, "target_object": "alphabet_soup_1", "single_finger_normal_force_n": math.nan}, ValueError),
        ({"enabled": True, "target_object": "alphabet_soup_1", "single_finger_normal_force_n": 13.0, "contact_threshold_n": -0.1}, ValueError),
        ({"enabled": True, "target_object": "alphabet_soup_1", "single_finger_normal_force_n": 13.0, "activation": "while_contact"}, ValueError),
        ({"enabled": True, "target_object": "alphabet_soup_1", "single_finger_normal_force_n": 13.0, "unknown": 1}, ValueError),
    ],
)
def test_invalid_enabled_overrides_are_rejected(patch, error_type):
    with pytest.raises(error_type):
        control_config.parse_target_contact_squeeze_override(_task_info(patch))


def test_unknown_control_override_family_is_rejected():
    task = _task_info()
    task["control_overrides"] = {"unknown": {}}

    with pytest.raises(ValueError, match="unsupported keys"):
        control_config.parse_target_contact_squeeze_override(task)


def test_task0_fixed_mean_13n_profile_is_complete_and_checksummed():
    profile_path = (
        ROOT
        / "benchmarks/datasets/libero/config_profiles/"
        "task0_fixed_mean_mass_friction_contact_force_13n_from_rlinf_sft_20k/"
        "libero_object.json"
    )
    suite = json.loads(profile_path.read_text())

    assert suite["total_tasks"] == 1
    assert len(suite["tasks"]) == 1
    task = suite["tasks"][0]
    assert task["task_id"] == 0
    assert task["physics"] == {
        "objects": {
            "alphabet_soup_1": {
                "damage": {
                    "max_squeeze_force": 1_000_000,
                    "consecutive_frames": 4,
                },
                "friction": {"static": 0.6, "dynamic": 0.45},
                "mass_kg": 1.05,
            }
        },
        "gripper": {"friction": {"static": 0.5, "dynamic": 0.5}},
    }
    override = control_config.parse_target_contact_squeeze_override(task)
    assert override is not None
    assert override.single_finger_normal_force_n == 13.0
    assert override.squeeze_force_n == 26.0

    expected_checksum = profile_path.with_name(
        "libero_object.json.sha256"
    ).read_text().split()[0]
    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == expected_checksum


def test_force_action_applies_fixed_target_after_feedforward_and_resets_latch():
    source = (
        ROOT
        / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/"
        "force_position_action.py"
    ).read_text()

    feedforward = source.index("ff = float(self.cfg.squeeze_ff_k_load_z)")
    fixed_override = source.index("override_target = torch.full_like")
    assert fixed_override > feedforward
    assert "self._target_contact_latched.logical_or_(detected)" in source
    assert "self._target_contact_latched[ids] = False" in source
    assert "self._target_contact_activation_step[ids] = -1" in source
    assert "self._target_contact_sensor.reset(ids)" in source
    assert "detected &= self._environment_step_count > 0" in source


def test_target_contact_sensor_is_target_centric_and_opt_in():
    source = (
        ROOT
        / "source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/"
        "franka_libero_env_cfg.py"
    ).read_text()

    assert "if squeeze_override is not None:" in source
    assert '"{ENV_REGEX_NS}/" + squeeze_override.target_object' in source
    assert '"{ENV_REGEX_NS}/Robot/gelsight_mini_case_left"' in source
    assert '"{ENV_REGEX_NS}/Robot/gelsight_mini_case_right"' in source
