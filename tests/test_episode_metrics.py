import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.episode_metrics import (
    FORCE_METRIC_KEYS,
    TrajectoryForceTracker,
    aggregate_success_force_metrics,
    extract_damage_threshold_snapshot,
    extract_friction_snapshot,
    extract_object_damage_details,
    resolve_episode_termination,
    summarize_episode_force_metrics,
    summarize_force_tracking_metrics,
    summarize_damage_threshold_statistics,
    summarize_friction_statistics,
    summarize_step_statistics,
    summarize_trajectory_force_metrics,
    validate_episode_records,
)


def _raw_lr(squeeze: float, *, left_norm_scale: float = 1.0) -> np.ndarray:
    force = np.zeros((2, 3), dtype=np.float32)
    force[0, 2] = 0.5 * squeeze * left_norm_scale
    force[1, 2] = -0.5 * squeeze
    return force


def test_trajectory_force_tracker_matches_reward_grasp_and_contact_gates():
    tracker = TrajectoryForceTracker(
        ("grasp_1",), contact_epsilon_n=1.0, min_valid_samples=4
    )

    pre_grasp = tracker.update(
        step_index=0,
        grasp_observations={"grasp_1": False},
        force_lr_raw=_raw_lr(100.0),
    )
    assert pre_grasp["reward_force_valid_step"] is False

    expected_squeezes = [20.0, 40.0, 60.0, 80.0]
    for step, squeeze in enumerate(expected_squeezes, start=1):
        trace = tracker.update(
            step_index=step,
            grasp_observations={"grasp_1": step == 1},
            force_lr_raw=_raw_lr(squeeze),
        )
        assert trace["reward_grasp_started"] is True
        assert trace["reward_force_valid_step"] is True

    # The grasp observation is latched, but a released/zero-force step is ignored.
    released = tracker.update(
        step_index=5,
        grasp_observations={"grasp_1": False},
        force_lr_raw=np.zeros((2, 3), dtype=np.float32),
    )
    assert released["reward_grasp_started"] is True
    assert released["reward_force_valid_step"] is False
    assert tracker.summary() == {
        "trajectory_mean_measured_squeeze": pytest.approx(50.0),
        "trajectory_force_valid_samples": 4,
        "trajectory_force_status": "complete",
        "trajectory_force_error": None,
        "grasp_started": True,
        "grasp_start_step": 1,
    }


def test_trajectory_force_tracker_rejects_single_finger_and_uses_raw_not_ema():
    tracker = TrajectoryForceTracker(
        ("grasp_1",), contact_epsilon_n=1.0, min_valid_samples=1
    )
    single_finger = np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, -30.0]])
    trace = tracker.update(
        step_index=0,
        grasp_observations={"grasp_1": True},
        force_lr_raw=single_finger,
    )
    assert trace["reward_force_valid_step"] is False

    # A hypothetical EMA value is deliberately absent: only raw vectors enter.
    raw_trace = tracker.update(
        step_index=1,
        grasp_observations={"grasp_1": False},
        force_lr_raw=_raw_lr(30.0),
    )
    assert raw_trace["reward_squeeze_meas_raw"] == pytest.approx(30.0)
    assert tracker.summary()["trajectory_mean_measured_squeeze"] == pytest.approx(
        30.0
    )


def test_trajectory_force_tracker_status_multi_source_and_reset():
    tracker = TrajectoryForceTracker(
        ("grasp_1", "grasp_2"),
        contact_epsilon_n=1.0,
        min_valid_samples=4,
    )
    for step in range(3):
        trace = tracker.update(
            step_index=step,
            grasp_observations={
                "grasp_1": step == 0,
                "grasp_2": step == 0,
            },
            force_lr_raw=_raw_lr(10.0 + step),
        )
        if step == 0:
            assert trace["reward_grasp_terms_active"] == ["grasp_1", "grasp_2"]
    assert tracker.summary()["trajectory_force_status"] == "insufficient_samples"
    assert tracker.summary()["trajectory_force_valid_samples"] == 3

    tracker.reset()
    assert tracker.summary()["trajectory_force_status"] == "grasp_not_started"
    assert tracker.summary()["trajectory_mean_measured_squeeze"] is None

    tracker.update(
        step_index=0,
        grasp_observations={"grasp_1": True, "grasp_2": False},
        force_lr_raw=np.zeros((2, 3), dtype=np.float32),
    )
    assert tracker.summary()["trajectory_force_status"] == "no_valid_samples"
    assert tracker.summary()["trajectory_force_valid_samples"] == 0
    assert tracker.summary()["trajectory_mean_measured_squeeze"] is None


def test_trajectory_force_aggregation_is_success_and_eligibility_aware():
    episodes = [
        {
            "success": True,
            "trajectory_force_status": "complete",
            "trajectory_force_valid_samples": 4,
            "trajectory_mean_measured_squeeze": 20.0,
        },
        {
            "success": True,
            "trajectory_force_status": "insufficient_samples",
            "trajectory_force_valid_samples": 3,
            "trajectory_mean_measured_squeeze": 10.0,
        },
        {
            "success": False,
            "trajectory_force_status": "complete",
            "trajectory_force_valid_samples": 5,
            "trajectory_mean_measured_squeeze": 40.0,
        },
    ]

    summary = summarize_trajectory_force_metrics(episodes)
    assert summary["success_trajectory_mean_measured_squeeze"] == 20.0
    assert summary["success_trajectory_force_episode_count"] == 1
    assert summary["all_eligible_trajectory_mean_measured_squeeze"] == 30.0
    assert summary["all_eligible_trajectory_force_episode_count"] == 2
    assert summary["successful_trajectory_force_ineligible_episodes"] == 1
    assert summary["trajectory_force_valid_sample_statistics"]["successful"][
        "median"
    ] == pytest.approx(3.5)


def test_force_tracking_uses_reward_valid_population_and_counts_saturation():
    summary = summarize_force_tracking_metrics(
        expected_steps=4,
        reward_valid_steps=[False, True, True, True],
        predicted_squeeze_values=[1.0, 80.0, 90.0, 100.0],
        effective_squeeze_target_values=[1.0, 152.0, 171.0, 190.0],
        measured_squeeze_raw_values=[0.0, 30.0, 40.0, 50.0],
        gripper_pred_values=[0.04, 0.02, 0.02, 0.02],
        gripper_cmd_values=[0.04, 0.0, 0.0, 0.01],
        gripper_closed_limit_m=0.0,
        gripper_open_limit_m=0.04,
    )

    assert summary["status"] == "complete"
    assert summary["sample_counts"] == {
        "env_steps": 4,
        "available_steps": 4,
        "unavailable_steps": 0,
        "reward_valid_steps": 3,
        "reward_valid_available_steps": 3,
    }
    assert summary["reward_valid"]["mean_predicted_squeeze_n"] == pytest.approx(
        90.0
    )
    assert summary["reward_valid"][
        "mean_effective_squeeze_target_n"
    ] == pytest.approx(171.0)
    assert summary["reward_valid"][
        "mean_measured_squeeze_raw_n"
    ] == pytest.approx(40.0)
    assert summary["reward_valid"][
        "mean_effective_target_error_n"
    ] == pytest.approx(131.0)
    command = summary["gripper_position_command"]
    assert command["lower_saturation_steps"] == 2
    assert command["upper_saturation_steps"] == 1
    assert command["reward_valid_lower_saturation_steps"] == 2
    assert command["reward_valid_lower_saturation_ratio"] == pytest.approx(2 / 3)


def test_force_tracking_marks_missing_scalars_partial_and_rejects_misalignment():
    summary = summarize_force_tracking_metrics(
        expected_steps=2,
        reward_valid_steps=[False, True],
        predicted_squeeze_values=[1.0, 2.0],
        effective_squeeze_target_values=[1.0, None],
        measured_squeeze_raw_values=[0.0, 1.0],
        gripper_pred_values=[0.04, 0.02],
        gripper_cmd_values=[0.04, 0.01],
        gripper_closed_limit_m=0.0,
        gripper_open_limit_m=0.04,
    )
    assert summary["status"] == "partial"
    assert summary["sample_counts"]["available_steps"] == 1
    assert summary["sample_counts"]["reward_valid_available_steps"] == 0

    with pytest.raises(ValueError, match="must match expected_steps"):
        summarize_force_tracking_metrics(
            expected_steps=2,
            reward_valid_steps=[True],
            predicted_squeeze_values=[1.0, 2.0],
            effective_squeeze_target_values=[1.0, 2.0],
            measured_squeeze_raw_values=[1.0, 2.0],
            gripper_pred_values=[0.01, 0.01],
            gripper_cmd_values=[0.0, 0.0],
            gripper_closed_limit_m=0.0,
            gripper_open_limit_m=0.04,
        )


def test_extract_friction_snapshot_is_json_safe():
    snapshot = extract_friction_snapshot(
        info={
            "physics_friction": {
                "gripper": {
                    "static_friction": np.asarray([0.8]),
                    "dynamic_friction": np.asarray([0.6]),
                },
                "tomato_sauce_1": {
                    "static_friction": np.asarray([0.7]),
                    "dynamic_friction": np.asarray([0.5]),
                },
            }
        }
    )

    assert snapshot == {
        "gripper": {"static_friction": 0.8, "dynamic_friction": 0.6},
        "objects": {
            "tomato_sauce_1": {
                "static_friction": 0.7,
                "dynamic_friction": 0.5,
            }
        },
    }


def test_summarize_friction_statistics_keeps_scope_counts():
    episodes = [
        {
            "friction": {
                "gripper": {"static_friction": 0.8, "dynamic_friction": 0.6},
                "objects": {
                    "tomato_sauce_1": {
                        "static_friction": 0.5,
                        "dynamic_friction": 0.4,
                    }
                },
            }
        },
        {
            "friction": {
                "gripper": {"static_friction": 1.0, "dynamic_friction": 0.8},
                "objects": {
                    "tomato_sauce_1": {
                        "static_friction": 0.7,
                        "dynamic_friction": 0.6,
                    }
                },
            }
        },
    ]

    summary = summarize_friction_statistics(episodes)
    assert summary["gripper"]["static_friction"] == {
        "count": 2,
        "mean": pytest.approx(0.9),
        "min": 0.8,
        "max": 1.0,
    }
    assert summary["object:tomato_sauce_1"]["dynamic_friction"] == {
        "count": 2,
        "mean": pytest.approx(0.5),
        "min": 0.4,
        "max": 0.6,
    }


def test_resolve_episode_termination_reports_object_damage_term():
    reason, term = resolve_episode_termination(
        info={
            "log": {
                "Episode_Termination/object_damage_tomato_sauce_1": 1.0,
                "Episode_Termination/object_1_dropped": 0.0,
            }
        },
        terminated=True,
        truncated=False,
    )

    assert reason == "object_damage"
    assert term == "object_damage_tomato_sauce_1"


def test_resolve_episode_termination_preserves_non_damage_and_timeout_cases():
    assert resolve_episode_termination(
        info={"log": {"Episode_Termination/object_1_dropped": 1.0}},
        terminated=True,
        truncated=False,
    ) == ("terminated", "object_1_dropped")
    assert resolve_episode_termination(
        info={}, terminated=False, truncated=True
    ) == ("truncated", None)
    assert resolve_episode_termination(
        info={}, terminated=False, truncated=False
    ) == ("running", None)


def test_extract_object_damage_details_is_json_safe_and_keeps_default_count():
    details = extract_object_damage_details(
        info={
            "object_damage": {
                "tomato_sauce_1": {
                    "max_squeeze_force": 1.0,
                    "consecutive_frames": 4,
                    "consecutive_count": np.asarray([4]),
                    "measured_squeeze_force": np.asarray([12.5]),
                }
            }
        },
        terminal_term="object_damage_tomato_sauce_1",
    )

    assert details == {
        "object_name": "tomato_sauce_1",
        "mode": "fixed",
        "mass_kg": None,
        "gravity_m_s2": None,
        "gripper_static_friction": None,
        "object_static_friction": None,
        "effective_static_friction": None,
        "tolerance_factor": None,
        "max_squeeze_force": 1.0,
        "consecutive_frames": 4,
        "consecutive_count": 4,
        "measured_squeeze_force": 12.5,
    }


def test_extract_mass_friction_damage_threshold_snapshot_is_json_safe():
    snapshot = extract_damage_threshold_snapshot(
        info={
            "object_damage_threshold": {
                "tomato_sauce_1": {
                    "mode": "mass_friction",
                    "mass_kg": np.asarray([0.1]),
                    "gravity_m_s2": np.asarray([9.81]),
                    "gripper_static_friction": np.asarray([0.5]),
                    "object_static_friction": np.asarray([0.7]),
                    "effective_static_friction": np.asarray([0.6]),
                    "tolerance_factor": 1.1,
                    "max_squeeze_force": np.asarray([1.7985]),
                    "consecutive_frames": 4,
                }
            }
        }
    )

    assert snapshot == {
        "object_name": "tomato_sauce_1",
        "mode": "mass_friction",
        "mass_kg": pytest.approx(0.1),
        "gravity_m_s2": pytest.approx(9.81),
        "gripper_static_friction": pytest.approx(0.5),
        "object_static_friction": pytest.approx(0.7),
        "effective_static_friction": pytest.approx(0.6),
        "tolerance_factor": pytest.approx(1.1),
        "max_squeeze_force": pytest.approx(1.7985),
        "consecutive_frames": 4,
    }


def test_extract_fixed_damage_threshold_snapshot_keeps_audited_random_mass():
    snapshot = extract_damage_threshold_snapshot(
        info={
            "object_damage_threshold": {
                "tomato_sauce_1": {
                    "mode": "fixed",
                    "mass_kg": np.asarray([1.25]),
                    "gravity_m_s2": np.asarray([np.nan]),
                    "gripper_static_friction": np.asarray([np.nan]),
                    "object_static_friction": np.asarray([np.nan]),
                    "effective_static_friction": np.asarray([np.nan]),
                    "tolerance_factor": None,
                    "max_squeeze_force": np.asarray([1_000_000.0]),
                    "consecutive_frames": 4,
                }
            }
        }
    )

    assert snapshot == {
        "object_name": "tomato_sauce_1",
        "mode": "fixed",
        "mass_kg": pytest.approx(1.25),
        "gravity_m_s2": None,
        "gripper_static_friction": None,
        "object_static_friction": None,
        "effective_static_friction": None,
        "tolerance_factor": None,
        "max_squeeze_force": pytest.approx(1_000_000.0),
        "consecutive_frames": 4,
    }


def test_summarize_damage_threshold_statistics_tracks_reset_values():
    episodes = [
        {
            "damage_threshold": {
                "object_name": "tomato_sauce_1",
                "mass_kg": 0.08,
                "effective_static_friction": 0.5,
                "max_squeeze_force": 1.72656,
            }
        },
        {
            "damage_threshold": {
                "object_name": "tomato_sauce_1",
                "mass_kg": 0.12,
                "effective_static_friction": 0.6,
                "max_squeeze_force": 2.1582,
            }
        },
    ]

    summary = summarize_damage_threshold_statistics(episodes)["tomato_sauce_1"]
    assert summary["mass_kg"] == {
        "count": 2,
        "mean": pytest.approx(0.1),
        "min": 0.08,
        "max": 0.12,
    }
    assert summary["effective_static_friction"]["count"] == 2
    assert summary["max_squeeze_force"]["mean"] == pytest.approx(1.94238)


def _force_sequences(num_steps: int = 20):
    strengths = np.arange(1, num_steps + 1, dtype=np.float32)
    left = np.stack([strengths, np.zeros_like(strengths), strengths], axis=1)
    right = np.stack([np.zeros_like(strengths), np.zeros_like(strengths), -strengths], axis=1)
    actions = np.zeros((num_steps, 13), dtype=np.float32)
    actions[:, 7:10] = left
    actions[:, 10:13] = right
    squeeze = 2.0 * strengths
    applied = strengths
    return actions, left, right, squeeze, applied


def test_full_episode_force_summary_preserves_eight_legacy_formulas():
    actions, left, right, squeeze, applied = _force_sequences()

    summary = summarize_episode_force_metrics(
        force_mode="full",
        env_steps=20,
        actions_13d=actions,
        squeeze_pred_values=squeeze,
        squeeze_meas_values=squeeze,
        ap_pred_values=applied,
        ap_meas_values=applied,
        fL_meas_values=left,
        fR_meas_values=right,
    )

    assert summary["force_status"] == "complete"
    assert summary["squeeze_avg_pred"] == pytest.approx(21.0)
    assert summary["squeeze_avg_meas"] == pytest.approx(21.0)
    assert summary["squeeze_max_pred"] == pytest.approx(40.0)
    assert summary["squeeze_max_meas"] == pytest.approx(40.0)
    assert summary["ap_avg_pred"] == pytest.approx(10.5)
    assert summary["ap_avg_meas"] == pytest.approx(10.5)
    assert summary["ap_max_pred"] == pytest.approx(20.0)
    assert summary["ap_max_meas"] == pytest.approx(20.0)
    assert summary["force_samples"]["predicted_action_steps"] == 20
    assert summary["force_samples"]["measured_force_steps"] == 20
    assert summary["force_samples"]["predicted_contact_steps"] == 20
    assert summary["force_samples"]["measured_contact_steps"] == 20
    assert summary["force_samples"]["coverage_ratio"] == pytest.approx(1.0)


def test_top_five_percent_uses_nonzero_frames_and_all_zero_force_is_complete():
    actions = np.zeros((5, 13), dtype=np.float32)
    forces = np.zeros((5, 3), dtype=np.float32)
    zeros = np.zeros(5, dtype=np.float32)

    summary = summarize_episode_force_metrics(
        force_mode="full",
        env_steps=5,
        actions_13d=actions,
        squeeze_pred_values=zeros,
        squeeze_meas_values=zeros,
        ap_pred_values=zeros,
        ap_meas_values=zeros,
        fL_meas_values=forces,
        fR_meas_values=forces,
    )

    assert summary["force_status"] == "complete"
    assert all(summary[key] == pytest.approx(0.0) for key in FORCE_METRIC_KEYS)
    assert summary["force_samples"]["predicted_contact_steps"] == 0
    assert summary["force_samples"]["measured_contact_steps"] == 0
    assert summary["force_samples"]["predicted_contact_ratio"] == pytest.approx(0.0)


def test_missing_force_samples_are_partial_and_not_applicable_mode_keeps_steps():
    actions, left, right, squeeze, applied = _force_sequences(num_steps=3)
    partial = summarize_episode_force_metrics(
        force_mode="full",
        env_steps=3,
        actions_13d=actions[:2],
        squeeze_pred_values=squeeze[:2],
        squeeze_meas_values=squeeze,
        ap_pred_values=applied[:2],
        ap_meas_values=applied,
        fL_meas_values=left,
        fR_meas_values=right,
    )
    not_applicable = summarize_episode_force_metrics(
        force_mode="not_applicable",
        env_steps=3,
    )

    assert partial["force_status"] == "partial"
    assert partial["force_samples"]["coverage_ratio"] == pytest.approx(2 / 3)
    assert not_applicable["force_status"] == "not_applicable"
    assert not_applicable["force_samples"]["coverage_ratio"] is None
    assert all(not_applicable[key] is None for key in FORCE_METRIC_KEYS)


def test_success_only_force_aggregation_and_step_groups_use_independent_counts():
    successful = {
        "success": True,
        "env_steps": 11,
        "inference_chunks": 2,
        **{key: float(index + 1) for index, key in enumerate(FORCE_METRIC_KEYS)},
    }
    successful_missing_one = {
        "success": True,
        "env_steps": 20,
        "inference_chunks": 2,
        **{key: 10.0 for key in FORCE_METRIC_KEYS},
    }
    successful_missing_one["ap_max_meas"] = None
    failed = {
        "success": False,
        "env_steps": 30,
        "inference_chunks": 3,
        **{key: 1000.0 for key in FORCE_METRIC_KEYS},
    }

    aggregates, counts = aggregate_success_force_metrics(
        [successful, successful_missing_one, failed]
    )
    steps = summarize_step_statistics([successful, successful_missing_one, failed])

    assert aggregates["squeeze_avg_pred"] == pytest.approx(5.5)
    assert counts["squeeze_avg_pred"] == 2
    assert aggregates["ap_max_meas"] == pytest.approx(8.0)
    assert counts["ap_max_meas"] == 1
    assert steps["all"]["env_steps_total"] == 61
    assert steps["successful"]["episodes"] == 2
    assert steps["failed"]["inference_chunks_total"] == 3


def test_episode_validation_detects_duplicate_missing_partial_and_illegal_steps():
    episodes = [
        {
            "experiment_index": 0,
            "success": True,
            "env_steps": 11,
            "inference_chunks": 2,
            "force_status": "complete",
        },
        {
            "experiment_index": 0,
            "success": False,
            "env_steps": 21,
            "inference_chunks": 2,
            "force_status": "partial",
        },
    ]

    warnings = validate_episode_records(
        episodes,
        expected_count=2,
        replan_steps=10,
        max_inference_chunks=2,
    )

    assert any("duplicate experiment_index" in warning for warning in warnings)
    assert any("do not cover" in warning for warning in warnings)
    assert any("outside" in warning for warning in warnings)
    assert any("partial force coverage" in warning for warning in warnings)
