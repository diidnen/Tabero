import ast
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools import run_task_evaluations as rte


def _episode_record(index: int, *, success: bool, env_steps: int, chunks: int) -> dict:
    return {
        "experiment_index": index,
        "hdf5_episode_index": index + 10,
        "success": success,
        "end_reason": "success" if success else "max_inference_steps",
        "env_steps": env_steps,
        "inference_chunks": chunks,
        "force_status": "complete",
        "force_samples": {
            "predicted_action_steps": env_steps,
            "measured_force_steps": env_steps,
            "squeeze_pred_steps": env_steps,
            "squeeze_meas_steps": env_steps,
            "ap_pred_steps": env_steps,
            "ap_meas_steps": env_steps,
            "predicted_contact_steps": env_steps,
            "measured_contact_steps": env_steps,
            "predicted_contact_ratio": 1.0,
            "measured_contact_ratio": 1.0,
            "coverage_ratio": 1.0,
        },
        "squeeze_avg_pred": 1.0,
        "squeeze_avg_meas": 2.0,
        "squeeze_max_pred": 3.0,
        "squeeze_max_meas": 4.0,
        "ap_avg_pred": 5.0,
        "ap_avg_meas": 6.0,
        "ap_max_pred": 7.0,
        "ap_max_meas": 8.0,
        "trajectory_mean_measured_squeeze": 2.0,
        "trajectory_force_valid_samples": 4,
        "trajectory_force_status": "complete",
        "trajectory_force_error": None,
        "grasp_started": True,
        "grasp_start_step": 1,
        "force_tracking": {
            "status": "complete",
            "semantics": {
                "population": "reward_force_valid_step",
                "predicted": "raw_model_squeeze_target",
                "effective_target": "controller_target_after_feed_forward_or_override",
                "measured": "raw_unfiltered_gripper_local_squeeze",
            },
            "sample_counts": {
                "env_steps": env_steps,
                "available_steps": env_steps,
                "unavailable_steps": 0,
                "reward_valid_steps": 4,
                "reward_valid_available_steps": 4,
            },
            "reward_valid": {
                "mean_predicted_squeeze_n": 1.0,
                "mean_effective_squeeze_target_n": 1.9,
                "mean_measured_squeeze_raw_n": 2.0,
                "mean_model_target_error_n": -1.0,
                "mean_effective_target_error_n": -0.1,
                "effective_target_mae_n": 0.1,
                "effective_target_rmse_n": 0.1,
            },
            "gripper_position_command": {
                "closed_limit_m": 0.0,
                "open_limit_m": 0.04,
                "saturation_epsilon_m": 1e-8,
                "lower_saturation_steps": 4,
                "upper_saturation_steps": 0,
                "lower_saturation_ratio": 4 / env_steps,
                "upper_saturation_ratio": 0.0,
                "reward_valid_lower_saturation_steps": 4,
                "reward_valid_upper_saturation_steps": 0,
                "reward_valid_lower_saturation_ratio": 1.0,
                "reward_valid_upper_saturation_ratio": 0.0,
            },
        },
        "trace_status": "disabled",
        "trace_rows": 0,
        "trace_error": None,
    }


def test_episode_metrics_parser_reports_malformed_structured_lines():
    payload, warning = rte.parse_episode_metrics_line("[Episode-Metrics] {bad json}\n")

    assert payload is None
    assert warning is not None
    assert "invalid Episode-Metrics JSON" in warning


def test_libero_path_under_tabero_checkout_does_not_auto_enable_tabero_subset():
    hdf5_folder = Path("/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5")

    assert not rte._should_auto_use_tabero_tasks(hdf5_folder, "")


def test_tabero_dataset_path_auto_enables_tabero_subset():
    hdf5_folder = Path("/datasets/tabero_force/replayed_demos")

    assert rte._should_auto_use_tabero_tasks(hdf5_folder, "")


def test_openpi_camera_names_cli_accepts_multiple_values():
    source = (ROOT / "benchmarks/openpi/openpi_inference_client.py").read_text()
    module = ast.parse(source)
    annotation = None

    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "OpenpiClientArguments":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id == "camera_names":
                        annotation = ast.unparse(item.annotation)

    assert annotation == "tuple[str, ...]"


def test_openpi_command_passes_lift_detection_contract():
    config = rte.EvaluationConfig(
        policy_model="openpi",
        lift_height_threshold_m=0.03,
        lift_hold_steps=5,
    )

    command = rte.build_command(config, "libero_object", 0)

    assert "--lift_height_threshold_m" in command
    assert command[command.index("--lift_height_threshold_m") + 1] == "0.03"
    assert "--lift_hold_steps" in command
    assert command[command.index("--lift_hold_steps") + 1] == "5"
    assert command[
        command.index("--reward_force_contact_epsilon_n") + 1
    ] == "1.0"
    assert command[
        command.index("--reward_force_min_valid_samples") + 1
    ] == "4"


def test_openpi_osc_actions_are_sent_as_7d_actions():
    source = (ROOT / "benchmarks/openpi/openpi_inference_client.py").read_text()

    osc_branch_start = source.find('elif args.control_mode == "osc":')
    assert osc_branch_start != -1

    next_branch_start = source.find('elif args.control_mode == "binary":', osc_branch_start)
    assert next_branch_start != -1

    osc_branch = source[osc_branch_start:next_branch_start]
    assert "action_chunk[:, :7]" in osc_branch
    assert "axisangle2quat" not in osc_branch


def test_openpi_sim_launcher_args_are_passed_to_client_command():
    config = rte.EvaluationConfig(
        policy_model="openpi",
        sim_device="cuda:0",
        sim_multi_gpu=True,
        sim_kit_args="--/renderer/activeGpu=8",
    )

    cmd = rte.build_command(config, "libero_object", 0)

    assert "--sim_device" in cmd
    assert cmd[cmd.index("--sim_device") + 1] == "cuda:0"
    assert "--sim_multi_gpu" in cmd
    assert "--sim_kit_args" not in cmd
    assert "--sim_kit_args=--/renderer/activeGpu=8" in cmd


def test_openpi_video_args_are_passed_to_task_specific_directory(tmp_path):
    config = rte.EvaluationConfig(
        policy_model="openpi",
        output_dir=tmp_path / "raw",
        record_videos=True,
    )

    cmd = rte.build_command(config, "libero_object", 0)

    assert "--record_videos" in cmd
    output_index = cmd.index("--record_camera_output_path") + 1
    assert cmd[output_index] == str(
        tmp_path / "raw/videos/libero_object_task0"
    )


def test_openpi_video_args_honor_explicit_output_directory(tmp_path):
    video_dir = tmp_path / "videos"
    config = rte.EvaluationConfig(
        policy_model="openpi",
        record_videos=True,
        record_camera_output_path=video_dir,
    )

    cmd = rte.build_command(config, "libero_object", 0)

    assert cmd[cmd.index("--record_camera_output_path") + 1] == str(video_dir)


def test_config_path_is_passed_to_openpi_client_and_isaac_environment(monkeypatch, tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "libero_object.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": 0,
                        "task_name": "task",
                        "language_instruction": "instruction",
                    }
                ]
            }
        )
    )
    config = rte.EvaluationConfig(
        policy_model="openpi",
        config_path=profile_dir,
        num_total_experiments=1,
    )

    command = rte.build_command(config, "libero_object", 0)
    assert command[command.index("--task_config_path") + 1] == str(profile_dir)

    captured = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = io.StringIO(
                "Total experiments: 1\n"
                "Successful experiments: 0\n"
                "Success rate: 0.00%\n"
            )

        def wait(self, timeout):
            assert timeout == 3600

    def fake_popen(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(rte.subprocess, "Popen", fake_popen)
    rte.run_single_evaluation(config, "libero_object", 0, ROOT)

    assert captured["env"]["LIBERO_CONFIG_DIR"] == str(profile_dir.resolve())


def test_dsrl_raw_image_flag_is_opt_in_for_openpi_client_command():
    default_cmd = rte.build_command(rte.EvaluationConfig(policy_model="openpi"), "libero_object", 0)
    enabled_cmd = rte.build_command(
        rte.EvaluationConfig(policy_model="openpi", send_dsrl_raw_image=True),
        "libero_object",
        0,
    )

    assert "--send-dsrl-raw-image" not in default_cmd
    assert enabled_cmd.count("--send-dsrl-raw-image") == 1


def test_openpi_client_passes_sim_launcher_args_to_app_launcher():
    source = (ROOT / "benchmarks/openpi/openpi_inference_client.py").read_text()
    module = ast.parse(source)
    fields = set()

    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "OpenpiClientArguments":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.add(item.target.id)

    assert "sim_device" in fields
    assert "sim_multi_gpu" in fields
    assert "sim_kit_args" in fields

    assert "multi_gpu=args.sim_multi_gpu" in source
    assert "kit_args=args.sim_kit_args" in source


def test_libero_horizon_policy_is_consistent_in_command_and_json(tmp_path):
    config = rte.EvaluationConfig(policy_model="openpi", max_inference_steps=80)

    assert rte.effective_max_inference_steps(config, "libero_10") == 50
    assert rte.effective_max_inference_steps(config, "libero_object") == 30

    command = rte.build_command(config, "libero_object", 0)
    assert command[command.index("--max_inference_steps") + 1] == "30"

    output_path = tmp_path / "result.json"
    result = {
        "task_suite": "libero_object",
        "task_id": 0,
        "task_name": "pick_up_object",
        "language_instruction": "pick up the object",
        "success_rate": 50.0,
        "successful_experiments": 1,
        "total_experiments": 2,
        "max_inference_steps": 30,
        "execution_time": 1.0,
        "status": "completed",
    }
    rte.save_success_rates_json([result], output_path, config)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["metadata"]["max_inference_steps_policy"]["libero_10"] == 50
    assert saved["metadata"]["max_inference_steps_policy"]["libero_object"] == 30
    assert saved["results"]["libero_object_task0"]["max_inference_steps"] == 30


def test_plain_prompt_evaluation_omits_adverbs_and_records_metadata(tmp_path):
    config = rte.EvaluationConfig(
        policy_model="openpi",
        prompt_adverb="",
        prompt_adverbs=(),
        prompt_seed=0,
    )

    command = rte.build_command(config, "libero_object", 6)

    assert "--prompt_adverb" not in command
    assert "--prompt_adverbs" not in command

    output_path = tmp_path / "result.json"
    result = {
        "task_suite": "libero_object",
        "task_id": 6,
        "task_name": "pick_up_the_butter_and_place_it_in_the_basket",
        "language_instruction": "pick up the butter and place it in the basket",
        "success_rate": 70.0,
        "successful_experiments": 7,
        "total_experiments": 10,
        "max_inference_steps": 30,
        "execution_time": 1.0,
        "status": "completed",
    }
    rte.save_success_rates_json([result], output_path, config)
    metadata = json.loads(output_path.read_text(encoding="utf-8"))["metadata"]

    assert metadata["prompt_mode"] == "plain"
    assert metadata["prompt_adverb"] == ""
    assert metadata["prompt_adverbs"] == []
    assert metadata["prompt_seed"] == 0


def test_zero_exit_without_complete_metrics_is_failed(monkeypatch, capsys):
    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO("CUDA error: invalid device ordinal\n")
            self.returncode = 0

        def wait(self, timeout):
            assert timeout == 3600

    monkeypatch.setattr(rte.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    config = rte.EvaluationConfig(policy_model="openpi", num_total_experiments=1)

    result = rte.run_single_evaluation(
        config,
        task_suite="libero_object",
        task_id=0,
        workspace_root=ROOT,
    )

    assert result["return_code"] == 0
    assert result["success_rate"] is None
    assert result["status"] == "failed"
    assert result["metrics_status"] == "partial"
    assert result["episodes"] == []
    assert "TASK FAILED: libero_object - Task 0" in capsys.readouterr().out


def test_nonzero_exit_with_complete_success_and_episode_metrics_is_completed(monkeypatch):
    records = [
        _episode_record(0, success=True, env_steps=11, chunks=2),
        _episode_record(1, success=False, env_steps=20, chunks=2),
    ]
    stdout = "".join(
        f"[Episode-Metrics] {json.dumps(record)}\n" for record in records
    ) + (
        "Evaluation Results:\n"
        "Total experiments: 2\n"
        "Successful experiments: 1\n"
        "Success rate: 50.00%\n"
        "[Hybrid] Task avg squeeze_pred=1.0000, squeeze_meas=2.0000 over 1 successes\n"
        "[Hybrid-Metrics] Task contact_metrics squeeze_max_mean=3.0000, "
        "app_max_mean=7.0000, app_mean_mean=5.0000, squeeze_max_meas_mean=4.0000, "
        "ap_max_meas_mean=8.0000, ap_mean_meas_mean=6.0000 over 1 successes\n"
    )

    class FakeProcess:
        def __init__(self):
            self.returncode = 17
            self.stdout = io.StringIO(stdout)

        def wait(self, timeout):
            assert timeout == 3600

    monkeypatch.setattr(rte.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    config = rte.EvaluationConfig(
        policy_model="openpi",
        control_mode="tactile",
        num_total_experiments=2,
    )

    result = rte.run_single_evaluation(config, "libero_object", 1, ROOT)

    assert result["return_code"] == 17
    assert result["status"] == "completed"
    assert result["metrics_status"] == "complete"
    assert [episode["success"] for episode in result["episodes"]] == [True, False]
    assert result["step_statistics"]["all"]["env_steps_total"] == 31
    assert result["force_metric_episode_counts"] == dict.fromkeys(rte.FORCE_METRIC_KEYS, 1)


def test_duplicate_missing_and_partial_force_records_do_not_change_success_status(monkeypatch):
    first = _episode_record(0, success=True, env_steps=10, chunks=1)
    duplicate = _episode_record(0, success=False, env_steps=10, chunks=1)
    duplicate["force_status"] = "partial"
    stdout = (
        f"[Episode-Metrics] {json.dumps(first)}\n"
        f"[Episode-Metrics] {json.dumps(duplicate)}\n"
        "Total experiments: 2\n"
        "Successful experiments: 1\n"
        "Success rate: 50.00%\n"
    )

    class FakeProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = io.StringIO(stdout)

        def wait(self, timeout):
            pass

    monkeypatch.setattr(rte.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    config = rte.EvaluationConfig(num_total_experiments=2)

    result = rte.run_single_evaluation(config, "libero_object", 0, ROOT)

    assert result["status"] == "completed"
    assert result["metrics_status"] == "partial"
    assert any("duplicate experiment_index" in warning for warning in result["metrics_warnings"])
    assert any("partial force coverage" in warning for warning in result["metrics_warnings"])


def test_step_trace_cli_resolution_command_and_validation(tmp_path):
    config = rte.EvaluationConfig(
        policy_model="openpi",
        record_step_traces=True,
        num_total_experiments=1,
        output_dir=tmp_path,
    )
    root = rte.resolve_step_trace_root(
        config,
        model_name="openpi_tactile",
        timestamp="20260801_010203",
    )
    assert root == tmp_path / "step_traces_openpi_tactile_20260801_010203"

    trace_path = root / "libero_object_task1.jsonl"
    command = rte.build_command(config, "libero_object", 1, step_trace_path=trace_path)
    assert command[command.index("--step_trace_path") + 1] == str(trace_path)

    episode = _episode_record(0, success=True, env_steps=4, chunks=1)
    episode["trace_status"] = "complete"
    episode["trace_rows"] = 4
    trace_path.parent.mkdir(parents=True)
    rows = []
    for step in range(4):
        rows.append(
            {
                "experiment_index": 0,
                "env_step_index": step,
                "inference_chunk_index": 0,
                "action_in_chunk_index": step,
                "fL_pred_local": [0.0, 0.0, 1.0],
                "fR_pred_local": [0.0, 0.0, -1.0],
                "fL_meas_local": [0.0, 0.0, 1.0],
                "fR_meas_local": [0.0, 0.0, -1.0],
                "squeeze_pred": 2.0,
                "squeeze_meas": 2.0,
                "ap_pred": 0.0,
                "ap_meas": 0.0,
                "predicted_contact": True,
                "measured_contact": True,
                "target_contact_override_enabled": True,
                "target_contact_detected": bool(step),
                "target_contact_override_latched": bool(step),
                "target_contact_activation_step": 1 if step else -1,
                "target_contact_force_norm": float(step),
                "effective_single_finger_target_n": 13.0 if step else 2.0,
                "effective_squeeze_target_n": 26.0 if step else 4.0,
                "reward_grasp_observed": step == 0,
                "reward_grasp_terms_active": ["grasp_1"] if step == 0 else [],
                "reward_grasp_started": True,
                "reward_force_valid_step": True,
                "reward_squeeze_meas_raw": 2.0,
            }
        )
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    descriptor, warnings = rte.validate_step_trace(
        trace_path,
        [episode],
        enabled=True,
        output_dir=tmp_path,
        replan_steps=10,
    )
    assert warnings == []
    assert descriptor == {
        "enabled": True,
        "path": "step_traces_openpi_tactile_20260801_010203/libero_object_task1.jsonl",
        "rows": 4,
        "status": "complete",
    }
    _, artifacts, _ = rte.collect_episode_metric_artifacts(
        config,
        [episode],
        [],
        trace_path,
        max_inference_steps=30,
        successful_experiments=1,
    )
    assert artifacts["metrics_status"] == "complete"


def test_step_trace_dir_without_record_flag_is_rejected(tmp_path):
    config = rte.EvaluationConfig(step_trace_dir=tmp_path / "traces")

    with pytest.raises(ValueError, match="requires --record-step-traces"):
        rte.resolve_step_trace_root(config, model_name="openpi_tactile", timestamp="stamp")


def test_json_and_txt_include_episode_metrics_and_na(tmp_path):
    episode = _episode_record(0, success=False, env_steps=10, chunks=1)
    episode["squeeze_avg_pred"] = None
    episode["friction"] = {
        "gripper": {"static_friction": 0.5, "dynamic_friction": 0.5},
        "objects": {
            "cream_cheese_1": {"static_friction": 0.5, "dynamic_friction": 0.5}
        },
    }
    episode["damage_threshold"] = {
        "object_name": "cream_cheese_1",
        "mode": "mass_friction",
        "mass_kg": 0.2,
        "gravity_m_s2": 9.81,
        "gripper_static_friction": 0.5,
        "object_static_friction": 0.5,
        "effective_static_friction": 0.5,
        "tolerance_factor": 1.1,
        "max_squeeze_force": 4.3164,
        "consecutive_frames": 4,
    }
    config = rte.EvaluationConfig(num_total_experiments=1, replan_steps=10)
    result = {
        "task_suite": "libero_object",
        "task_id": 1,
        "task_name": "task",
        "language_instruction": "instruction",
        "success_rate": 0.0,
        "successful_experiments": 0,
        "total_experiments": 1,
        "max_inference_steps": 30,
        "execution_time": 1.0,
        "status": "completed",
        "metrics_status": "complete",
        "metrics_warnings": [],
        "step_statistics": rte.summarize_step_statistics([episode]),
        "force_metric_episode_counts": dict.fromkeys(rte.FORCE_METRIC_KEYS, 0),
        **rte.summarize_trajectory_force_metrics([episode]),
        "damage_threshold_statistics": rte.summarize_damage_threshold_statistics(
            [episode]
        ),
        "friction_statistics": rte.summarize_friction_statistics([episode]),
        "episodes": [episode],
        "step_trace": {"enabled": False, "path": None, "rows": 0, "status": "disabled"},
    }
    json_path = tmp_path / "result.json"
    txt_path = tmp_path / "result.txt"

    rte.save_success_rates_json([result], json_path, config)
    rte.save_success_rates_txt([result], txt_path, config)

    json_result = json.loads(json_path.read_text())
    assert json_result["metadata"]["episode_metrics_schema_version"] == 5
    assert json_result["metadata"]["reward_force_contact_epsilon_n"] == 1.0
    assert json_result["metadata"]["reward_force_min_valid_samples"] == 4
    task = json_result["results"]["libero_object_task1"]
    text = txt_path.read_text()
    assert task["metrics_status"] == "complete"
    assert task["episodes"][0]["env_steps"] == 10
    assert task["episodes"][0]["friction"]["gripper"]["static_friction"] == 0.5
    assert task["friction_statistics"]["gripper"]["dynamic_friction"]["mean"] == 0.5
    assert task["episodes"][0]["damage_threshold"]["max_squeeze_force"] == 4.3164
    assert task["damage_threshold_statistics"]["cream_cheese_1"][
        "max_squeeze_force"
    ]["count"] == 1
    assert "Per-experiment step and force coverage:" in text
    assert "Friction statistics:" in text
    assert "Per-experiment friction:" in text
    assert "Damage threshold statistics:" in text
    assert "Per-experiment damage threshold:" in text
    assert "cream_cheese_1 | mass_friction | 0.2000" in text
    assert "cream_cheese_1 | 0.5000 | 0.5000" in text
    assert "Per-experiment force metrics:" in text
    assert "Per-experiment force-target tracking:" in text
    assert "Reward-aligned trajectory force:" in text
    assert "Per-experiment reward-aligned trajectory force:" in text
    assert task["success_trajectory_force_episode_count"] == 0
    assert task["all_eligible_trajectory_mean_measured_squeeze"] == 2.0
    assert "N/A" in text
