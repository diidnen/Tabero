# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run policy inference evaluation on Libero task suites and record success rates."""

import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import tyro

# 确保项目根目录在 sys.path 中，使得在不同运行方式下都能导入 `scripts.tools`。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.episode_metrics import (
    FORCE_METRIC_KEYS,
    aggregate_success_force_metrics,
    summarize_damage_threshold_statistics,
    summarize_friction_statistics,
    summarize_step_statistics,
    summarize_trajectory_force_metrics,
    validate_episode_records,
)


def _load_tabero_task_subset(workspace_root: Path) -> dict[str, list[int]]:
    """Load Tabero task subset mapping from JSON.

    Expected path:
      benchmarks/datasets/tabero/config/tabero_tasks.json
    """
    path = workspace_root / "benchmarks" / "datasets" / "tabero" / "config" / "tabero_tasks.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Normalize values to list[int]
        out: dict[str, list[int]] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, list):
                    out[k] = [int(x) for x in v]
        return out
    except Exception:
        return {}


def get_task_suites_and_tasks() -> dict[str, list[int]]:
    """Get all available task suites and their task IDs."""
    return {
        "libero_10": list(range(10)),
        "libero_spatial": list(range(10)),
        "libero_object": list(range(10)),
        "libero_goal": list(range(10)),
    }


def get_task_name_and_instruction(task_suite: str, task_id: int, config_base_path: Path) -> tuple[str, str]:
    """Get task name and language instruction from config file."""
    config_path = config_base_path / f"{task_suite}.json"
    if not config_path.exists():
        return f"task_{task_id}", f"Task {task_id}"

    with open(config_path) as f:
        config = json.load(f)

    for task in config["tasks"]:
        if task["task_id"] == task_id:
            return task["task_name"], task["language_instruction"]

    return f"task_{task_id}", f"Task {task_id}"


def _should_auto_use_tabero_tasks(hdf5_folder: Optional[Path], replayed_demos_dir: str) -> bool:
    """Infer Tabero subset mode from dataset directory names, not parent checkout names."""
    for raw_path in (hdf5_folder, replayed_demos_dir):
        if not raw_path:
            continue
        parts = [part.lower() for part in Path(raw_path).parts]
        if "tabero_force" in parts:
            return True
        for idx, part in enumerate(parts[:-1]):
            if part == "datasets" and parts[idx + 1] in {"tabero", "tabero_force"}:
                return True
    return False


@dataclass
class EvaluationConfig:
    """Configuration for running task evaluations."""

    policy_model: str = "openpi"
    control_mode: str = "diffik"
    # OpenPI tactile 消融：透传到 openpi client 的 `--abs7d`
    # （tactile 观测/模型分支不变，但动作按绝对 7D 执行，补 0 到 13D，并关闭 pos_kp/squeeze_kp 修正）
    abs7d: bool = False
    # 任务环境 ID：
    # - 默认留空：由 `benchmarks/openpi/openpi_inference_client.py` 根据 control_mode 自动选择：
    #   * diffik  -> Isaac-Libero-Franka-IK-v0
    #   * osc     -> Isaac-Libero-Franka-OscPose-v0
    #   * hybrid  -> Isaac-Libero-Franka-Hybrid-ContactForce-v0      (13D 力–位混合控制 + contact forces)
    #   * tactile -> Isaac-Libero-Franka-Hybrid-Tactile-v0           (13D 力–位混合控制 + 触觉)
    # - 如需自定义环境，可通过 CLI 显式传入 `--task xxx`。
    task: str = ""

    server_host: str = "127.0.1.1"
    server_port: int = 8000

    num_total_experiments: int = 50
    num_success_steps: int = 8
    max_inference_steps: int = 80
    replan_steps: int = 10
    lift_height_threshold_m: float = 0.03
    lift_hold_steps: int = 5
    reward_force_contact_epsilon_n: float = 1.0
    reward_force_min_valid_samples: int = 4

    camera_names: tuple[str, ...] = ("agentview_cam", "eye_in_hand_cam")
    target_image_size: tuple[int, int, int] = (224, 224, 3)
    send_dsrl_raw_image: bool = False
    num_steps_wait: int = 5

    task_suites: tuple[str, ...] = ()
    task_ids: tuple[int, ...] = ()

    hdf5_folder: Optional[Path] = None
    config_path: Optional[Path] = None

    output_dir: Path = Path("./evaluation_results")
    output_format: str = "both"
    # Per-episode summaries are always recorded. Full per-env-step force traces are opt-in.
    record_step_traces: bool = False
    step_trace_dir: Optional[Path] = None
    record_videos: bool = False
    record_camera_output_path: Optional[Path] = None

    headless: bool = True
    visualize: bool = False
    # Passed through to the IsaacLab AppLauncher in the OpenPI client.
    # This controls Kit active_gpu/physics_gpu, which may otherwise default to physical GPU0.
    sim_device: str = ""
    # Keep false by default so Isaac/Kit does not initialize all visible GPUs during single-env eval.
    sim_multi_gpu: bool = False
    # Extra Kit args for renderer selection, e.g. "--/renderer/activeGpu=8".
    sim_kit_args: str = ""
    randomize_light: bool = False
    debug_mode: int = 0
    # Optional: override OpenPI client's debug output root (e.g., for debug_mode=6 captures).
    # If empty, OpenPI client uses its own default.
    debug_path: str = ""
    seed: int = 11

    openpi_script: str = "benchmarks/openpi/openpi_inference_client.py"
    gr00t_script: str = "benchmarks/gr00t/gr00t_inference_client.py"

    # 是否使用 Tabero 任务子集（True: 只评估 Tabero 固定列表中的任务；False: 使用原版 Libero 全任务）
    use_tabero_tasks: bool = False

    # Tabero-style prompt rewrite (adverb augmentation). Passed through to OpenPI client.
    prompt_adverb: str = ""
    # 与 convert_all_libero_to_tabero.py 保持一致：strong_adverbs=("firmly","tightly"), soft_adverbs=("gently","softly")
    # 这里作为评估侧的默认候选集合（OpenPI 侧会确定性选择其中一个并决定 prefix/suffix 风格）。
    prompt_adverbs: tuple[str, ...] = ("firmly", "tightly", "gently", "softly")
    prompt_seed: int = 0

    # If True, only evaluate tasks that have matching HDF5 files under hdf5_folder
    # (pattern: <suite>_task<id>_*_demo.hdf5). Useful when you must rely on assembled HDF5 for scene setup.
    require_hdf5: bool = False


def effective_max_inference_steps(
    config: EvaluationConfig, task_suite: str
) -> int:
    """Return the inference horizon actually passed to an evaluation client."""
    if task_suite == "libero_10":
        return 50
    if task_suite in ("libero_goal", "libero_spatial", "libero_object"):
        return 30
    return config.max_inference_steps


def build_command(
    config: EvaluationConfig,
    task_suite: str,
    task_id: int,
    step_trace_path: Optional[Path] = None,
) -> list[str]:
    """Build command line arguments for subprocess."""
    python_script = config.openpi_script if config.policy_model == "openpi" else config.gr00t_script
    max_inference_steps = effective_max_inference_steps(config, task_suite)

    cmd = [
        sys.executable, python_script,
        "--server_host", config.server_host,
        "--server_port", str(config.server_port),
        "--num_total_experiments", str(config.num_total_experiments),
        "--num_success_steps", str(config.num_success_steps),
        "--max_inference_steps", str(max_inference_steps),
        "--task_suite", task_suite,
        "--task_id", str(task_id),
        "--seed", str(config.seed),
        "--lift_height_threshold_m", str(config.lift_height_threshold_m),
        "--lift_hold_steps", str(config.lift_hold_steps),
        "--reward_force_contact_epsilon_n", str(
            config.reward_force_contact_epsilon_n
        ),
        "--reward_force_min_valid_samples", str(
            config.reward_force_min_valid_samples
        ),
    ]

    if config.policy_model == "openpi":
        cmd.extend([
            "--control_mode", config.control_mode,
            "--replan_steps", str(config.replan_steps),
            "--camera_names"] + list(config.camera_names) + [
            "--target_image_size"] + [str(x) for x in config.target_image_size] + [
            "--num_steps_wait", str(config.num_steps_wait),
        ])
        if config.abs7d:
            cmd.append("--abs7d")
        if config.send_dsrl_raw_image:
            cmd.append("--send-dsrl-raw-image")
        if config.task:
            cmd.extend(["--task", config.task])
        if config.config_path:
            cmd.extend(["--task_config_path", str(config.config_path)])
        # Optional prompt rewrite knobs
        # Always pass prompt_seed explicitly (including 0) so prompt behavior is stable and traceable.
        cmd.extend(["--prompt_seed", str(config.prompt_seed)])
        if config.prompt_adverb:
            cmd.extend(["--prompt_adverb", str(config.prompt_adverb)])
        if config.prompt_adverbs:
            cmd.append("--prompt_adverbs")
            cmd.extend([str(x) for x in config.prompt_adverbs])
        if config.randomize_light:
            cmd.append("--randomize_light")
        if step_trace_path is not None:
            cmd.extend(["--step_trace_path", str(step_trace_path)])
        if config.record_videos:
            video_dir = config.record_camera_output_path or (
                config.output_dir / "videos" / f"{task_suite}_task{task_id}"
            )
            cmd.append("--record_videos")
            cmd.extend(["--record_camera_output_path", str(video_dir)])

    if config.headless and not config.visualize:
        cmd.append("--headless")
    if config.sim_device:
        cmd.extend(["--sim_device", str(config.sim_device)])
    if config.sim_multi_gpu:
        cmd.append("--sim_multi_gpu")
    if config.sim_kit_args:
        # Use equals form because Kit args commonly start with "--" and tyro would otherwise
        # parse the value as a new option in the child OpenPI client.
        cmd.append(f"--sim_kit_args={config.sim_kit_args}")

    if config.debug_mode > 0:
        cmd.extend(["--debug_mode", str(config.debug_mode)])
        if config.debug_path:
            cmd.extend(["--debug_path", str(config.debug_path)])

    if config.hdf5_folder:
        cmd.extend(["--hdf5_folder", str(config.hdf5_folder)])

    return cmd


_AVG_PATTERN = re.compile(
    r"squeeze_pred=([+-]?(?:\d+\.?\d*|\d*\.?\d+))(?:[eE][+-]?\d+)?\s*,\s*"
    r"squeeze_meas=([+-]?(?:\d+\.?\d*|\d*\.?\d+))(?:[eE][+-]?\d+)?"
)
_KEYVAL_PATTERN = re.compile(
    r"([a-zA-Z0-9_]+)\s*=\s*([+-]?(?:\d+\.?\d*|\d*\.?\d+))(?:[eE][+-]?\d+)?"
)
_EPISODE_METRICS_PREFIX = "[Episode-Metrics] "


def parse_success_metrics(
    output_lines: list[str],
) -> tuple[
    Optional[float],
    Optional[int],
    Optional[int],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """从 OpenPI 子进程 stdout 中解析成功率和 Hybrid 相关力学指标."""

    success_rate: Optional[float] = None
    successful_experiments: Optional[int] = None
    total_experiments: Optional[int] = None

    # 9 个 metrics，按字典暂存，最后统一取出
    metrics: dict[str, float] = {}

    for line in output_lines:
        if "Success rate:" in line:
            # Success rate: 68.00%
            with suppress(Exception):
                success_rate = float(line.split("Success rate:", 1)[1].strip().replace("%", ""))
            continue

        if "Successful experiments:" in line:
            with suppress(Exception):
                successful_experiments = int(line.split("Successful experiments:", 1)[1].strip())
            continue

        if "Total experiments:" in line:
            with suppress(Exception):
                total_experiments = int(line.split("Total experiments:", 1)[1].strip())
            continue

        if "[Hybrid] Task avg squeeze_pred=" in line:
            # 例子：
            # [Hybrid] Task avg squeeze_pred=0.1234, squeeze_meas=0.5678 over 10 successes
            match = _AVG_PATTERN.search(line)
            if match:
                with suppress(Exception):
                    metrics["squeeze_avg_pred"] = float(match.group(1))
                with suppress(Exception):
                    metrics["squeeze_avg_meas"] = float(match.group(2))
            continue

        if "[Hybrid-Metrics] Task contact_metrics" in line:
            # 例子：
            # [Hybrid-Metrics] Task contact_metrics squeeze_max_mean=0.1, app_max_mean=0.2, ...
            for m in _KEYVAL_PATTERN.finditer(line):
                key, val = m.group(1), m.group(2)
                with suppress(Exception):
                    metrics[key] = float(val)

    return (
        success_rate,
        successful_experiments,
        total_experiments,
        metrics.get("squeeze_avg_pred"),
        metrics.get("squeeze_avg_meas"),
        metrics.get("squeeze_max_mean"),
        metrics.get("app_max_mean"),
        metrics.get("app_mean_mean"),
        metrics.get("squeeze_max_meas_mean"),
        metrics.get("ap_max_meas_mean"),
        metrics.get("ap_mean_meas_mean"),
    )


def parse_episode_metrics_line(line: str) -> tuple[Optional[dict], Optional[str]]:
    """Parse one structured child record without affecting success-rate parsing."""
    stripped = line.strip()
    if not stripped.startswith(_EPISODE_METRICS_PREFIX):
        return None, None
    raw_payload = stripped[len(_EPISODE_METRICS_PREFIX) :]
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        return None, f"invalid Episode-Metrics JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "Episode-Metrics payload is not an object"
    return payload, None


def validate_step_trace(
    path: Optional[Path],
    episodes: list[dict],
    *,
    enabled: bool,
    output_dir: Path,
    replan_steps: int,
) -> tuple[dict, list[str]]:
    """Validate JSONL row coverage and return an additive result descriptor."""
    if not enabled:
        return {"enabled": False, "path": None, "rows": 0, "status": "disabled"}, []

    warnings: list[str] = []
    relative_path = None if path is None else os.path.relpath(path, output_dir)
    descriptor = {"enabled": True, "path": relative_path, "rows": 0, "status": "partial"}
    if path is None or not path.is_file():
        warnings.append(f"step trace file is missing: {path}")
        return descriptor, warnings

    rows_by_episode: dict[int, set[int]] = {}
    reward_values_by_episode: dict[int, list[float]] = {}
    episode_by_index = {
        episode["experiment_index"]: episode
        for episode in episodes
        if type(episode.get("experiment_index")) is int
    }
    required_row_fields = {
        "experiment_index",
        "env_step_index",
        "inference_chunk_index",
        "action_in_chunk_index",
        "fL_pred_local",
        "fR_pred_local",
        "fL_meas_local",
        "fR_meas_local",
        "squeeze_pred",
        "squeeze_meas",
        "ap_pred",
        "ap_meas",
        "predicted_contact",
        "measured_contact",
        "target_contact_override_enabled",
        "target_contact_detected",
        "target_contact_override_latched",
        "target_contact_activation_step",
        "target_contact_force_norm",
        "effective_single_finger_target_n",
        "effective_squeeze_target_n",
        "reward_grasp_observed",
        "reward_grasp_terms_active",
        "reward_grasp_started",
        "reward_force_valid_step",
        "reward_squeeze_meas_raw",
    }
    try:
        with open(path, encoding="utf-8") as trace_file:
            for line_number, line in enumerate(trace_file, start=1):
                if not line.strip():
                    continue
                descriptor["rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(f"step trace line {line_number} is invalid JSON: {exc}")
                    continue
                if not isinstance(row, dict):
                    warnings.append(f"step trace line {line_number} is not an object")
                    continue
                missing_fields = sorted(required_row_fields - row.keys())
                if missing_fields:
                    warnings.append(
                        f"step trace line {line_number} is missing fields: {', '.join(missing_fields)}"
                    )
                for bool_field in (
                    "reward_grasp_observed",
                    "reward_grasp_started",
                    "reward_force_valid_step",
                ):
                    if type(row.get(bool_field)) is not bool:
                        warnings.append(
                            f"step trace line {line_number} has invalid {bool_field}"
                        )
                if not isinstance(row.get("reward_grasp_terms_active"), list):
                    warnings.append(
                        f"step trace line {line_number} has invalid "
                        "reward_grasp_terms_active"
                    )
                experiment_index = row.get("experiment_index")
                env_step_index = row.get("env_step_index")
                if type(experiment_index) is not int or type(env_step_index) is not int:
                    warnings.append(f"step trace line {line_number} has invalid indices")
                    continue
                inference_chunk_index = row.get("inference_chunk_index")
                action_in_chunk_index = row.get("action_in_chunk_index")
                episode = episode_by_index.get(experiment_index)
                if episode is None:
                    warnings.append(
                        f"step trace line {line_number} references unknown episode {experiment_index}"
                    )
                else:
                    inference_chunks = episode.get("inference_chunks")
                    if (
                        type(inference_chunk_index) is not int
                        or type(inference_chunks) is not int
                        or not 0 <= inference_chunk_index < inference_chunks
                    ):
                        warnings.append(
                            f"step trace line {line_number} has invalid inference_chunk_index"
                        )
                    if (
                        type(action_in_chunk_index) is not int
                        or not 0 <= action_in_chunk_index < replan_steps
                    ):
                        warnings.append(
                            f"step trace line {line_number} has invalid action_in_chunk_index"
                        )
                step_indices = rows_by_episode.setdefault(experiment_index, set())
                if env_step_index in step_indices:
                    warnings.append(
                        f"step trace has duplicate env_step_index={env_step_index} for episode {experiment_index}"
                    )
                step_indices.add(env_step_index)
                if row.get("reward_force_valid_step") is True:
                    reward_squeeze = row.get("reward_squeeze_meas_raw")
                    if (
                        isinstance(reward_squeeze, bool)
                        or not isinstance(reward_squeeze, (int, float))
                        or not math.isfinite(float(reward_squeeze))
                    ):
                        warnings.append(
                            f"step trace line {line_number} has invalid "
                            "reward_squeeze_meas_raw for a valid reward-force step"
                        )
                    else:
                        reward_values_by_episode.setdefault(
                            experiment_index, []
                        ).append(float(reward_squeeze))
    except OSError as exc:
        warnings.append(f"failed to read step trace: {exc}")
        return descriptor, warnings

    expected_total = 0
    for episode in episodes:
        experiment_index = episode.get("experiment_index")
        env_steps = episode.get("env_steps")
        if type(experiment_index) is not int or type(env_steps) is not int:
            continue
        expected_total += env_steps
        actual_indices = rows_by_episode.get(experiment_index, set())
        if actual_indices != set(range(env_steps)):
            warnings.append(
                f"step trace episode {experiment_index} does not cover env_step_index 0..{env_steps - 1}"
            )
        reward_values = reward_values_by_episode.get(experiment_index, [])
        if episode.get("trajectory_force_valid_samples") != len(reward_values):
            warnings.append(
                f"step trace episode {experiment_index} reward-force samples="
                f"{len(reward_values)} do not match episode metric="
                f"{episode.get('trajectory_force_valid_samples')!r}"
            )
        episode_mean = episode.get("trajectory_mean_measured_squeeze")
        trace_mean = sum(reward_values) / len(reward_values) if reward_values else None
        if episode_mean is None:
            if trace_mean is not None:
                warnings.append(
                    f"step trace episode {experiment_index} has reward-force mean "
                    f"{trace_mean} while episode metric is None"
                )
        elif trace_mean is None or not math.isclose(
            float(episode_mean), trace_mean, rel_tol=1.0e-7, abs_tol=1.0e-7
        ):
            warnings.append(
                f"step trace episode {experiment_index} reward-force mean="
                f"{trace_mean!r} does not match episode metric={episode_mean!r}"
            )
    if descriptor["rows"] != expected_total:
        warnings.append(f"step trace rows={descriptor['rows']} do not match total env_steps={expected_total}")

    if not warnings:
        descriptor["status"] = "complete"
    return descriptor, warnings


def collect_episode_metric_artifacts(
    config: EvaluationConfig,
    episodes: list[dict],
    episode_parse_warnings: list[str],
    step_trace_path: Optional[Path],
    *,
    max_inference_steps: int,
    successful_experiments: Optional[int] = None,
) -> tuple[list[dict], dict, dict[str, Optional[float]]]:
    """Validate and summarize additive episode metrics independently of task success."""
    episodes = sorted(
        episodes,
        key=lambda episode: (
            episode.get("experiment_index")
            if type(episode.get("experiment_index")) is int
            else config.num_total_experiments
        ),
    )
    metrics_warnings = list(episode_parse_warnings)
    metrics_warnings.extend(
        validate_episode_records(
            episodes,
            expected_count=config.num_total_experiments,
            replan_steps=config.replan_steps,
            max_inference_chunks=max_inference_steps,
        )
    )

    if successful_experiments is not None:
        recorded_successes = sum(episode.get("success") is True for episode in episodes)
        if recorded_successes != successful_experiments:
            metrics_warnings.append(
                f"episode success count={recorded_successes} does not match task summary={successful_experiments}"
            )

    if config.policy_model == "openpi" and config.control_mode in {
        "hybrid",
        "tactile",
        "binary",
    }:
        for episode in episodes:
            if (
                episode.get("success") is True
                and (
                    episode.get("trajectory_force_status") != "complete"
                    or type(episode.get("trajectory_force_valid_samples"))
                    is not int
                    or episode.get("trajectory_force_valid_samples", 0)
                    < config.reward_force_min_valid_samples
                )
            ):
                metrics_warnings.append(
                    "successful episode "
                    f"{episode.get('experiment_index')} has trajectory_force_status="
                    f"{episode.get('trajectory_force_status')!r}, valid_samples="
                    f"{episode.get('trajectory_force_valid_samples')!r}"
                )
            tracking = episode.get("force_tracking")
            if not isinstance(tracking, dict) or tracking.get("status") != "complete":
                metrics_warnings.append(
                    "episode "
                    f"{episode.get('experiment_index')} has force_tracking status="
                    f"{tracking.get('status') if isinstance(tracking, dict) else None!r}"
                )
                continue
            reward_valid_tracking = tracking.get("reward_valid", {})
            tracking_measured = reward_valid_tracking.get(
                "mean_measured_squeeze_raw_n"
            )
            trajectory_measured = episode.get(
                "trajectory_mean_measured_squeeze"
            )
            if trajectory_measured is not None and (
                not isinstance(tracking_measured, (int, float))
                or isinstance(tracking_measured, bool)
                or not math.isclose(
                    float(tracking_measured),
                    float(trajectory_measured),
                    rel_tol=1.0e-7,
                    abs_tol=1.0e-7,
                )
            ):
                metrics_warnings.append(
                    "episode "
                    f"{episode.get('experiment_index')} force_tracking measured mean="
                    f"{tracking_measured!r} does not match trajectory metric="
                    f"{trajectory_measured!r}"
                )

    if config.record_step_traces:
        for episode in episodes:
            experiment_index = episode.get("experiment_index")
            if episode.get("trace_status") != "complete":
                metrics_warnings.append(
                    f"episode {experiment_index} has trace_status={episode.get('trace_status')!r}"
                )
            if episode.get("trace_rows") != episode.get("env_steps"):
                metrics_warnings.append(
                    f"episode {experiment_index} trace_rows={episode.get('trace_rows')!r} "
                    f"does not match env_steps={episode.get('env_steps')!r}"
                )

    if config.record_videos:
        for episode in episodes:
            experiment_index = episode.get("experiment_index")
            video = episode.get("video")
            if not isinstance(video, dict):
                metrics_warnings.append(
                    f"episode {experiment_index} is missing its video descriptor"
                )
                continue
            if video.get("status") != "complete":
                metrics_warnings.append(
                    f"episode {experiment_index} video_status={video.get('status')!r}: "
                    f"{video.get('error')!r}"
                )
            if video.get("frames") != episode.get("env_steps"):
                metrics_warnings.append(
                    f"episode {experiment_index} video frames={video.get('frames')!r} "
                    f"do not match env_steps={episode.get('env_steps')!r}"
                )
            video_path = video.get("path")
            if not isinstance(video_path, str) or not Path(video_path).is_file():
                metrics_warnings.append(
                    f"episode {experiment_index} video file is missing: {video_path!r}"
                )

    step_trace, trace_warnings = validate_step_trace(
        step_trace_path,
        episodes,
        enabled=config.record_step_traces,
        output_dir=config.output_dir,
        replan_steps=config.replan_steps,
    )
    metrics_warnings.extend(trace_warnings)
    force_aggregates, force_metric_episode_counts = aggregate_success_force_metrics(episodes)
    trajectory_force_aggregates = summarize_trajectory_force_metrics(episodes)
    artifacts = {
        "metrics_status": "complete" if not metrics_warnings else "partial",
        "metrics_warnings": metrics_warnings,
        "step_statistics": summarize_step_statistics(episodes),
        "force_metric_episode_counts": force_metric_episode_counts,
        **trajectory_force_aggregates,
        "damage_threshold_statistics": summarize_damage_threshold_statistics(episodes),
        "friction_statistics": summarize_friction_statistics(episodes),
        "episodes": episodes,
        "step_trace": step_trace,
    }
    return episodes, artifacts, force_aggregates


def force_aggregates_to_legacy_fields(aggregates: dict[str, Optional[float]]) -> dict:
    """Map the eight public metric names back to the pre-existing JSON field names."""
    return {
        "avg_squeeze_pred": aggregates["squeeze_avg_pred"],
        "avg_squeeze_meas": aggregates["squeeze_avg_meas"],
        "task_squeeze_max_mean": aggregates["squeeze_max_pred"],
        "task_app_max_mean": aggregates["ap_max_pred"],
        "task_app_mean_mean": aggregates["ap_avg_pred"],
        "task_squeeze_max_meas_mean": aggregates["squeeze_max_meas"],
        "task_ap_max_meas_mean": aggregates["ap_max_meas"],
        "task_ap_mean_meas_mean": aggregates["ap_avg_meas"],
    }


def run_single_evaluation(
    config: EvaluationConfig,
    task_suite: str,
    task_id: int,
    workspace_root: Path,
    step_trace_path: Optional[Path] = None,
) -> dict:
    """Run a single task evaluation and return results."""
    config_base_path = config.config_path or (workspace_root / "benchmarks/datasets/libero/config")
    task_name, language_instruction = get_task_name_and_instruction(task_suite, task_id, config_base_path)

    print(f"\n{'='*80}")
    print(f"Running evaluation for {task_suite} - Task {task_id}")
    print(f"Task Name: {task_name}")
    print(f"Language Instruction: {language_instruction}")
    print(f"{'='*80}")

    max_inference_steps = effective_max_inference_steps(config, task_suite)
    cmd = build_command(config, task_suite, task_id, step_trace_path=step_trace_path)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if config.config_path:
        env["LIBERO_CONFIG_DIR"] = str(config.config_path.resolve())
    if "USE_RELATIVE_MODE" not in env:
        env["USE_RELATIVE_MODE"] = "False"

    start_time = time.time()
    output_lines: list[str] = []
    episodes: list[dict] = []
    episode_parse_warnings: list[str] = []
    process: Optional[subprocess.Popen] = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=workspace_root,
            env=env,
        )

        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line)
                # 屏蔽 OpenPI 内部的部分统计行，只在本脚本里做表格展示
                stripped = line.strip()
                episode_record, episode_warning = parse_episode_metrics_line(line)
                if episode_record is not None:
                    episodes.append(episode_record)
                    print(line, end='')
                    continue
                if episode_warning is not None:
                    episode_parse_warnings.append(episode_warning)
                    print(line, end='')
                    continue
                if (
                    stripped.startswith("[Hybrid-Metrics] Task contact_metrics")
                    or stripped.startswith("[Hybrid] Task avg squeeze_pred=")
                    or stripped.startswith("Evaluation Results:")
                    or stripped.startswith("Total experiments:")
                    or stripped.startswith("Successful experiments:")
                    or stripped.startswith("Success rate:")
                ):
                    continue
                print(line, end='')

        process.wait(timeout=3600)
        end_time = time.time()

        (
            success_rate,
            successful_experiments,
            total_experiments,
            avg_squeeze_pred,
            avg_squeeze_meas,
            task_squeeze_max_mean,
            task_app_max_mean,
            task_app_mean_mean,
            task_squeeze_max_meas_mean,
            task_ap_max_meas_mean,
            task_ap_mean_meas_mean,
        ) = parse_success_metrics(output_lines)

        episodes, metric_artifacts, episode_force_aggregates = collect_episode_metric_artifacts(
            config,
            episodes,
            episode_parse_warnings,
            step_trace_path,
            max_inference_steps=max_inference_steps,
            successful_experiments=successful_experiments,
        )
        metrics_warnings = metric_artifacts["metrics_warnings"]

        legacy_force_values = {
            "squeeze_avg_pred": avg_squeeze_pred,
            "squeeze_avg_meas": avg_squeeze_meas,
            "squeeze_max_pred": task_squeeze_max_mean,
            "squeeze_max_meas": task_squeeze_max_meas_mean,
            "ap_avg_pred": task_app_mean_mean,
            "ap_avg_meas": task_ap_mean_meas_mean,
            "ap_max_pred": task_app_max_mean,
            "ap_max_meas": task_ap_max_meas_mean,
        }
        for metric_key in FORCE_METRIC_KEYS:
            legacy_value = legacy_force_values[metric_key]
            episode_value = episode_force_aggregates[metric_key]
            if legacy_value is None:
                legacy_force_values[metric_key] = episode_value
            elif episode_value is not None and abs(float(legacy_value) - float(episode_value)) > 5.1e-4:
                metrics_warnings.append(
                    f"task {metric_key}={legacy_value} does not match episode aggregate {episode_value}"
                )

        avg_squeeze_pred = legacy_force_values["squeeze_avg_pred"]
        avg_squeeze_meas = legacy_force_values["squeeze_avg_meas"]
        task_squeeze_max_mean = legacy_force_values["squeeze_max_pred"]
        task_squeeze_max_meas_mean = legacy_force_values["squeeze_max_meas"]
        task_app_mean_mean = legacy_force_values["ap_avg_pred"]
        task_ap_mean_meas_mean = legacy_force_values["ap_avg_meas"]
        task_app_max_mean = legacy_force_values["ap_max_pred"]
        task_ap_max_meas_mean = legacy_force_values["ap_max_meas"]
        metric_artifacts["metrics_status"] = "complete" if not metrics_warnings else "partial"

        # Some subprocesses may print the final evaluation stats successfully but still exit with
        # a non-zero return code (e.g., teardown issues when closing IsaacSim). For reporting,
        # treat the task as "completed" if we could reliably parse success stats for the full
        # requested number of experiments.
        parsed_full_eval = (
            (success_rate is not None)
            and (successful_experiments is not None)
            and (total_experiments is not None)
            and (total_experiments == config.num_total_experiments)
        )
        status = "completed" if parsed_full_eval else "failed"

        print(f"\n{'='*80}")
        print(f"TASK {status.upper()}: {task_suite} - Task {task_id}")
        print(f"{'='*80}")
        if success_rate is not None:
            print(f"✓ Success Rate: {success_rate:.2f}% ({successful_experiments}/{total_experiments} experiments)")
        else:
            print(f"✗ Failed to extract success rate (return code: {process.returncode})")
        print(
            f"Metrics Status: {metric_artifacts['metrics_status']} "
            f"({len(episodes)}/{config.num_total_experiments} episodes)"
        )
        for warning in metrics_warnings:
            print(f"⚠ Metrics: {warning}")
        print(f"Execution Time: {end_time - start_time:.1f}s")
        print(f"{'='*80}\n")

        return {
            "task_suite": task_suite,
            "task_id": task_id,
            "task_name": task_name,
            "language_instruction": language_instruction,
            "success_rate": success_rate,
            "successful_experiments": successful_experiments,
            "total_experiments": total_experiments,
            "max_inference_steps": max_inference_steps,
            "avg_squeeze_pred": avg_squeeze_pred,
            "avg_squeeze_meas": avg_squeeze_meas,
            "task_squeeze_max_mean": task_squeeze_max_mean,
            "task_app_max_mean": task_app_max_mean,
            "task_app_mean_mean": task_app_mean_mean,
            "task_squeeze_max_meas_mean": task_squeeze_max_meas_mean,
            "task_ap_max_meas_mean": task_ap_max_meas_mean,
            "task_ap_mean_meas_mean": task_ap_mean_meas_mean,
            **metric_artifacts,
            "return_code": process.returncode,
            "execution_time": end_time - start_time,
            "status": status,
        }

    except subprocess.TimeoutExpired:
        if process is not None:
            with suppress(Exception):
                process.kill()
            with suppress(Exception):
                process.wait(timeout=30)
        end_time = time.time()
        _, metric_artifacts, force_aggregates = collect_episode_metric_artifacts(
            config,
            episodes,
            episode_parse_warnings,
            step_trace_path,
            max_inference_steps=max_inference_steps,
        )
        print(f"\n{'='*80}")
        print(f"✗ TASK TIMEOUT: {task_suite} - Task {task_id}")
        print(f"{'='*80}\n")
        return {
            "task_suite": task_suite,
            "task_id": task_id,
            "task_name": task_name,
            "language_instruction": language_instruction,
            "success_rate": None,
            "successful_experiments": None,
            "total_experiments": None,
            "max_inference_steps": max_inference_steps,
            **force_aggregates_to_legacy_fields(force_aggregates),
            **metric_artifacts,
            "return_code": -1,
            "execution_time": end_time - start_time,
            "status": "timeout",
        }
    except Exception as e:
        end_time = time.time()
        _, metric_artifacts, force_aggregates = collect_episode_metric_artifacts(
            config,
            episodes,
            episode_parse_warnings,
            step_trace_path,
            max_inference_steps=max_inference_steps,
        )
        print(f"\n{'='*80}")
        print(f"✗ TASK ERROR: {task_suite} - Task {task_id}")
        print(f"Error: {e}")
        print(f"{'='*80}\n")
        return {
            "task_suite": task_suite,
            "task_id": task_id,
            "task_name": task_name,
            "language_instruction": language_instruction,
            "success_rate": None,
            "successful_experiments": None,
            "total_experiments": None,
            "max_inference_steps": max_inference_steps,
            **force_aggregates_to_legacy_fields(force_aggregates),
            **metric_artifacts,
            "return_code": -1,
            "execution_time": end_time - start_time,
            "status": "error",
            "error": str(e),
        }


def save_success_rates_json(results: list[dict], output_file: Path, config: EvaluationConfig):
    """Save success rates to a JSON file."""
    if not results:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if config.prompt_adverbs:
        prompt_mode = "adverb_set"
    elif config.prompt_adverb:
        prompt_mode = "single_adverb"
    else:
        prompt_mode = "plain"

    output_data = {
        "metadata": {
            "policy_model": config.policy_model,
            "control_mode": config.control_mode if config.policy_model == "openpi" else "N/A",
            "abs7d": bool(config.abs7d) if config.policy_model == "openpi" else False,
            "task_environment": config.task if config.task else "auto",
            "num_total_experiments": config.num_total_experiments,
            "num_success_steps": config.num_success_steps,
            "episode_metrics_schema_version": 5,
            "replan_steps": config.replan_steps,
            "lift_height_threshold_m": config.lift_height_threshold_m,
            "lift_hold_steps": config.lift_hold_steps,
            "reward_force_contact_epsilon_n": (
                config.reward_force_contact_epsilon_n
            ),
            "reward_force_min_valid_samples": (
                config.reward_force_min_valid_samples
            ),
            "record_step_traces": config.record_step_traces,
            "step_trace_dir": str(config.step_trace_dir) if config.step_trace_dir is not None else None,
            "record_videos": config.record_videos,
            "record_camera_output_path": (
                str(config.record_camera_output_path)
                if config.record_camera_output_path is not None
                else None
            ),
            "prompt_mode": prompt_mode,
            "prompt_adverb": config.prompt_adverb,
            "prompt_adverbs": list(config.prompt_adverbs),
            "prompt_seed": config.prompt_seed,
            "max_inference_steps_policy": {
                "libero_10": effective_max_inference_steps(config, "libero_10"),
                "libero_goal": effective_max_inference_steps(config, "libero_goal"),
                "libero_spatial": effective_max_inference_steps(config, "libero_spatial"),
                "libero_object": effective_max_inference_steps(config, "libero_object"),
                "default": config.max_inference_steps,
            },
            "max_environment_steps_policy": {
                "libero_10": effective_max_inference_steps(config, "libero_10") * config.replan_steps,
                "libero_goal": effective_max_inference_steps(config, "libero_goal") * config.replan_steps,
                "libero_spatial": effective_max_inference_steps(config, "libero_spatial") * config.replan_steps,
                "libero_object": effective_max_inference_steps(config, "libero_object") * config.replan_steps,
                "default": config.max_inference_steps * config.replan_steps,
            },
            "timestamp": datetime.now().isoformat(),
        },
        "results": {}
    }

    for result in results:
        task_key = f"{result['task_suite']}_task{result['task_id']}"
        output_data["results"][task_key] = {
            "task_suite": result["task_suite"],
            "task_id": result["task_id"],
            "task_name": result["task_name"],
            "language_instruction": result["language_instruction"],
            "success_rate": result["success_rate"],
            "successful_experiments": result["successful_experiments"],
            "total_experiments": result["total_experiments"],
            "max_inference_steps": result.get("max_inference_steps"),
            "execution_time": result["execution_time"],
            "status": result["status"],
            # Hybrid 下可选：成功 experiment 的平均挤压力（可能为 None）
            "avg_squeeze_pred": result.get("avg_squeeze_pred"),
            "avg_squeeze_meas": result.get("avg_squeeze_meas"),
            # Hybrid 下可选：task 级别的挤压力 / 加持力 metrics（可能为 None）
            "task_squeeze_max_mean": result.get("task_squeeze_max_mean"),
            "task_app_max_mean": result.get("task_app_max_mean"),
            "task_app_mean_mean": result.get("task_app_mean_mean"),
            # 新增：实测 Top5% 最大挤压力 / 加持力 & 实测加持力平均值
            "task_squeeze_max_meas_mean": result.get("task_squeeze_max_meas_mean"),
            "task_ap_max_meas_mean": result.get("task_ap_max_meas_mean"),
            "task_ap_mean_meas_mean": result.get("task_ap_mean_meas_mean"),
            "metrics_status": result.get("metrics_status", "partial"),
            "metrics_warnings": result.get("metrics_warnings", ["episode metrics were not provided"]),
            "step_statistics": result.get("step_statistics", summarize_step_statistics([])),
            "force_metric_episode_counts": result.get(
                "force_metric_episode_counts", dict.fromkeys(FORCE_METRIC_KEYS, 0)
            ),
            "success_trajectory_mean_measured_squeeze": result.get(
                "success_trajectory_mean_measured_squeeze"
            ),
            "success_trajectory_force_episode_count": result.get(
                "success_trajectory_force_episode_count", 0
            ),
            "all_eligible_trajectory_mean_measured_squeeze": result.get(
                "all_eligible_trajectory_mean_measured_squeeze"
            ),
            "all_eligible_trajectory_force_episode_count": result.get(
                "all_eligible_trajectory_force_episode_count", 0
            ),
            "successful_trajectory_force_total_episodes": result.get(
                "successful_trajectory_force_total_episodes", 0
            ),
            "successful_trajectory_force_ineligible_episodes": result.get(
                "successful_trajectory_force_ineligible_episodes", 0
            ),
            "trajectory_force_valid_sample_statistics": result.get(
                "trajectory_force_valid_sample_statistics", {}
            ),
            "damage_threshold_statistics": result.get(
                "damage_threshold_statistics", {}
            ),
            "friction_statistics": result.get("friction_statistics", {}),
            "episodes": result.get("episodes", []),
            "step_trace": result.get(
                "step_trace", {"enabled": False, "path": None, "rows": 0, "status": "disabled"}
            ),
        }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Success rates saved to: {output_file}")


def _txt_value(value, *, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _write_episode_metrics_txt(output, result: dict) -> None:
    """Write additive metrics sections while preserving the legacy task summary."""
    metrics_status = result.get("metrics_status", "partial")
    metrics_warnings = result.get("metrics_warnings", ["episode metrics were not provided"])
    output.write(f"    Metrics status: {metrics_status}\n")
    if metrics_warnings:
        output.write("    Metrics warnings:\n")
        for warning in metrics_warnings:
            output.write(f"      - {warning}\n")

    output.write("    Reward-aligned trajectory force:\n")
    output.write(
        "      success_trajectory_mean_measured_squeeze="
        f"{_txt_value(result.get('success_trajectory_mean_measured_squeeze'))} "
        "eligible_successes="
        f"{_txt_value(result.get('success_trajectory_force_episode_count', 0))} "
        "successful_total="
        f"{_txt_value(result.get('successful_trajectory_force_total_episodes', 0))} "
        "successful_ineligible="
        f"{_txt_value(result.get('successful_trajectory_force_ineligible_episodes', 0))}\n"
    )
    output.write(
        "      all_eligible_trajectory_mean_measured_squeeze="
        f"{_txt_value(result.get('all_eligible_trajectory_mean_measured_squeeze'))} "
        "eligible_episodes="
        f"{_txt_value(result.get('all_eligible_trajectory_force_episode_count', 0))}\n"
    )
    output.write(
        "      group | count | valid_samples_mean | valid_samples_median | "
        "valid_samples_min | valid_samples_max\n"
    )
    trajectory_sample_stats = result.get(
        "trajectory_force_valid_sample_statistics", {}
    )
    for group in ("all", "successful"):
        stats = trajectory_sample_stats.get(group, {})
        output.write(
            "      "
            + " | ".join(
                [
                    group,
                    _txt_value(stats.get("count")),
                    _txt_value(stats.get("mean")),
                    _txt_value(stats.get("median")),
                    _txt_value(stats.get("min")),
                    _txt_value(stats.get("max")),
                ]
            )
            + "\n"
        )

    output.write("    Step statistics:\n")
    output.write(
        "      group | episodes | env_total | env_mean | env_min | env_max | "
        "chunks_total | chunks_mean | chunks_min | chunks_max\n"
    )
    step_statistics = result.get("step_statistics", summarize_step_statistics([]))
    for group in ("all", "successful", "failed"):
        summary = step_statistics.get(group, {})
        output.write(
            "      "
            + " | ".join(
                [
                    group,
                    _txt_value(summary.get("episodes")),
                    _txt_value(summary.get("env_steps_total")),
                    _txt_value(summary.get("env_steps_mean"), precision=2),
                    _txt_value(summary.get("env_steps_min")),
                    _txt_value(summary.get("env_steps_max")),
                    _txt_value(summary.get("inference_chunks_total")),
                    _txt_value(summary.get("inference_chunks_mean"), precision=2),
                    _txt_value(summary.get("inference_chunks_min")),
                    _txt_value(summary.get("inference_chunks_max")),
                ]
            )
            + "\n"
        )

    trace = result.get(
        "step_trace", {"enabled": False, "path": None, "rows": 0, "status": "disabled"}
    )
    output.write(
        "    Step trace: "
        f"enabled={trace.get('enabled', False)} "
        f"path={_txt_value(trace.get('path'))} "
        f"rows={_txt_value(trace.get('rows'))} "
        f"status={trace.get('status', 'partial')}\n"
    )

    friction_statistics = result.get("friction_statistics", {})
    output.write("    Friction statistics:\n")
    if not friction_statistics:
        output.write("      N/A\n")
    else:
        output.write(
            "      scope | static_count | static_mean | static_min | static_max | "
            "dynamic_count | dynamic_mean | dynamic_min | dynamic_max\n"
        )
        for scope_name, scope in sorted(friction_statistics.items()):
            static = scope.get("static_friction", {})
            dynamic = scope.get("dynamic_friction", {})
            output.write(
                "      "
                + " | ".join(
                    [
                        str(scope_name),
                        _txt_value(static.get("count")),
                        _txt_value(static.get("mean")),
                        _txt_value(static.get("min")),
                        _txt_value(static.get("max")),
                        _txt_value(dynamic.get("count")),
                        _txt_value(dynamic.get("mean")),
                        _txt_value(dynamic.get("min")),
                        _txt_value(dynamic.get("max")),
                    ]
                )
                + "\n"
            )

    damage_threshold_statistics = result.get("damage_threshold_statistics", {})
    output.write("    Damage threshold statistics:\n")
    if not damage_threshold_statistics:
        output.write("      N/A\n")
    else:
        output.write(
            "      object | field | count | mean | min | max\n"
        )
        for object_name, fields in sorted(damage_threshold_statistics.items()):
            for field_name in (
                "mass_kg",
                "effective_static_friction",
                "max_squeeze_force",
            ):
                summary = fields.get(field_name, {})
                output.write(
                    "      "
                    + " | ".join(
                        [
                            str(object_name),
                            field_name,
                            _txt_value(summary.get("count")),
                            _txt_value(summary.get("mean")),
                            _txt_value(summary.get("min")),
                            _txt_value(summary.get("max")),
                        ]
                    )
                    + "\n"
                )

    episodes = result.get("episodes", [])
    output.write("    Per-experiment step and force coverage:\n")
    output.write(
        "      exp | hdf5_episode | success | end_reason | terminal_term | env_steps | chunks | force_status | "
        "pred_samples | meas_samples | pred_contact | meas_contact | coverage\n"
    )
    if not episodes:
        output.write("      N/A\n")
    for episode in episodes:
        samples = episode.get("force_samples", {})
        pred_contact = (
            f"{_txt_value(samples.get('predicted_contact_steps'))}/"
            f"{_txt_value(samples.get('predicted_contact_ratio'), precision=3)}"
        )
        meas_contact = (
            f"{_txt_value(samples.get('measured_contact_steps'))}/"
            f"{_txt_value(samples.get('measured_contact_ratio'), precision=3)}"
        )
        output.write(
            "      "
            + " | ".join(
                [
                    _txt_value(episode.get("experiment_index")),
                    _txt_value(episode.get("hdf5_episode_index")),
                    _txt_value(episode.get("success")),
                    _txt_value(episode.get("end_reason")),
                    _txt_value(episode.get("terminal_term")),
                    _txt_value(episode.get("env_steps")),
                    _txt_value(episode.get("inference_chunks")),
                    _txt_value(episode.get("force_status")),
                    _txt_value(samples.get("predicted_action_steps")),
                    _txt_value(samples.get("measured_force_steps")),
                    pred_contact,
                    meas_contact,
                    _txt_value(samples.get("coverage_ratio"), precision=3),
                ]
            )
            + "\n"
        )

    output.write("    Per-experiment damage threshold:\n")
    output.write(
        "      exp | object_name | mode | mass_kg | gravity_m_s2 | gripper_static | "
        "object_static | effective_static | tolerance_factor | max_squeeze_force | consecutive_frames\n"
    )
    if not episodes:
        output.write("      N/A\n")
    for episode in episodes:
        threshold = episode.get("damage_threshold") or {}
        output.write(
            "      "
            + " | ".join(
                [
                    _txt_value(episode.get("experiment_index")),
                    _txt_value(threshold.get("object_name")),
                    _txt_value(threshold.get("mode")),
                    _txt_value(threshold.get("mass_kg")),
                    _txt_value(threshold.get("gravity_m_s2")),
                    _txt_value(threshold.get("gripper_static_friction")),
                    _txt_value(threshold.get("object_static_friction")),
                    _txt_value(threshold.get("effective_static_friction")),
                    _txt_value(threshold.get("tolerance_factor")),
                    _txt_value(threshold.get("max_squeeze_force")),
                    _txt_value(threshold.get("consecutive_frames")),
                ]
            )
            + "\n"
        )

    output.write("    Per-experiment friction:\n")
    output.write(
        "      exp | gripper_static | gripper_dynamic | object_name | object_static | object_dynamic\n"
    )
    if not episodes:
        output.write("      N/A\n")
    for episode in episodes:
        friction = episode.get("friction") or {}
        gripper = friction.get("gripper") or {}
        objects = friction.get("objects") or {}
        if objects:
            object_name = sorted(objects)[0]
            object_pair = objects[object_name]
        else:
            object_name = None
            object_pair = {}
        output.write(
            "      "
            + " | ".join(
                [
                    _txt_value(episode.get("experiment_index")),
                    _txt_value(gripper.get("static_friction")),
                    _txt_value(gripper.get("dynamic_friction")),
                    _txt_value(object_name),
                    _txt_value(object_pair.get("static_friction")),
                    _txt_value(object_pair.get("dynamic_friction")),
                ]
            )
            + "\n"
        )

    output.write("    Per-experiment force metrics:\n")
    output.write(
        "      exp | squeeze_avg_pred | squeeze_avg_meas | squeeze_max_pred | squeeze_max_meas | "
        "ap_avg_pred | ap_avg_meas | ap_max_pred | ap_max_meas\n"
    )
    if not episodes:
        output.write("      N/A\n")
    for episode in episodes:
        output.write(
            "      "
            + " | ".join(
                [_txt_value(episode.get("experiment_index"))]
                + [_txt_value(episode.get(key)) for key in FORCE_METRIC_KEYS]
            )
            + "\n"
        )

    output.write("    Per-experiment force-target tracking:\n")
    output.write(
        "      exp | status | valid | model_raw_mean | effective_target_mean | "
        "measured_raw_mean | effective_error_mean | lower_sat_valid | lower_sat_valid_ratio\n"
    )
    if not episodes:
        output.write("      N/A\n")
    for episode in episodes:
        tracking = episode.get("force_tracking") or {}
        samples = tracking.get("sample_counts") or {}
        reward_valid = tracking.get("reward_valid") or {}
        command = tracking.get("gripper_position_command") or {}
        output.write(
            "      "
            + " | ".join(
                [
                    _txt_value(episode.get("experiment_index")),
                    _txt_value(tracking.get("status")),
                    _txt_value(samples.get("reward_valid_available_steps")),
                    _txt_value(reward_valid.get("mean_predicted_squeeze_n")),
                    _txt_value(
                        reward_valid.get("mean_effective_squeeze_target_n")
                    ),
                    _txt_value(
                        reward_valid.get("mean_measured_squeeze_raw_n")
                    ),
                    _txt_value(
                        reward_valid.get("mean_effective_target_error_n")
                    ),
                    _txt_value(
                        command.get("reward_valid_lower_saturation_steps")
                    ),
                    _txt_value(
                        command.get("reward_valid_lower_saturation_ratio")
                    ),
                ]
            )
            + "\n"
        )

    output.write("    Per-experiment reward-aligned trajectory force:\n")
    output.write(
        "      exp | status | grasp_started | grasp_start_step | valid_samples | "
        "trajectory_mean_measured_squeeze | error\n"
    )
    if not episodes:
        output.write("      N/A\n")
    for episode in episodes:
        output.write(
            "      "
            + " | ".join(
                [
                    _txt_value(episode.get("experiment_index")),
                    _txt_value(episode.get("trajectory_force_status")),
                    _txt_value(episode.get("grasp_started")),
                    _txt_value(episode.get("grasp_start_step")),
                    _txt_value(episode.get("trajectory_force_valid_samples")),
                    _txt_value(
                        episode.get("trajectory_mean_measured_squeeze")
                    ),
                    _txt_value(episode.get("trajectory_force_error")),
                ]
            )
            + "\n"
        )


def save_success_rates_txt(results: list[dict], output_file: Path, config: EvaluationConfig):  # noqa: C901
    """Save success rates to a TXT file."""
    if not results:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("POLICY EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Policy Model: {config.policy_model}\n")
        if config.policy_model == "openpi":
            f.write(f"  Control Mode: {config.control_mode}\n")
            f.write(f"  abs7d: {config.abs7d}\n")
        f.write(f"  Experiments per task: {config.num_total_experiments}\n")
        f.write(f"  Success steps required: {config.num_success_steps}\n")
        f.write(
            "  Reward-force contact epsilon (N): "
            f"{config.reward_force_contact_epsilon_n}\n"
        )
        f.write(
            "  Reward-force minimum valid samples: "
            f"{config.reward_force_min_valid_samples}\n"
        )
        f.write(f"  Default max inference steps: {config.max_inference_steps}\n")
        f.write(f"  Replan steps: {config.replan_steps}\n")
        f.write(f"  Record step traces: {config.record_step_traces}\n")
        f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "=" * 60 + "\n\n")

        suites = {}
        for result in results:
            suite = result["task_suite"]
            if suite not in suites:
                suites[suite] = []
            suites[suite].append(result)

        suite_averages = {}
        for suite, suite_results in suites.items():
            valid_results = [r for r in suite_results if r["status"] == "completed" and r["success_rate"] is not None]
            suite_averages[suite] = sum(r["success_rate"] for r in valid_results) / len(valid_results) if valid_results else None

        for suite, suite_results in sorted(suites.items()):
            f.write(f"{suite.upper()}:\n")
            f.write("-" * 40 + "\n")
            suite_results.sort(key=lambda x: x["task_id"])

            for result in suite_results:
                task_id = result["task_id"]
                task_name = result["task_name"]
                success_rate = result["success_rate"]
                successful = result["successful_experiments"]
                total = result["total_experiments"]
                status = result["status"]
                exec_time = result["execution_time"]
                avg_squeeze_pred = result.get("avg_squeeze_pred")
                avg_squeeze_meas = result.get("avg_squeeze_meas")
                task_squeeze_max_mean = result.get("task_squeeze_max_mean")
                task_app_max_mean = result.get("task_app_max_mean")
                task_app_mean_mean = result.get("task_app_mean_mean")
                task_squeeze_max_meas_mean = result.get("task_squeeze_max_meas_mean")
                task_ap_max_meas_mean = result.get("task_ap_max_meas_mean")
                task_ap_mean_meas_mean = result.get("task_ap_mean_meas_mean")

                f.write(
                    f"  Task {task_id} effective limits: "
                    f"max inference chunks={result.get('max_inference_steps', 'N/A')}; "
                    f"max environment steps="
                    f"{result.get('max_inference_steps', 0) * config.replan_steps if result.get('max_inference_steps') is not None else 'N/A'}\n"
                )
                if success_rate is not None:
                    f.write(f"  Task {task_id} ({task_name}): {success_rate:.2f}% ({successful}/{total}) [{exec_time:.1f}s]\n")
                    # 9 个 Hybrid metric（与终端表头保持一致的命名）
                    # - squeeze_avg_pred / squeeze_avg_meas
                    # - squeeze_max_pred / squeeze_max_meas
                    # - ap_avg_pred / ap_avg_meas
                    # - ap_max_pred / ap_max_meas
                    has_any_hf_metric = any(
                        v is not None
                        for v in [
                            avg_squeeze_pred,
                            avg_squeeze_meas,
                            task_squeeze_max_mean,
                            task_squeeze_max_meas_mean,
                            task_app_mean_mean,
                            task_ap_mean_meas_mean,
                            task_app_max_mean,
                            task_ap_max_meas_mean,
                        ]
                    )
                    if has_any_hf_metric:
                        f.write("    Hybrid metrics (success only, task-level mean over experiments):\n")
                        if avg_squeeze_pred is not None or avg_squeeze_meas is not None:
                            f.write("      ")
                            if avg_squeeze_pred is not None:
                                f.write(f"squeeze_avg_pred={avg_squeeze_pred:.4f} ")
                            else:
                                f.write("squeeze_avg_pred=N/A ")
                            if avg_squeeze_meas is not None:
                                f.write(f"squeeze_avg_meas={avg_squeeze_meas:.4f}")
                            else:
                                f.write("squeeze_avg_meas=N/A")
                            f.write("\n")

                        if (
                            task_squeeze_max_mean is not None
                            or task_squeeze_max_meas_mean is not None
                        ):
                            f.write("      ")
                            if task_squeeze_max_mean is not None:
                                f.write(f"squeeze_max_pred={task_squeeze_max_mean:.4f} ")
                            else:
                                f.write("squeeze_max_pred=N/A ")
                            if task_squeeze_max_meas_mean is not None:
                                f.write(f"squeeze_max_meas={task_squeeze_max_meas_mean:.4f}")
                            else:
                                f.write("squeeze_max_meas=N/A")
                            f.write("\n")

                        if task_app_mean_mean is not None or task_ap_mean_meas_mean is not None:
                            f.write("      ")
                            if task_app_mean_mean is not None:
                                f.write(f"ap_avg_pred={task_app_mean_mean:.4f} ")
                            else:
                                f.write("ap_avg_pred=N/A ")
                            if task_ap_mean_meas_mean is not None:
                                f.write(f"ap_avg_meas={task_ap_mean_meas_mean:.4f}")
                            else:
                                f.write("ap_avg_meas=N/A")
                            f.write("\n")

                        if task_app_max_mean is not None or task_ap_max_meas_mean is not None:
                            f.write("      ")
                            if task_app_max_mean is not None:
                                f.write(f"ap_max_pred={task_app_max_mean:.4f} ")
                            else:
                                f.write("ap_max_pred=N/A ")
                            if task_ap_max_meas_mean is not None:
                                f.write(f"ap_max_meas={task_ap_max_meas_mean:.4f}")
                            else:
                                f.write("ap_max_meas=N/A")
                            f.write("\n")
                else:
                    f.write(f"  Task {task_id} ({task_name}): N/A ({status})\n")

                _write_episode_metrics_txt(f, result)

            avg_sr = suite_averages[suite]
            if avg_sr is not None:
                completed_count = len([r for r in suite_results if r["status"] == "completed"])
                f.write(f"\n  Suite Average: {avg_sr:.2f}% ({completed_count}/{len(suite_results)} tasks completed)\n")
            else:
                f.write("\n  Suite Average: N/A (no completed tasks)\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 60 + "\n")

        for suite, avg_sr in sorted(suite_averages.items()):
            f.write(f"  {suite}: {avg_sr:.2f}%\n" if avg_sr is not None else f"  {suite}: N/A\n")

        valid_averages = [sr for sr in suite_averages.values() if sr is not None]
        if valid_averages:
            overall_avg = sum(valid_averages) / len(valid_averages)
            f.write(f"\n  Overall Average: {overall_avg:.2f}% ({len(valid_averages)}/{len(suite_averages)} suites)\n")
        else:
            f.write("\n  Overall Average: N/A\n")

        f.write("=" * 60 + "\n")

        # 在 TXT 文件最后附上「所有任务」的大表（与终端 print_metrics_ascii_table 一致）
        # 构造行数据
        rows: list[list[str]] = []
        for result in results:
            if result.get("status") != "completed":
                continue

            task_suite = result.get("task_suite", "")
            task_id = result.get("task_id", -1)
            success_rate_val = result.get("success_rate")  # 百分比
            avg_squeeze_pred = result.get("avg_squeeze_pred")
            avg_squeeze_meas = result.get("avg_squeeze_meas")
            task_squeeze_max_mean = result.get("task_squeeze_max_mean")
            task_squeeze_max_meas_mean = result.get("task_squeeze_max_meas_mean")
            # NOTE: result dict uses "task_app_mean_mean" (double 'p') as the canonical key.
            # Older code mistakenly looked up "task_ap_mean_mean", which caused ap_avg_pred to be N/A
            # in the final TXT metrics table (even though per-task Hybrid metrics were present).
            task_ap_mean_mean = result.get("task_app_mean_mean")
            task_ap_mean_meas_mean = result.get("task_ap_mean_meas_mean")
            task_app_max_mean = result.get("task_app_max_mean")
            task_ap_max_meas_mean = result.get("task_ap_max_meas_mean")

            if isinstance(task_suite, str) and task_suite.startswith("libero_"):
                suite_short = task_suite.split("libero_")[1]
            else:
                suite_short = task_suite
            task_label = f"{suite_short} {task_id}"

            def _fmt(value: Optional[float], digits: int) -> str:
                if value is None:
                    return "N/A"
                return f"{value:.{digits}f}"

            # success_rate 以 0.x 形式展示
            if success_rate_val is not None:
                sr_display = success_rate_val / 100.0
            else:
                sr_display = None

            rows.append(
                [
                    task_label,
                    _fmt(sr_display, 2),
                    _fmt(avg_squeeze_pred, 4),
                    _fmt(avg_squeeze_meas, 4),
                    _fmt(task_squeeze_max_mean, 4),
                    _fmt(task_squeeze_max_meas_mean, 4),
                    _fmt(task_ap_mean_mean, 4),
                    _fmt(task_ap_mean_meas_mean, 4),
                    _fmt(task_app_max_mean, 4),
                    _fmt(task_ap_max_meas_mean, 4),
                ]
            )

        if rows:
            headers = [
                "task id",
                "success_rate",
                "squeeze_avg_pred",
                "squeeze_avg_meas",
                "squeeze_max_pred",
                "squeeze_max_meas",
                "ap_avg_pred",
                "ap_avg_meas",
                "ap_max_pred",
                "ap_max_meas",
            ]

            cols = list(zip(headers, *rows))
            col_widths = [max(len(str(cell)) for cell in col) + 2 for col in cols]

            def _hline() -> str:
                return "+" + "+".join("-" * w for w in col_widths) + "+"

            def _format_row(cells: list[str]) -> str:
                padded: list[str] = []
                for cell, width in zip(cells, col_widths):
                    cell_str = str(cell)
                    space = width - len(cell_str)
                    left = space // 2
                    right = space - left
                    padded.append(" " * left + cell_str + " " * right)
                return "|" + "|".join(padded) + "|"

            f.write("\n=== metric (all tasks) ===\n\n")
            f.write(_hline() + "\n")
            f.write(_format_row(headers) + "\n")
            f.write(_hline() + "\n")
            for row in rows:
                f.write(_format_row(row) + "\n")
            f.write(_hline() + "\n")

    print(f"Success rates saved to: {output_file}")


def print_summary(results: list[dict]):
    """Print a summary to console."""
    if not results:
        return

    suites = {}
    for result in results:
        suite = result["task_suite"]
        if suite not in suites:
            suites[suite] = []
        suites[suite].append(result)

    completed_results = [r for r in results if r["status"] == "completed" and r["success_rate"] is not None]

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {len(results)}")
    print(f"Completed tasks: {len(completed_results)}")
    print(f"Failed tasks: {len([r for r in results if r['status'] == 'failed'])}")
    print(f"Timeout tasks: {len([r for r in results if r['status'] == 'timeout'])}")

    if completed_results:
        success_rates = [r["success_rate"] for r in completed_results]
        avg_success_rate = sum(success_rates) / len(success_rates)
        min_success_rate = min(success_rates)
        max_success_rate = max(success_rates)
        print(f"\nOverall average success rate: {avg_success_rate:.2f}%")
        print(f"Success rate range: {min_success_rate:.2f}% - {max_success_rate:.2f}%")

    print(f"\n{'='*60}")
    print("SUITE AVERAGES")
    print(f"{'='*60}")

    for suite, suite_results in sorted(suites.items()):
        valid_results = [r for r in suite_results if r["status"] == "completed" and r["success_rate"] is not None]
        if valid_results:
            suite_avg = sum(r["success_rate"] for r in valid_results) / len(valid_results)
            print(f"{suite}: {suite_avg:.2f}% ({len(valid_results)}/{len(suite_results)} tasks)")
        else:
            print(f"{suite}: N/A ({len(suite_results)} tasks, none completed)")

    print(f"{'='*60}\n")


def print_metrics_ascii_table(results: list[dict]):
    """打印符合用户模板的整体 metrics 表."""
    if not results:
        return

    # 收集已完成任务的核心指标
    rows: list[list[str]] = []
    for result in results:
        if result.get("status") != "completed":
            continue

        task_suite = result.get("task_suite", "")
        task_id = result.get("task_id", -1)
        success_rate = result.get("success_rate")  # 单位：百分比
        avg_squeeze_pred = result.get("avg_squeeze_pred")
        avg_squeeze_meas = result.get("avg_squeeze_meas")
        task_squeeze_max_mean = result.get("task_squeeze_max_mean")
        task_squeeze_max_meas_mean = result.get("task_squeeze_max_meas_mean")
        # 注意：结果字典中统一使用 "task_app_mean_mean" / "task_app_max_mean" 作为 key
        # 这里读取后赋值给 task_ap_* 变量，仅用于表格展示字段名（ap_*）
        task_ap_mean_mean = result.get("task_app_mean_mean")
        task_ap_mean_meas_mean = result.get("task_ap_mean_meas_mean")
        task_ap_max_mean = result.get("task_app_max_mean")
        task_ap_max_meas_mean = result.get("task_ap_max_meas_mean")
        reward_aligned_squeeze = result.get(
            "success_trajectory_mean_measured_squeeze"
        )

        # 任务标签：如 libero_goal -> goal 0
        if isinstance(task_suite, str) and task_suite.startswith("libero_"):
            suite_short = task_suite.split("libero_")[1]
        else:
            suite_short = task_suite
        task_label = f"{suite_short} {task_id}"

        def _fmt(value: Optional[float], digits: int) -> str:
            if value is None:
                return "N/A"
            return f"{value:.{digits}f}"

        # 成功率按 0.x 形式展示：success_rate 本身是百分比
        if success_rate is not None:
            sr_display = success_rate / 100.0
        else:
            sr_display = None

        rows.append(
            [
                task_label,
                _fmt(sr_display, 2),
                _fmt(avg_squeeze_pred, 4),
                _fmt(avg_squeeze_meas, 4),
                _fmt(task_squeeze_max_mean, 4),
                _fmt(task_squeeze_max_meas_mean, 4),
                _fmt(task_ap_mean_mean, 4),
                _fmt(task_ap_mean_meas_mean, 4),
                _fmt(task_ap_max_mean, 4),
                _fmt(task_ap_max_meas_mean, 4),
                _fmt(reward_aligned_squeeze, 4),
            ]
        )

    if not rows:
        return

    headers = [
        "task id",
        "success_rate",
        "squeeze_avg_pred",
        "squeeze_avg_meas",
        "squeeze_max_pred",
        "squeeze_max_meas",
        "ap_avg_pred",
        "ap_avg_meas",
        "ap_max_pred",
        "ap_max_meas",
        "trajectory_mean_measured_squeeze",
    ]

    # 计算每一列宽度
    cols = list(zip(headers, *rows))
    col_widths = [max(len(str(cell)) for cell in col) + 2 for col in cols]  # 两侧各空 1 格

    def _hline() -> str:
        return "+" + "+".join("-" * w for w in col_widths) + "+"

    def _format_row(cells: list[str]) -> str:
        padded = []
        for cell, width in zip(cells, col_widths):
            cell_str = str(cell)
            space = width - len(cell_str)
            left = space // 2
            right = space - left
            padded.append(" " * left + cell_str + " " * right)
        return "|" + "|".join(padded) + "|"

    print("\n=== metric ===\n")
    print(_hline())
    print(_format_row(headers))
    print(_hline())
    for row in rows:
        print(_format_row(row))
    print(_hline())


def _fmt_float(value: Optional[float], precision: int = 3) -> str:
    """Format float value for pretty table output."""
    if value is None:
        return "   N/A   "
    return f"{value:.{precision}f}".rjust(9)


def print_live_force_table_header():
    """Print table header for per-task Hybrid metrics."""
    header = [
        "Task",
        "Succ(%)",
        "SqPred",
        "SqMeas",
        "SqMaxPred",
        "SqMaxMeas",
        "ApPred",
        "ApMeas",
        "ApMaxPred",
        "ApMaxMeas",
        "TrajSqMeas",
    ]
    line = " | ".join(h.center(12) for h in header)
    print("\n" + "=" * len(line))
    print(line)
    print("-" * len(line))


def print_live_force_table_row(result: dict):
    """Print one row of Hybrid metrics for a single task.

    列约定（从左到右）：
    - Task: 任务名（形如 "goal task 0"）
    - Succ(%): 成功率
    - SqPred / SqMeas: 预测 / 实测 挤压力平均值（来自 [Hybrid] Task avg ...）
    - SqMaxPred: 预测最大挤压力（单条 demo 内 Top5% 帧均值，再在 task 级取平均）
    - SqMaxMeas: 实测最大挤压力（同样的 Top5% 规则）
    - ApPred: 预测加持力平均值（task 级别）
    - ApMeas: 实测加持力平均值（task 级别）
    - ApMaxPred: 预测最大加持力（单条 demo 内 Top5% 帧均值，再在 task 级取平均）
    - ApMaxMeas: 实测最大加持力（同样的 Top5% 规则）
    """
    task_suite = result.get("task_suite", "")
    task_id = result.get("task_id", -1)

    # 任务标签：优先使用 Libero 风格的简写（例如 "goal task 0"）
    if isinstance(task_suite, str) and task_suite.startswith("libero_"):
        suite_short = task_suite.split("libero_")[1]
    else:
        suite_short = task_suite
    task_label = f"{suite_short} task {task_id}"

    success_rate = result.get("success_rate")
    avg_squeeze_pred = result.get("avg_squeeze_pred")
    avg_squeeze_meas = result.get("avg_squeeze_meas")
    task_squeeze_max_mean = result.get("task_squeeze_max_mean")
    task_squeeze_max_meas_mean = result.get("task_squeeze_max_meas_mean")
    task_ap_mean_mean = result.get("task_app_mean_mean")
    task_ap_mean_meas_mean = result.get("task_ap_mean_meas_mean")
    task_ap_max_mean = result.get("task_app_max_mean")
    task_ap_max_meas_mean = result.get("task_ap_max_meas_mean")
    reward_aligned_squeeze = result.get(
        "success_trajectory_mean_measured_squeeze"
    )

    cols = [
        task_label.ljust(12),
        _fmt_float(success_rate, precision=2),
        _fmt_float(avg_squeeze_pred, precision=4),
        _fmt_float(avg_squeeze_meas, precision=4),
        _fmt_float(task_squeeze_max_mean, precision=4),
        _fmt_float(task_squeeze_max_meas_mean, precision=4),
        _fmt_float(task_ap_mean_mean, precision=4),
        _fmt_float(task_ap_mean_meas_mean, precision=4),
        _fmt_float(task_ap_max_mean, precision=4),
        _fmt_float(task_ap_max_meas_mean, precision=4),
        _fmt_float(reward_aligned_squeeze, precision=4),
    ]

    print(" | ".join(cols))


def resolve_step_trace_root(
    config: EvaluationConfig,
    *,
    model_name: str,
    timestamp: str,
) -> Optional[Path]:
    """Resolve and validate the optional force-only JSONL output root."""
    if config.step_trace_dir is not None and not config.record_step_traces:
        raise ValueError("--step-trace-dir requires --record-step-traces")
    if not config.record_step_traces:
        return None
    if config.policy_model != "openpi":
        raise ValueError("--record-step-traces is supported only for the OpenPI client")
    if config.step_trace_dir is not None:
        return config.step_trace_dir
    return config.output_dir / f"step_traces_{model_name}_{timestamp}"


def main():
    """Main entry point."""
    config = tyro.cli(EvaluationConfig)
    if (
        not math.isfinite(config.lift_height_threshold_m)
        or config.lift_height_threshold_m <= 0.0
    ):
        raise ValueError("--lift-height-threshold-m must be finite and positive")
    if isinstance(config.lift_hold_steps, bool) or config.lift_hold_steps <= 0:
        raise ValueError("--lift-hold-steps must be a positive integer")
    if (
        not math.isfinite(config.reward_force_contact_epsilon_n)
        or config.reward_force_contact_epsilon_n < 0.0
    ):
        raise ValueError(
            "--reward-force-contact-epsilon-n must be finite and non-negative"
        )
    if (
        isinstance(config.reward_force_min_valid_samples, bool)
        or config.reward_force_min_valid_samples < 1
    ):
        raise ValueError(
            "--reward-force-min-valid-samples must be a positive integer"
        )

    workspace_root = Path(__file__).parent.parent.parent.resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 如果用户没有显式传 --hdf5_folder，则尝试从 set_replay_env.sh 导出的 HDF5_TRAJ_SOURCE_DIR 读取
    # （该路径指向 libero assembled_hdf5，用于场景 setup / initial_state reset）。
    if config.hdf5_folder is None:
        env_hdf5 = os.environ.get("HDF5_TRAJ_SOURCE_DIR", "").strip()
        if env_hdf5:
            config.hdf5_folder = Path(env_hdf5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.policy_model == "openpi":
        model_name = f"{config.policy_model}_{config.control_mode}"
        if config.abs7d:
            model_name += "_abs7d"
    else:
        model_name = config.policy_model

    try:
        step_trace_root = resolve_step_trace_root(
            config,
            model_name=model_name,
            timestamp=timestamp,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    if step_trace_root is not None:
        config.step_trace_dir = step_trace_root
        print(f"Step traces: {step_trace_root}")

    all_task_suites = get_task_suites_and_tasks()

    # 1) 先按 task_suites 过滤 suite
    if config.task_suites:
        task_suites = {k: v for k, v in all_task_suites.items() if k in config.task_suites}
        if not task_suites:
            print(f"Error: No valid task suites found in {config.task_suites}")
            print(f"Available task suites: {list(all_task_suites.keys())}")
            return
    else:
        task_suites = all_task_suites
    # 2) Tabero 子集（默认：只要数据路径是 tabero/tabero_force，就启用；也可用 --use_tabero_tasks 强制启用）
    subset_map = _load_tabero_task_subset(workspace_root)
    auto_tabero = False
    with suppress(Exception):
        auto_tabero = _should_auto_use_tabero_tasks(
            config.hdf5_folder,
            os.environ.get("REPLAYED_DEMOS_DIR", ""),
        )

    if config.use_tabero_tasks or auto_tabero:
        if not subset_map:
            print("⚠️  [run_task_evaluations] Tabero subset enabled but tabero_tasks.json not found or invalid.")
        for suite, tasks in list(task_suites.items()):
            if suite in subset_map:
                task_suites[suite] = [tid for tid in subset_map[suite] if tid in tasks]
            else:
                task_suites[suite] = tasks

    # 3) 可选：再与 CLI 传入的 task_ids 取交集
    if config.task_ids:
        for suite in task_suites:
            task_suites[suite] = [tid for tid in task_suites[suite] if tid in config.task_ids]

    total_task_count = sum(len(tasks) for tasks in task_suites.values())
    print(f"\n{'='*60}")
    print(f"Starting evaluation with {config.policy_model.upper()}")
    if config.policy_model == "openpi":
        print(f"Control mode: {config.control_mode}")
        if config.abs7d:
            print("abs7d: True")
    print(f"{'='*60}")
    print(f"Will evaluate {total_task_count} tasks across {len(task_suites)} task suites")
    print(f"Task suites: {list(task_suites.keys())}")
    print(f"Experiments per task: {config.num_total_experiments}")
    print(f"{'='*60}\n")

    # 4) 可选：按 assembled HDF5 实际存在的任务文件过滤（避免无 HDF5 时退化为默认 reset）
    if config.require_hdf5:
        if config.hdf5_folder is None:
            print("Error: --require_hdf5 requires --hdf5_folder to be set.")
            return
        hdf5_root = Path(config.hdf5_folder)
        if not hdf5_root.exists():
            print(f"Error: hdf5_folder does not exist: {hdf5_root}")
            return
        for suite in list(task_suites.keys()):
            keep: list[int] = []
            for tid in task_suites[suite]:
                pattern = f"{suite}_task{tid}_*_demo.hdf5"
                if list(hdf5_root.glob(pattern)):
                    keep.append(tid)
            task_suites[suite] = keep
        total_task_count = sum(len(tasks) for tasks in task_suites.values())
        print(f"[require_hdf5] After filtering: {total_task_count} tasks\n")

    results = []
    completed_tasks = 0

    for task_suite, task_ids in sorted(task_suites.items()):
        for task_id in task_ids:
            step_trace_path = (
                step_trace_root / f"{task_suite}_task{task_id}.jsonl"
                if step_trace_root is not None
                else None
            )
            result = run_single_evaluation(
                config,
                task_suite,
                task_id,
                workspace_root,
                step_trace_path=step_trace_path,
            )
            results.append(result)
            completed_tasks += 1

            # 每个 task 完成后打印一次只包含当前 task 的小表
            print_metrics_ascii_table([result])

            print(f"\nProgress: {completed_tasks}/{total_task_count} tasks completed")

    if config.output_format in ["json", "both"]:
        json_file = config.output_dir / f"success_rates_{model_name}_{timestamp}.json"
        save_success_rates_json(results, json_file, config)

    if config.output_format in ["txt", "both"]:
        txt_file = config.output_dir / f"success_rates_{model_name}_{timestamp}.txt"
        save_success_rates_txt(results, txt_file, config)

    print_summary(results)

    # 所有 task 完成后，再额外打印一次「包含全部任务」的 metrics 总览表
    print_metrics_ascii_table(results)


if __name__ == "__main__":
    main()
