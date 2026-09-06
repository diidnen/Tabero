#!/usr/bin/env python3
"""Collect a small, native-Tabero continuous-force feasibility pilot.

This is a data collector, not a controller and not a trainer.  Every force
branch restores the exact in-memory snapshot made from one HDF5 demo and then
executes the same absolute EEF continuation.  The only systematic branch
change is ``set_target_squeeze_force_n(float)`` on native ForcePositionAction.

The collector deliberately keeps baseline validation separate from force
branches, distinguishes tracking/execution infeasibility from task failure,
and writes one raw JSON trace plus one manifest row per rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK_NAME = "Isaac-Libero-Franka-Hybrid-ContactForce-v0"
TRACKING_TOLERANCE_N = 0.25
MIN_BILATERAL_FORCE_N = 0.1
LIFT_DELTA_M = 0.005
COARSE_FORCES_N = (1.35, 2.15, 4.85)
SAFE_FORCE_MIN_N = 0.75
SAFE_FORCE_MAX_N = 5.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task_suite", default="libero_goal")
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--object_name", default="akita_black_bowl_1")
    parser.add_argument("--snapshot_step", type=int, default=60)
    parser.add_argument("--demo_ids", type=int, nargs="+", default=None)
    parser.add_argument("--num_contexts", type=int, default=10)
    parser.add_argument("--max_rollouts", type=int, default=70)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--asset_root",
        default=None,
        help="Optional local or remote Isaac asset root used instead of the Kit cloud default.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return float(np.asarray(value).reshape(-1)[0])


def jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def cpu_tree(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def make_action(env: Any, eef_pose: Any, gripper_pos: Any) -> Any:
    import torch
    import isaaclab.utils.math as math_utils
    from tac_manip.tasks.manipulation.libero.mdp.observations import (
        contact_force_in_gripper_frame,
    )

    pose = eef_pose.to(env.device)
    grip = gripper_pos.to(env.device)
    axis_angle = math_utils.axis_angle_from_quat(pose[3:7].reshape(1, 4))[0]
    action = torch.zeros((1, 13), device=env.device, dtype=pose.dtype)
    action[0, 0:3] = pose[0:3]
    action[0, 3:6] = axis_angle
    action[0, 6] = grip.reshape(-1)[0]
    measured = contact_force_in_gripper_frame(
        env, contact_sensor_name="contact_gripper", history_length=1
    )[:, -1, :, :].reshape(1, 6)
    action[:, 7:13] = measured
    return action


def measure(env: Any, object_name: str) -> dict[str, Any]:
    from tac_manip.tasks.manipulation.libero.mdp.observations import (
        contact_force_in_gripper_frame,
    )

    forces = contact_force_in_gripper_frame(
        env, contact_sensor_name="contact_gripper", history_length=1
    )[0, -1]
    left_z = abs(scalar(forces[0, 2]))
    right_z = abs(scalar(forces[1, 2]))
    robot = env.scene["robot"]
    finger_ids = robot.find_joints(env.cfg.gripper_joint_names)[0]
    return {
        "measured_force_n": 2.0 * min(left_z, right_z),
        "left_force_n": left_z,
        "right_force_n": right_z,
        "finger_gap_m": scalar(robot.data.joint_pos[0, finger_ids].mean()),
        "bilateral_contact": bool(
            left_z >= MIN_BILATERAL_FORCE_N and right_z >= MIN_BILATERAL_FORCE_N
        ),
        "object_height_m": scalar(env.scene[object_name].data.root_pos_w[0, 2]),
        "object_position_w": jsonable(env.scene[object_name].data.root_pos_w[0]),
    }


def release_index(actions: Any, snapshot_step: int) -> int:
    import torch

    indices = torch.where(
        (actions[:, 7] > 0.0)
        & (torch.arange(actions.shape[0], device=actions.device) > snapshot_step)
    )[0]
    return int(indices[0].item()) if indices.numel() else int(actions.shape[0])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def branch_outcome(
    rows: list[dict[str, Any]],
    *,
    target_force_n: float | None,
    preload: dict[str, Any],
    release_at: int,
    baseline_context_valid: bool,
) -> dict[str, Any]:
    pre_release = [row for row in rows if row["action_index"] < release_at]
    contact_rows = [row for row in pre_release if row["bilateral_contact"]]
    last_contact = contact_rows[-1] if contact_rows else None
    # Select the best stable, consecutive bilateral-contact window before the
    # expected release.  A late release transient must not turn an otherwise
    # tracked target into EXECUTION_INFEASIBLE.
    contact_windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in pre_release:
        if row["bilateral_contact"] and (not current or row["action_index"] == current[-1]["action_index"] + 1):
            current.append(row)
        else:
            if current:
                contact_windows.append(current)
            current = [row] if row["bilateral_contact"] else []
    if current:
        contact_windows.append(current)
    candidate_windows = [window[-8:] for window in contact_windows if len(window) >= 3]
    if target_force_n is not None and candidate_windows:
        tracking_window = min(
            candidate_windows,
            key=lambda window: abs(float(np.mean([r["measured_force_n"] for r in window])) - float(target_force_n)),
        )
    else:
        tracking_window = candidate_windows[-1] if candidate_windows else []
    measured_values = [float(row["measured_force_n"]) for row in tracking_window]
    measured_mean = float(np.mean(measured_values)) if measured_values else None
    measured_final = (
        float(last_contact["measured_force_n"]) if last_contact is not None else None
    )
    tracking_error = (
        abs(float(measured_mean) - float(target_force_n))
        if target_force_n is not None and measured_mean is not None
        else None
    )
    statuses = [row.get("native_force_status", {}) for row in pre_release]
    saturations = sorted(
        {
            str(status.get("saturation"))
            for status in statuses
            if status.get("saturation") not in (None, "NONE")
        }
    )
    failure_reasons = sorted(
        {
            str(status.get("failure_reason"))
            for status in statuses
            if status.get("failure_reason")
        }
    )
    execution_feasible = target_force_n is None or bool(
        contact_rows
        and measured_mean is not None
        and tracking_error is not None
        and tracking_error <= TRACKING_TOLERANCE_N
        and not saturations
    )
    tracking_valid = target_force_n is None or bool(
        execution_feasible and measured_final is not None
    )
    pre_release_contact_loss = next(
        (row["action_index"] for row in pre_release if not row["bilateral_contact"]),
        None,
    )
    release_reached = any(row["action_index"] >= release_at for row in rows)
    max_height = max(
        (float(row["object_height_m"]) for row in pre_release),
        default=float(preload["object_height_m"]),
    )
    lift_success = bool(max_height >= float(preload["object_height_m"]) + LIFT_DELTA_M)
    transport_success = bool(
        lift_success
        and pre_release_contact_loss is None
        and release_reached
    )
    task_success: bool | None
    if target_force_n is not None and not execution_feasible:
        task_success = None
    else:
        task_success = bool(transport_success)
    if not execution_feasible and target_force_n is not None:
        execution_failure_reason = ";".join(failure_reasons) or (
            "FORCE_TRACKING_OUTSIDE_TOLERANCE"
        )
    else:
        execution_failure_reason = None
    if task_success is True:
        task_failure_reason = None
    elif task_success is None:
        task_failure_reason = "TASK_NOT_LABELED_EXECUTION_INFEASIBLE"
    elif pre_release_contact_loss is not None:
        task_failure_reason = "PREMATURE_CONTACT_LOSS"
    elif not lift_success:
        task_failure_reason = "NO_LIFT_BEFORE_RELEASE"
    else:
        task_failure_reason = "DOWNSTREAM_CONTINUATION_INCOMPLETE"
    return {
        "requested_force_n": target_force_n,
        "measured_force_mean_n": measured_mean,
        "measured_force_final_n": measured_final,
        "tracking_error_n": tracking_error,
        "tracking_window_action_start": tracking_window[0]["action_index"] if tracking_window else None,
        "tracking_window_action_end": tracking_window[-1]["action_index"] if tracking_window else None,
        "execution_feasible": bool(execution_feasible),
        "force_tracking_valid": bool(tracking_valid),
        "saturation_status": saturations or "NONE",
        "execution_failure_reason": execution_failure_reason,
        "lift_success": lift_success,
        "transport_success": transport_success,
        "release_reached": release_reached,
        "task_success": task_success,
        "drop_step": pre_release_contact_loss,
        "task_failure_reason": task_failure_reason,
        "baseline_context_valid": bool(baseline_context_valid),
        "pre_release_max_height_m": max_height,
        "pre_release_contact_loss": pre_release_contact_loss,
    }


def run_branch(
    env: Any,
    *,
    snapshot: Any,
    eef: Any,
    grip: Any,
    snapshot_step: int,
    release_at: int,
    object_name: str,
    term: Any,
    target_force_n: float | None,
    phase: str,
    preload: dict[str, Any],
    baseline_context_valid: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    env.reset_to(snapshot, torch.tensor([0], device=env.device), is_relative=False)
    term.clear_target_squeeze_force_n()
    if target_force_n is not None:
        term.set_target_squeeze_force_n(float(target_force_n))
    rows: list[dict[str, Any]] = []
    for index in range(snapshot_step + 1, eef.shape[0]):
        action = make_action(env, eef[index], grip[index])
        arm_target = action[0, 0:6].detach().cpu().tolist()
        _, _, terminated, truncated, _ = env.step(action)
        row = measure(env, object_name)
        row.update(
            {
                "physics_env_step": len(rows),
                "action_index": index,
                "phase": phase,
                "requested_force_n": target_force_n,
                "arm_target": arm_target,
                "terminated": bool(scalar(terminated)),
                "truncated": bool(scalar(truncated)),
                "native_force_status": term.get_force_status(),
                "d_cmd_m": scalar(term.last_d_cmd[0]),
            }
        )
        rows.append(row)
    outcome = branch_outcome(
        rows,
        target_force_n=target_force_n,
        preload=preload,
        release_at=release_at,
        baseline_context_valid=baseline_context_valid,
    )
    return rows, outcome


def arm_invariance(
    baseline_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]]
) -> bool:
    baseline = {int(row["action_index"]): row["arm_target"] for row in baseline_rows}
    branch = {int(row["action_index"]): row["arm_target"] for row in branch_rows}
    if baseline.keys() != branch.keys():
        return False
    return all(
        np.allclose(np.asarray(baseline[index], dtype=float), np.asarray(branch[index], dtype=float), atol=1e-7, rtol=0.0)
        for index in baseline
    )


def unique_forces(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        value = round(float(np.clip(value, SAFE_FORCE_MIN_N, SAFE_FORCE_MAX_N)), 3)
        if not any(abs(value - old) < 1e-6 for old in result):
            result.append(value)
    return result


def refinement_forces(coarse: list[float], outcomes: list[dict[str, Any]]) -> list[float]:
    valid = [
        (float(force), outcome)
        for force, outcome in zip(coarse, outcomes)
        if outcome["execution_feasible"]
    ]
    successes = sorted(force for force, outcome in valid if outcome["task_success"] is True)
    failures = sorted(force for force, outcome in valid if outcome["task_success"] is False)
    brackets = [
        (fail, success)
        for fail in failures
        for success in successes
        if fail < success
    ]
    if brackets:
        low, high = min(brackets, key=lambda pair: pair[1] - pair[0])
        width = high - low
        return [low + width * 0.25, low + width * 0.50, low + width * 0.75]
    if successes:
        high = min(successes)
        low = max(SAFE_FORCE_MIN_N, high - 1.5)
        return [low, low + (high - low) * 0.35, low + (high - low) * 0.70]
    if failures:
        low = max(failures)
        high = min(SAFE_FORCE_MAX_N, low + 1.5)
        return [low + (high - low) * 0.35, low + (high - low) * 0.70, high]
    return [1.05, 2.55, 4.75]


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(row[key])) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys})


def configure_env(args: argparse.Namespace) -> Any:
    import gymnasium as gym
    import tac_manip  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = parse_env_cfg(TASK_NAME, device=args.device, num_envs=1)
    cfg.env_name = TASK_NAME
    if hasattr(cfg, "seed"):
        cfg.seed = args.seed
    cfg.terminations.time_out = None
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    cfg.observations.policy.concatenate_terms = False
    cfg.sim.physx.enable_ccd = True
    cfg.actions.arm_action.squeeze_kp = 0.0008
    cfg.actions.arm_action.squeeze_deadzone = 0.25
    cfg.actions.arm_action.meas_force_filter_alpha = 1.0
    cfg.scene.contact_gripper.prim_path = "{ENV_REGEX_NS}/Robot/panda_.*finger"
    for name in ("agentview_cam", "eye_in_hand_cam"):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    return gym.make(TASK_NAME, cfg=cfg).unwrapped


def main() -> None:
    args = parse_args()
    os.environ["TASK_SUITE"] = args.task_suite
    os.environ["TASK_ID"] = str(args.task_id)
    app = AppLauncher(args).app
    if args.asset_root:
        import carb

        carb.settings.get_settings().set(
            "/persistent/isaac/asset_root/cloud", args.asset_root.rstrip("/")
        )

    import torch
    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    output = Path(args.output)
    for directory in ("contexts", "rollouts", "traces", "manifests", "qa", "summary"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    env = configure_env(args)
    ds = HDF5DatasetFileHandler()
    ds.open(args.dataset)
    episode_names = list(ds.get_episode_names())
    requested_demo_ids = args.demo_ids or list(range(min(len(episode_names), max(args.num_contexts * 2, args.num_contexts))))
    if any(index < 0 or index >= len(episode_names) for index in requested_demo_ids):
        raise ValueError(f"demo id is outside episode list [0, {len(episode_names) - 1}]")
    term = env.action_manager.get_term("arm_action")
    if not callable(getattr(term, "set_target_squeeze_force_n", None)):
        raise RuntimeError("active arm_action is not native Tabero ForcePositionAction")

    context_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    valid_contexts = 0
    attempted_rollouts = 0
    rng = np.random.default_rng(args.seed)
    print(f"PILOT_OUTPUT={output}", flush=True)
    print(f"EPISODES_AVAILABLE={len(episode_names)} CANDIDATE_CONTEXTS={requested_demo_ids}", flush=True)

    for demo_index in requested_demo_ids:
        if valid_contexts >= args.num_contexts or attempted_rollouts >= args.max_rollouts:
            break
        episode_name = episode_names[demo_index]
        context_id = f"{args.task_suite}_task{args.task_id}_demo{demo_index}_step{args.snapshot_step}"
        try:
            episode = ds.load_episode(episode_name, env.device)
            actions8 = episode.data["actions"]
            eef = episode.data["obs"]["eef_pose"]
            grip = episode.data["obs"]["gripper_pos"]
            if eef.shape[0] != actions8.shape[0] or eef.shape[0] <= args.snapshot_step + 1:
                raise RuntimeError("trajectory length is incompatible with snapshot step")
            env.reset()
            env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
            term.clear_target_squeeze_force_n()
            for index in range(args.snapshot_step + 1):
                env.step(make_action(env, eef[index], grip[index]))
            snapshot = env.scene.get_state(is_relative=False)
            preload = measure(env, args.object_name)
            release_at = release_index(actions8, args.snapshot_step)
            snapshot_path = output / "contexts" / f"{context_id}.pt"
            torch.save(cpu_tree(snapshot), snapshot_path)

            baseline_rows, baseline = run_branch(
                env,
                snapshot=snapshot,
                eef=eef,
                grip=grip,
                snapshot_step=args.snapshot_step,
                release_at=release_at,
                object_name=args.object_name,
                term=term,
                target_force_n=None,
                phase="baseline_continuation",
                preload=preload,
                baseline_context_valid=False,
            )
            baseline_valid = bool(
                baseline["task_success"] is True
                and baseline["release_reached"]
                and baseline["pre_release_contact_loss"] is None
            )
            context_record = {
                "context_id": context_id,
                "task": TASK_NAME,
                "task_suite": args.task_suite,
                "task_id": args.task_id,
                "demo_index": demo_index,
                "episode_name": episode_name,
                "target_object": args.object_name,
                "grasp_step": args.snapshot_step,
                "continuation_start": args.snapshot_step + 1,
                "release_index": release_at,
                "trajectory_id": episode_name,
                "snapshot_path": str(snapshot_path),
                "baseline_context_valid": baseline_valid,
                "baseline": baseline,
                "split_group": context_id,
                "status": "VALID" if baseline_valid else "CONTEXT_INVALID_REPLAY",
            }
            context_rows.append(context_record)
            write_json(output / "contexts" / f"{context_id}.json", context_record)
            attempted_rollouts += 1
            baseline_rollout_id = f"{context_id}__baseline"
            write_json(output / "traces" / f"{baseline_rollout_id}.json", {
                "rollout_id": baseline_rollout_id,
                "context_id": context_id,
                "branch_type": "baseline",
                "metadata": context_record,
                "trace": baseline_rows,
            })
            baseline_row = {
                "rollout_id": baseline_rollout_id,
                "context_id": context_id,
                "branch_type": "baseline",
                "requested_force_n": "",
                **baseline,
            }
            rollout_rows.append(baseline_row)
            print(f"CONTEXT={context_id} BASELINE_VALID={baseline_valid} PRE_RELEASE_LOSS={baseline['pre_release_contact_loss']}", flush=True)
            if not baseline_valid:
                continue

            valid_contexts += 1
            jitter = float(rng.uniform(-0.08, 0.08))
            coarse = unique_forces([force + jitter for force in COARSE_FORCES_N])
            branch_records: list[tuple[float, list[dict[str, Any]], dict[str, Any]]] = []
            for stage, forces in (("coarse", coarse),):
                for force in forces:
                    if attempted_rollouts >= args.max_rollouts:
                        break
                    rows, outcome = run_branch(
                        env, snapshot=snapshot, eef=eef, grip=grip,
                        snapshot_step=args.snapshot_step, release_at=release_at,
                        object_name=args.object_name, term=term,
                        target_force_n=force, phase=f"{stage}_force_branch",
                        preload=preload, baseline_context_valid=True,
                    )
                    branch_records.append((force, rows, outcome))
                    outcome["arm_action_invariance"] = arm_invariance(baseline_rows, rows)
                    attempted_rollouts += 1
                    rollout_id = f"{context_id}__force_{force:.3f}N"
                    metadata = {**context_record, "rollout_id": rollout_id, "branch_type": "force", "sampling_stage": stage}
                    write_json(output / "traces" / f"{rollout_id}.json", {"rollout_id": rollout_id, "context_id": context_id, "metadata": metadata, "outcome": outcome, "trace": rows})
                    rollout_rows.append({"rollout_id": rollout_id, "context_id": context_id, "branch_type": "force", "sampling_stage": stage, **outcome})
                    print(f"ROLLOUT={rollout_id} FEASIBLE={outcome['execution_feasible']} SUCCESS={outcome['task_success']} MEAN_FORCE={outcome['measured_force_mean_n']} DROP={outcome['drop_step']}", flush=True)
            coarse_outcomes = [record[2] for record in branch_records]
            refine = unique_forces(refinement_forces([record[0] for record in branch_records], coarse_outcomes))
            existing = {round(record[0], 3) for record in branch_records}
            for force in refine:
                if attempted_rollouts >= args.max_rollouts or len(branch_records) >= 6:
                    break
                if round(force, 3) in existing:
                    continue
                rows, outcome = run_branch(
                    env, snapshot=snapshot, eef=eef, grip=grip,
                    snapshot_step=args.snapshot_step, release_at=release_at,
                    object_name=args.object_name, term=term,
                    target_force_n=force, phase="boundary_refinement",
                    preload=preload, baseline_context_valid=True,
                )
                branch_records.append((force, rows, outcome))
                outcome["arm_action_invariance"] = arm_invariance(baseline_rows, rows)
                existing.add(round(force, 3))
                attempted_rollouts += 1
                rollout_id = f"{context_id}__force_{force:.3f}N"
                metadata = {**context_record, "rollout_id": rollout_id, "branch_type": "force", "sampling_stage": "boundary_refinement"}
                write_json(output / "traces" / f"{rollout_id}.json", {"rollout_id": rollout_id, "context_id": context_id, "metadata": metadata, "outcome": outcome, "trace": rows})
                rollout_rows.append({"rollout_id": rollout_id, "context_id": context_id, "branch_type": "force", "sampling_stage": "boundary_refinement", **outcome})
                print(f"ROLLOUT={rollout_id} FEASIBLE={outcome['execution_feasible']} SUCCESS={outcome['task_success']} MEAN_FORCE={outcome['measured_force_mean_n']} DROP={outcome['drop_step']}", flush=True)
            context_record["force_branch_count"] = len(branch_records)
            context_record["force_branches"] = [
                {"requested_force_n": force, **outcome} for force, _, outcome in branch_records
            ]
            context_record["boundary_bracket"] = boundary_bracket(branch_records)
            write_json(output / "contexts" / f"{context_id}.json", context_record)

        except Exception as exc:
            context_rows.append({
                "context_id": context_id,
                "task": TASK_NAME,
                "task_suite": args.task_suite,
                "task_id": args.task_id,
                "demo_index": demo_index,
                "episode_name": episode_name,
                "target_object": args.object_name,
                "status": "REPLAY_FAILURE",
                "replay_failure_reason": f"{type(exc).__name__}: {exc}",
                "baseline_context_valid": False,
                "split_group": context_id,
            })
            print(f"CONTEXT={context_id} REPLAY_FAILURE={type(exc).__name__}: {exc}", flush=True)

    csv_write(output / "manifests" / "context_manifest.csv", context_rows)
    csv_write(output / "manifests" / "pilot_manifest.csv", rollout_rows)
    summary = summarize(context_rows, rollout_rows, attempted_rollouts, valid_contexts, args)
    write_json(output / "summary" / "pilot_summary.json", summary)
    write_json(output / "qa" / "collection_config.json", {
        "pilot_target_rollouts": 60,
        "tracking_tolerance_n": TRACKING_TOLERANCE_N,
        "coarse_forces_n": COARSE_FORCES_N,
        "safe_force_range_n": [SAFE_FORCE_MIN_N, SAFE_FORCE_MAX_N],
        "native_action_semantics": "absolute_EE_pose_plus_native_force_position",
        "object_specific_force_lookup_used": False,
        "arm_continuous_during_collection": True,
    })
    print("PILOT_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    env.close()
    app.close()


def boundary_bracket(records: list[tuple[float, list[dict[str, Any]], dict[str, Any]]]) -> dict[str, Any] | None:
    valid = [(force, outcome) for force, _, outcome in records if outcome["execution_feasible"]]
    failures = sorted(force for force, outcome in valid if outcome["task_success"] is False)
    successes = sorted(force for force, outcome in valid if outcome["task_success"] is True)
    candidates = [(fail, success) for fail in failures for success in successes if fail < success]
    if not candidates:
        return None
    fail, success = min(candidates, key=lambda pair: pair[1] - pair[0])
    return {"highest_failed_force_n": fail, "lowest_successful_force_n": success, "width_n": success - fail}


def summarize(context_rows: list[dict[str, Any]], rollout_rows: list[dict[str, Any]], attempted: int, valid_contexts: int, args: argparse.Namespace) -> dict[str, Any]:
    force_rows = [row for row in rollout_rows if row.get("branch_type") == "force"]
    valid_force_rows = [row for row in force_rows if row.get("execution_feasible") is True and row.get("force_tracking_valid") is True]
    successes = [row for row in valid_force_rows if row.get("task_success") is True]
    failures = [row for row in valid_force_rows if row.get("task_success") is False]
    by_context: dict[str, list[dict[str, Any]]] = {}
    for row in valid_force_rows:
        by_context.setdefault(str(row["context_id"]), []).append(row)
    both = [cid for cid, rows in by_context.items() if any(r.get("task_success") is True for r in rows) and any(r.get("task_success") is False for r in rows)]
    all_success = [cid for cid, rows in by_context.items() if rows and all(r.get("task_success") is True for r in rows)]
    all_fail = [cid for cid, rows in by_context.items() if rows and all(r.get("task_success") is False for r in rows)]
    boundaries = [row.get("boundary_bracket") for row in context_rows if row.get("boundary_bracket")]
    forces = [float(row["requested_force_n"]) for row in force_rows if row.get("requested_force_n") not in (None, "")]
    infeasible = [row for row in force_rows if row.get("execution_feasible") is False]
    invalid_contexts = [row for row in context_rows if row.get("status") == "CONTEXT_INVALID_REPLAY"]
    replay_failures = [row for row in context_rows if row.get("status") == "REPLAY_FAILURE"]
    sufficient = bool(
        len(valid_force_rows) >= 30
        and successes
        and failures
        and len(both) >= max(2, min(3, valid_contexts))
        and len(boundaries) >= max(2, min(3, valid_contexts))
        and len(set(round(force, 3) for force in forces)) >= 10
    )
    return {
        "pilot_target_rollouts": 60,
        "pilot_attempted_rollouts": attempted,
        "pilot_valid_executed_rollouts": len(valid_force_rows),
        "context_count": valid_contexts,
        "context_records": len(context_rows),
        "task_success_count": len(successes),
        "task_fail_count": len(failures),
        "execution_infeasible_count": len(infeasible),
        "invalid_context_count": len(invalid_contexts),
        "replay_failure_count": len(replay_failures),
        "success_rate_valid_executed": len(successes) / len(valid_force_rows) if valid_force_rows else None,
        "fail_rate_valid_executed": len(failures) / len(valid_force_rows) if valid_force_rows else None,
        "contexts_with_both_success_and_fail": len(both),
        "contexts_all_success": len(all_success),
        "contexts_all_fail": len(all_fail),
        "boundary_context_count": len(boundaries),
        "boundary_context_rate": len(boundaries) / valid_contexts if valid_contexts else None,
        "force_values_are_continuous": len(set(round(force, 3) for force in forces)) >= 10 and any(abs(force - round(force)) > 1e-6 for force in forces),
        "force_min_n": min(forces) if forces else None,
        "force_max_n": max(forces) if forces else None,
        "force_mean_n": float(np.mean(forces)) if forces else None,
        "force_median_n": float(np.median(forces)) if forces else None,
        "unique_force_count": len(set(round(force, 3) for force in forces)),
        "object_specific_force_lookup_used": False,
        "tabero_native_force_position_used": True,
        "arm_continuous_during_collection": True,
        "pilot_data_sufficient_for_training_smoke": sufficient,
        "collector_stable": bool(valid_contexts and not replay_failures),
        "parallelism_benchmark_run": False,
        "recommended_collection_parallelism": "NOT_EVALUATED_IN_PILOT",
        "split_grouping": "context_id/root; no random branch split",
        "dataset_definition": "same context + continuous requested force + native measured force + downstream outcome",
        "task_suite": args.task_suite,
        "task_id": args.task_id,
    }


if __name__ == "__main__":
    main()
