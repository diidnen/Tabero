#!/usr/bin/env python3
"""Run Tabero's native ForcePositionAction with an opt-in scalar Newton target.

The runner is deliberately only a replay/trace harness.  All low-level arm
and gripper control remains in ForcePositionAction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--task_suite", default="libero_goal")
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--demo_id", type=int, default=0)
    p.add_argument("--object_name", required=True)
    p.add_argument("--snapshot_step", type=int, default=60)
    p.add_argument("--scene_snapshot", default=None)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--asset_root",
        default=None,
        help="Optional local or remote Isaac asset root used instead of the Kit cloud default.",
    )
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def scalar(x) -> float:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return float(np.asarray(x).reshape(-1)[0])


def make_action(env, eef_pose, gripper_pos):
    import torch
    import isaaclab.utils.math as math_utils
    from tac_manip.tasks.manipulation.libero.mdp.observations import (
        contact_force_in_gripper_frame,
    )

    pose = eef_pose.to(env.device)
    grip = gripper_pos.to(env.device)
    aa = math_utils.axis_angle_from_quat(pose[3:7].reshape(1, 4))[0]
    action = torch.zeros((1, 13), device=env.device, dtype=pose.dtype)
    action[0, 0:3] = pose[0:3]
    action[0, 3:6] = aa
    action[0, 6] = grip.reshape(-1)[0]
    force = contact_force_in_gripper_frame(
        env, contact_sensor_name="contact_gripper", history_length=1
    )[:, -1, :, :].reshape(1, 6)
    action[:, 7:13] = force
    return action


def measure(env, object_name: str) -> dict:
    import torch
    from tac_manip.tasks.manipulation.libero.mdp.observations import (
        contact_force_in_gripper_frame,
    )

    f = contact_force_in_gripper_frame(
        env, contact_sensor_name="contact_gripper", history_length=1
    )[0, -1]
    left_z, right_z = abs(scalar(f[0, 2])), abs(scalar(f[1, 2]))
    robot = env.scene["robot"]
    finger_ids = robot.find_joints(env.cfg.gripper_joint_names)[0]
    row = {
        "measured_force_n": 2.0 * min(left_z, right_z),
        "left_force_n": left_z,
        "right_force_n": right_z,
        "bilateral_contact": bool(left_z >= 0.1 and right_z >= 0.1),
        "finger_gap_m": scalar(robot.data.joint_pos[0, finger_ids].mean()),
        "object_height_m": scalar(env.scene[object_name].data.root_pos_w[0, 2]),
    }
    return row


def step_once(env, action, object_name: str, term, target, phase, index, rows):
    env.step(action)
    row = measure(env, object_name)
    row.update({"target_force_n": target, "phase": phase, "action_index": index})
    row["native_force_status"] = term.get_force_status()
    row["d_cmd_m"] = scalar(term.last_d_cmd[0])
    rows.append(row)


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

    import gymnasium as gym
    import torch
    import tac_manip  # noqa: F401
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    task = "Isaac-Libero-Franka-Hybrid-ContactForce-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.env_name = task
    cfg.terminations.time_out = None
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    cfg.observations.policy.concatenate_terms = False
    cfg.sim.physx.enable_ccd = True
    # Newton mode uses the same native law with the gain used by the
    # validated tactile configuration; native policy defaults remain intact.
    cfg.actions.arm_action.squeeze_kp = 0.0008
    cfg.actions.arm_action.squeeze_deadzone = 0.25
    cfg.actions.arm_action.meas_force_filter_alpha = 1.0
    cfg.scene.contact_gripper.prim_path = "{ENV_REGEX_NS}/Robot/panda_.*finger"
    for name in ("agentview_cam", "eye_in_hand_cam"):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    env = gym.make(task, cfg=cfg).unwrapped

    ds = HDF5DatasetFileHandler()
    ds.open(args.dataset)
    episode_names = list(ds.get_episode_names())
    episode = ds.load_episode(episode_names[args.demo_id], env.device)
    actions8 = episode.data["actions"]
    eef = episode.data["obs"]["eef_pose"]
    grip = episode.data["obs"]["gripper_pos"]
    if eef.shape[0] != actions8.shape[0]:
        raise RuntimeError("eef_pose/action length mismatch")

    env.reset()
    if args.scene_snapshot:
        snapshot = torch.load(args.scene_snapshot, map_location=env.device, weights_only=False)
        env.reset_to(snapshot, torch.tensor([0], device=env.device), is_relative=False)
    else:
        env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
    term = env.action_manager.get_term("arm_action")
    if not hasattr(term, "set_target_squeeze_force_n"):
        raise RuntimeError("active arm_action is not Tabero ForcePositionAction")
    term.clear_target_squeeze_force_n()
    if args.scene_snapshot is None:
        for i in range(args.snapshot_step + 1):
            env.step(make_action(env, eef[i], grip[i]))
        snapshot = env.scene.get_state(is_relative=False)
    preload = measure(env, args.object_name)
    snapshot_eef = eef[args.snapshot_step]
    snapshot_grip = grip[args.snapshot_step]
    release_candidates = torch.where((actions8[:, 7] > 0.0) & (torch.arange(actions8.shape[0], device=actions8.device) > args.snapshot_step))[0]
    release_index = int(release_candidates[0].item()) if release_candidates.numel() else int(eef.shape[0])
    results = {
        "task": task,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "demo_id": args.demo_id,
        "object_name": args.object_name,
        "snapshot_step": args.snapshot_step,
        "continuation_start_step": args.snapshot_step + 1,
        "action_semantics": "ABSOLUTE_EE_POSE_PLUS_NATIVE_FORCE_POSITION",
        "original_preload": preload,
        "branches": {},
    }

    for name, target in [("baseline", None), ("newton_2n", 2.0), ("newton_4n", 4.0), ("newton_6n", 6.0)]:
        env.reset_to(snapshot, torch.tensor([0], device=env.device), is_relative=False)
        term.clear_target_squeeze_force_n()
        if target is not None:
            term.set_target_squeeze_force_n(target)
        rows = []
        # Keep the native absolute arm target active while the native
        # squeeze feedback settles; this is not an arm pause or rebase.
        if target is not None:
            for _ in range(20):
                step_once(env, make_action(env, snapshot_eef, snapshot_grip), args.object_name, term, target, "newton_settle", args.snapshot_step, rows)
        for i in range(args.snapshot_step + 1, eef.shape[0]):
            step_once(env, make_action(env, eef[i], grip[i]), args.object_name, term, target, "continuation", i, rows)
        results["branches"][name] = {
            "target_force_n": target,
            "final": rows[-1] if rows else None,
            "max_object_height_m": max((r["object_height_m"] for r in rows), default=preload["object_height_m"]),
            "bilateral_all": all(r["bilateral_contact"] for r in rows),
            "release_index": release_index,
            "pre_release_contact_loss": next((r["action_index"] for r in rows if r["action_index"] < release_index and not r["bilateral_contact"]), None),
            "pre_release_retained": all(r["bilateral_contact"] for r in rows if r["action_index"] < release_index),
            "pre_release_max_height_m": max((r["object_height_m"] for r in rows if r["action_index"] < release_index), default=preload["object_height_m"]),
            "trace": rows,
        }
        print(f"BRANCH={name} TARGET={target} FINAL_FORCE={rows[-1]['measured_force_n'] if rows else None} PRE_RELEASE_CONTACT_LOSS={results['branches'][name]['pre_release_contact_loss']} PRE_RELEASE_MAX_H={results['branches'][name]['pre_release_max_height_m']:.6f}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x))
    print(f"NATIVE_FORCE_POSITION_TRACE={out}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
