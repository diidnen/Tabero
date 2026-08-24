import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.episode_video import load_mode6_capture  # noqa: E402
from scripts.tools.export_mode6_five_panel_video import (  # noqa: E402
    export_mode6_five_panel_video,
)


def _write_capture(
    root: Path, *, frames: int = 4, scalar_schema_version: int = 2
) -> Path:
    exp_dir = root / "exp_000"
    camera_dir = exp_dir / "camera_rgb"
    tactile_dir = exp_dir / "tactile_markers_rgb"
    camera_dir.mkdir(parents=True)
    tactile_dir.mkdir(parents=True)
    (exp_dir / "exp_meta.json").write_text(
        json.dumps(
            {
                "task_suite": "libero_object",
                "task_id": 0,
                "exp_idx": 0,
                "prompt": "pick up the alphabet soup and place it in the basket gently",
                "adverb_used": "gently",
                "mode6_scalar_schema_version": scalar_schema_version,
                "gripper_command_semantics": "controller_d_cmd",
                "gripper_measured_semantics": "mean_raw_finger_joint_position",
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for frame_index in range(frames):
        agent = np.full((48, 64, 3), frame_index * 10, dtype=np.uint8)
        eye = np.full((48, 64, 3), 40 + frame_index * 10, dtype=np.uint8)
        tactile = np.full((24, 32, 3), 80 + frame_index * 10, dtype=np.uint8)
        assert cv2.imwrite(str(camera_dir / f"frame_{frame_index:04d}_agentview.png"), agent)
        assert cv2.imwrite(str(camera_dir / f"frame_{frame_index:04d}_eye.png"), eye)
        assert cv2.imwrite(
            str(tactile_dir / f"frame_{frame_index:04d}_gsmini_left_markers_rgb.png"), tactile
        )
        assert cv2.imwrite(
            str(tactile_dir / f"frame_{frame_index:04d}_gsmini_right_markers_rgb.png"), tactile
        )
        payload = {
                    "frame": frame_index,
                    "mode6_scalar_schema_version": scalar_schema_version,
                    "squeeze_pred": float(frame_index + 1),
                    "squeeze_meas": float(frame_index) / 2.0,
                    "gripper_pred": 0.041 - 0.005 * frame_index,
                    "gripper_cmd": 0.04 - 0.005 * frame_index,
                    "gripper_meas": 0.039 - 0.004 * frame_index,
                    "gripper_joint_left": 0.038 - 0.004 * frame_index,
                    "gripper_joint_right": 0.04 - 0.004 * frame_index,
                }
        if scalar_schema_version >= 3:
            payload["squeeze_target_eff"] = 1.9 * float(frame_index + 1)
            payload["squeeze_meas_raw"] = 0.6 * float(frame_index)
        rows.append(json.dumps(payload))
    (exp_dir / "forces.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    trace_dir = root / "traces"
    trace_dir.mkdir(exist_ok=True)
    trace_rows = [
        json.dumps(
            {
                "task_suite": "libero_object",
                "task_id": 0,
                "experiment_index": 0,
                "env_step_index": frame_index,
                "squeeze_pred": float(frame_index + 1),
                "squeeze_meas": float(frame_index) / 2.0,
                "effective_squeeze_target_n": 1.9 * float(frame_index + 1),
                "reward_squeeze_meas_raw": 0.6 * float(frame_index),
            }
        )
        for frame_index in range(frames)
    ]
    (trace_dir / "libero_object_task0.jsonl").write_text(
        "\n".join(trace_rows) + "\n",
        encoding="utf-8",
    )
    return exp_dir


def test_export_mode6_five_panel_video_encodes_and_audits(tmp_path):
    exp_dir = _write_capture(tmp_path)
    output = tmp_path / "combined_5view_force_curve.mp4"

    manifest = export_mode6_five_panel_video(
        capture_exp_dir=exp_dir,
        output=output,
        fps=20.0,
        width=640,
        height=360,
    )

    assert output.is_file()
    assert output.with_suffix(".manifest.json").is_file()
    assert manifest["capture_boundary"]["frames"] == 4
    assert manifest["schema_version"] == 5
    assert manifest["video"]["codec"] == "h264"
    assert manifest["video"]["pixel_format"] == "yuv420p"
    assert manifest["video"]["width"] == 640
    assert manifest["video"]["height"] == 360
    assert manifest["video"]["frames"] == 4
    assert manifest["video"]["fps"] == 20.0
    assert manifest["force_semantics"]["reward_aligned_raw_force"] is True
    assert manifest["force_semantics"]["step_trace_overlay"] is True
    assert manifest["force_semantics"]["controller_effective_target"] == (
        "step trace effective_squeeze_target_n"
    )
    assert "step_trace.jsonl" in manifest["source_hashes"]
    assert manifest["gripper_semantics"]["display_unit"] == "mm"
    assert manifest["gripper_semantics"]["prediction_plotted"] is True
    assert manifest["gripper_semantics"]["prediction"] == (
        "forces.jsonl gripper_pred (raw model d_pred)"
    )
    assert manifest["gripper_semantics"]["command_is_controller_corrected_d_cmd"] is True
    assert manifest["gripper_semantics"]["measured_is_signed_policy_observation_mean"] is False


def test_schema_v3_capture_exports_effective_target_and_reward_raw_force(tmp_path):
    exp_dir = _write_capture(tmp_path, scalar_schema_version=3)
    capture = load_mode6_capture(exp_dir)
    output = tmp_path / "combined_5view_force_tracking_curve.mp4"

    manifest = export_mode6_five_panel_video(
        capture_exp_dir=exp_dir,
        output=output,
        fps=20.0,
        width=640,
        height=360,
    )

    assert capture.scalar_schema_version == 3
    assert capture.squeeze_target_eff.tolist() == pytest.approx([1.9, 3.8, 5.7, 7.6])
    assert capture.squeeze_meas_raw.tolist() == pytest.approx([0.0, 0.6, 1.2, 1.8])
    assert manifest["force_semantics"]["controller_effective_target"] == (
        "forces.jsonl squeeze_target_eff"
    )
    assert manifest["force_semantics"]["plotted_measured"] == "measured_raw"
    assert manifest["force_semantics"]["reward_aligned_raw_force"] is True
    assert manifest["force_semantics"]["step_trace_overlay"] is False


def test_schema_v2_export_requires_matching_step_trace(tmp_path):
    exp_dir = _write_capture(tmp_path, scalar_schema_version=2)
    (tmp_path / "traces" / "libero_object_task0.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="Pass --step-trace"):
        export_mode6_five_panel_video(
            capture_exp_dir=exp_dir,
            output=tmp_path / "missing_trace.mp4",
            width=640,
            height=360,
        )


def test_schema_v2_export_rejects_misaligned_step_trace(tmp_path):
    exp_dir = _write_capture(tmp_path, scalar_schema_version=2)
    trace_path = tmp_path / "traces" / "libero_object_task0.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["squeeze_pred"] = 999.0
    trace_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="squeeze_pred does not align"):
        export_mode6_five_panel_video(
            capture_exp_dir=exp_dir,
            output=tmp_path / "misaligned_trace.mp4",
            width=640,
            height=360,
        )


def test_mode6_loader_rejects_cross_stream_frame_mismatch(tmp_path):
    exp_dir = _write_capture(tmp_path)
    (exp_dir / "tactile_markers_rgb" / "frame_0003_gsmini_right_markers_rgb.png").unlink()

    with pytest.raises(ValueError, match="do not match forces.jsonl"):
        load_mode6_capture(exp_dir)


def test_mode6_loader_rejects_nonfinite_force(tmp_path):
    exp_dir = _write_capture(tmp_path, frames=1)
    (exp_dir / "forces.jsonl").write_text(
        '{"frame": 0, "squeeze_pred": NaN, "squeeze_meas": 1.0, '
        '"mode6_scalar_schema_version": 2, '
        '"gripper_pred": 0.041, "gripper_cmd": 0.04, "gripper_meas": 0.039, '
        '"gripper_joint_left": 0.038, "gripper_joint_right": 0.04}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be finite or null"):
        load_mode6_capture(exp_dir)


def test_mode6_loader_rejects_missing_gripper_scalar(tmp_path):
    exp_dir = _write_capture(tmp_path, frames=1)
    (exp_dir / "forces.jsonl").write_text(
        '{"frame": 0, "mode6_scalar_schema_version": 2, '
        '"squeeze_pred": 1.0, "squeeze_meas": 0.5, "gripper_cmd": 0.04}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gripper_meas"):
        load_mode6_capture(exp_dir)


def test_mode6_loader_rejects_legacy_signed_gripper_capture(tmp_path):
    exp_dir = _write_capture(tmp_path, frames=1)
    meta_path = exp_dir / "exp_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("mode6_scalar_schema_version")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy captures used a signed gripper observation"):
        load_mode6_capture(exp_dir)


def test_mode6_loader_rejects_inconsistent_measured_joint_mean(tmp_path):
    exp_dir = _write_capture(tmp_path, frames=1)
    force_path = exp_dir / "forces.jsonl"
    payload = json.loads(force_path.read_text(encoding="utf-8"))
    payload["gripper_meas"] = 0.02
    force_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not equal mean"):
        load_mode6_capture(exp_dir)


def test_export_refuses_to_overwrite_video_or_manifest(tmp_path):
    exp_dir = _write_capture(tmp_path, frames=1)
    output = tmp_path / "combined_5view_force_curve.mp4"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_mode6_five_panel_video(capture_exp_dir=exp_dir, output=output, width=640, height=360)
