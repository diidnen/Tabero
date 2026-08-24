#!/usr/bin/env python3
"""Export one strict debug-mode-6 capture as a five-panel H.264 video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.episode_video import (  # noqa: E402
    H264FrameWriter,
    compose_five_panel_frame,
    load_mode6_capture,
    render_force_gripper_dashboard,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bgr(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"failed to decode image: {path}")
    return frame


def _resolve_step_trace_path(
    capture_exp_dir: Path,
    *,
    task_suite: str,
    task_id: int,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        resolved = Path(explicit_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"step trace does not exist: {resolved}")
        return resolved

    filename = f"{task_suite}_task{task_id}.jsonl"
    for ancestor in (capture_exp_dir, *capture_exp_dir.parents):
        candidate = ancestor / "traces" / filename
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "mode-6 scalar schema v2 does not contain controller-effective or raw "
        "measured squeeze. Pass --step-trace or place the matching trace at "
        f"<run>/traces/{filename}."
    )


def _finite_trace_float(payload: dict, field: str, *, line_number: int) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"step trace line {line_number} field {field!r} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"step trace line {line_number} field {field!r} must be finite.")
    return result


def _resolve_force_plot_series(
    capture,
    *,
    step_trace_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, Path | None]:
    """Return effective target and raw measurement, overlaying schema-v2 captures from trace."""

    if capture.scalar_schema_version >= 3:
        return capture.squeeze_target_eff, capture.squeeze_meas_raw, None

    meta_path = capture.exp_dir / "exp_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    task_suite = meta.get("task_suite")
    task_id = meta.get("task_id")
    experiment_index = meta.get("exp_idx")
    if not isinstance(task_suite, str) or not task_suite:
        raise ValueError(f"{meta_path} must contain a non-empty string task_suite.")
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise ValueError(f"{meta_path} must contain an integer task_id.")
    if isinstance(experiment_index, bool) or not isinstance(experiment_index, int):
        raise ValueError(f"{meta_path} must contain an integer exp_idx.")

    resolved_trace = _resolve_step_trace_path(
        capture.exp_dir,
        task_suite=task_suite,
        task_id=task_id,
        explicit_path=step_trace_path,
    )
    rows_by_index: dict[int, tuple[dict, int]] = {}
    with resolved_trace.open("r", encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if not line.strip():
                raise ValueError(f"step trace line {line_number} is blank.")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"step trace line {line_number} is invalid JSON: {exc}"
                ) from exc
            if payload.get("experiment_index") != experiment_index:
                continue
            if payload.get("task_suite") != task_suite or payload.get("task_id") != task_id:
                raise ValueError(
                    f"step trace line {line_number} task identity does not match {meta_path}."
                )
            frame_index = payload.get("env_step_index")
            if isinstance(frame_index, bool) or not isinstance(frame_index, int):
                raise ValueError(
                    f"step trace line {line_number} env_step_index must be an integer."
                )
            if frame_index in rows_by_index:
                raise ValueError(
                    f"step trace contains duplicate env_step_index={frame_index} "
                    f"for experiment_index={experiment_index}."
                )
            rows_by_index[frame_index] = (payload, line_number)

    effective: list[float] = []
    measured_raw: list[float] = []
    for position, frame_index in enumerate(capture.frame_indices):
        if frame_index not in rows_by_index:
            raise ValueError(
                f"step trace is missing env_step_index={frame_index} for "
                f"experiment_index={experiment_index}."
            )
        payload, line_number = rows_by_index[frame_index]
        trace_pred = _finite_trace_float(payload, "squeeze_pred", line_number=line_number)
        trace_meas = _finite_trace_float(payload, "squeeze_meas", line_number=line_number)
        if not math.isclose(trace_pred, float(capture.squeeze_pred[position]), abs_tol=1e-7):
            raise ValueError(
                f"step trace line {line_number} squeeze_pred does not align with "
                f"forces.jsonl frame {frame_index}."
            )
        if not math.isclose(trace_meas, float(capture.squeeze_meas[position]), abs_tol=1e-7):
            raise ValueError(
                f"step trace line {line_number} squeeze_meas does not align with "
                f"forces.jsonl frame {frame_index}."
            )
        effective.append(
            _finite_trace_float(
                payload,
                "effective_squeeze_target_n",
                line_number=line_number,
            )
        )
        measured_raw.append(
            _finite_trace_float(
                payload,
                "reward_squeeze_meas_raw",
                line_number=line_number,
            )
        )
    return (
        np.asarray(effective, dtype=np.float64),
        np.asarray(measured_raw, dtype=np.float64),
        resolved_trace,
    )


def _probe_video(path: Path) -> dict:
    command = [
        "/usr/bin/ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}, got {len(streams)}")
    stream = streams[0]
    return {
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream.get("r_frame_rate"),
        "frames": int(stream["nb_frames"]),
    }


def export_mode6_five_panel_video(
    *,
    capture_exp_dir: Path,
    output: Path,
    fps: float = 20.0,
    width: int = 1920,
    height: int = 1080,
    overwrite: bool = False,
    manifest_path: Path | None = None,
    step_trace_path: Path | None = None,
) -> dict:
    """Encode a five-panel video and return its auditable manifest."""

    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}.")
    if width < 640 or height < 360 or width % 2 or height % 2:
        raise ValueError(f"width/height must be even and at least 640x360, got {width}x{height}.")

    capture = load_mode6_capture(capture_exp_dir)
    resolved_output = Path(output).expanduser().resolve()
    resolved_manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else resolved_output.with_suffix(".manifest.json")
    )
    if resolved_manifest.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest {resolved_manifest}")
    squeeze_target_eff, squeeze_meas_raw, resolved_step_trace = (
        _resolve_force_plot_series(capture, step_trace_path=step_trace_path)
    )

    writer = H264FrameWriter(
        resolved_output,
        fps=float(fps),
        size=(int(width), int(height)),
        overwrite=overwrite,
    )
    try:
        chart_size = (max(240, round(width * 0.42)), max(360, round(height * 0.40)))
        for position, frame_index in enumerate(capture.frame_indices):
            force_plot = render_force_gripper_dashboard(
                capture.squeeze_pred,
                capture.squeeze_meas,
                capture.gripper_cmd,
                capture.gripper_meas,
                squeeze_target_eff=squeeze_target_eff,
                squeeze_meas_raw=squeeze_meas_raw,
                gripper_pred_m=capture.gripper_pred,
                current_index=position,
                size=chart_size,
            )
            composed = compose_five_panel_frame(
                agentview_bgr=_read_bgr(capture.agentview_paths[position]),
                eye_in_hand_bgr=_read_bgr(capture.eye_in_hand_paths[position]),
                left_tactile_bgr=_read_bgr(capture.left_tactile_paths[position]),
                right_tactile_bgr=_read_bgr(capture.right_tactile_paths[position]),
                force_plot_bgr=force_plot,
                prompt=capture.prompt,
                adverb_used=capture.adverb_used,
                size=(int(width), int(height)),
            )
            writer.write(composed)
            if frame_index != position:
                raise RuntimeError(f"unexpected non-contiguous frame index {frame_index} at position {position}")
        writer.finalize(expected_frames=capture.frame_count)
    except Exception:
        writer.abort()
        raise

    probe = _probe_video(resolved_output)
    expected_frame_rate = f"{int(fps)}/1" if float(fps).is_integer() else None
    audit_errors: list[str] = []
    if probe["codec"] != "h264":
        audit_errors.append(f"codec={probe['codec']!r}, expected 'h264'")
    if probe["pixel_format"] != "yuv420p":
        audit_errors.append(f"pixel_format={probe['pixel_format']!r}, expected 'yuv420p'")
    if (probe["width"], probe["height"]) != (int(width), int(height)):
        audit_errors.append(
            f"dimensions={probe['width']}x{probe['height']}, expected {int(width)}x{int(height)}"
        )
    if probe["frames"] != capture.frame_count:
        audit_errors.append(f"frames={probe['frames']}, expected {capture.frame_count}")
    if expected_frame_rate is not None and probe["frame_rate"] != expected_frame_rate:
        audit_errors.append(f"frame_rate={probe['frame_rate']!r}, expected {expected_frame_rate!r}")
    if audit_errors:
        raise RuntimeError("encoded video audit failed: " + "; ".join(audit_errors))

    source_hashes = {
        "exp_meta.json": _sha256(capture.exp_dir / "exp_meta.json"),
        "forces.jsonl": _sha256(capture.exp_dir / "forces.jsonl"),
    }
    if resolved_step_trace is not None:
        source_hashes["step_trace.jsonl"] = _sha256(resolved_step_trace)

    manifest = {
        "schema_version": 5,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capture_exp_dir": str(capture.exp_dir),
        "output": str(resolved_output),
        "prompt": capture.prompt,
        "adverb_used": capture.adverb_used,
        "layout": [
            "agentview",
            "eye_in_hand",
            "gsmini_left_markers_rgb",
            "gsmini_right_markers_rgb",
            (
                "squeeze_model_vs_effective_target_vs_measured_and_"
                "gripper_pred_vs_cmd_vs_meas_curves"
            ),
        ],
        "force_semantics": {
            "capture_scalar_schema_version": capture.scalar_schema_version,
            "model_raw_target": "forces.jsonl squeeze_pred",
            "controller_effective_target": (
                "forces.jsonl squeeze_target_eff"
                if capture.scalar_schema_version >= 3
                else "step trace effective_squeeze_target_n"
            ),
            "measured_raw": (
                "forces.jsonl squeeze_meas_raw"
                if capture.scalar_schema_version >= 3
                else "step trace reward_squeeze_meas_raw"
            ),
            "measured_ema": "forces.jsonl squeeze_meas",
            "plotted_measured": "measured_raw",
            "unit": "N",
            "reward_aligned_raw_force": True,
            "step_trace_overlay": resolved_step_trace is not None,
            "step_trace_path": (
                str(resolved_step_trace) if resolved_step_trace is not None else None
            ),
        },
        "gripper_semantics": {
            "prediction": "forces.jsonl gripper_pred (raw model d_pred)",
            "prediction_plotted": True,
            "command": "forces.jsonl gripper_cmd (controller-corrected and clamped d_cmd)",
            "measured": "forces.jsonl gripper_meas = mean(gripper_joint_left, gripper_joint_right)",
            "left_joint": "forces.jsonl gripper_joint_left (raw post-step joint position)",
            "right_joint": "forces.jsonl gripper_joint_right (raw post-step joint position)",
            "display_unit": "mm",
            "source_unit": "m",
            "command_is_controller_corrected_d_cmd": True,
            "measured_is_full_jaw_gap": False,
            "measured_is_signed_policy_observation_mean": False,
        },
        "capture_boundary": {
            "frames": capture.frame_count,
            "first_frame": capture.frame_indices[0],
            "last_frame": capture.frame_indices[-1],
            "rule": "Use the exact debug_mode=6 image/forces intersection; do not synthesize a terminal frame.",
        },
        "video": {
            **probe,
            "fps": float(fps),
            "sha256": _sha256(resolved_output),
        },
        "source_hashes": source_hashes,
        "audit": {
            "streams_aligned": True,
            "contiguous_frame_indices": True,
            "ffprobe_passed": True,
            "errors": [],
        },
    }
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose one debug_mode=6 experiment into a five-panel force video."
    )
    parser.add_argument("--capture-exp-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--step-trace",
        type=Path,
        default=None,
        help=(
            "Matching evaluator JSONL used to supply controller target and raw "
            "measurement for scalar-schema-v2 captures; auto-discovered by default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = export_mode6_five_panel_video(
        capture_exp_dir=args.capture_exp_dir,
        output=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        overwrite=args.overwrite,
        manifest_path=args.manifest,
        step_trace_path=args.step_trace,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
