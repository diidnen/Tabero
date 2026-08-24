"""Dependency-light episode video composition used by direct evaluations."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_MODE6_FRAME_RE = re.compile(r"^frame_(\d+)_(.+)\.png$")


def _safe_name(value: str) -> str:
    normalized = _SAFE_NAME_RE.sub("_", str(value).strip()).strip("._-")
    return normalized or "unknown"


def _validate_bgr_image(name: str, frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H,W,3), got {image.shape}.")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8 BGR, got {image.dtype}.")
    return image


def letterbox_bgr(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a BGR frame into ``size=(width,height)`` without distortion."""

    image = _validate_bgr_image("frame", frame)
    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {size!r}.")

    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def _optional_finite_float(value: object, *, field: str, line_number: int) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"forces.jsonl line {line_number} field {field!r} must be numeric or null.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"forces.jsonl line {line_number} field {field!r} must be finite or null.")
    return result


def _indexed_pngs(directory: Path, suffix: str) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sorted(directory.glob(f"frame_*_{suffix}.png")):
        match = _MODE6_FRAME_RE.fullmatch(path.name)
        if match is None or match.group(2) != suffix:
            continue
        frame_index = int(match.group(1))
        if frame_index in paths:
            raise ValueError(f"duplicate frame index {frame_index} for {suffix!r} under {directory}.")
        paths[frame_index] = path.resolve()
    if not paths:
        raise FileNotFoundError(f"no frame_*_{suffix}.png files found under {directory}.")
    return paths


@dataclass(frozen=True)
class Mode6Capture:
    """Strictly aligned inputs from one ``debug_mode=6`` experiment."""

    exp_dir: Path
    prompt: str
    adverb_used: str
    frame_indices: tuple[int, ...]
    agentview_paths: tuple[Path, ...]
    eye_in_hand_paths: tuple[Path, ...]
    left_tactile_paths: tuple[Path, ...]
    right_tactile_paths: tuple[Path, ...]
    scalar_schema_version: int
    squeeze_pred: np.ndarray
    squeeze_target_eff: np.ndarray
    squeeze_meas: np.ndarray
    squeeze_meas_raw: np.ndarray
    gripper_pred: np.ndarray
    gripper_cmd: np.ndarray
    gripper_meas: np.ndarray
    gripper_joint_left: np.ndarray
    gripper_joint_right: np.ndarray

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)


def load_mode6_capture(exp_dir: Path) -> Mode6Capture:
    """Load one mode-6 capture and reject any cross-stream frame mismatch."""

    resolved_exp_dir = Path(exp_dir).expanduser().resolve()
    meta_path = resolved_exp_dir / "exp_meta.json"
    forces_path = resolved_exp_dir / "forces.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing mode-6 metadata: {meta_path}")
    if not forces_path.is_file():
        raise FileNotFoundError(f"missing mode-6 force trace: {forces_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prompt = meta.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{meta_path} must contain a non-empty string field 'prompt'.")
    adverb_used = meta.get("adverb_used", "")
    if not isinstance(adverb_used, str):
        raise ValueError(f"{meta_path} field 'adverb_used' must be a string.")
    scalar_schema_version = meta.get("mode6_scalar_schema_version")
    if scalar_schema_version not in (2, 3):
        raise ValueError(
            f"{meta_path} must declare mode6_scalar_schema_version=2 or 3; "
            "legacy captures used a signed gripper observation that is not a joint-position curve."
        )
    if meta.get("gripper_command_semantics") != "controller_d_cmd":
        raise ValueError(f"{meta_path} must declare gripper_command_semantics='controller_d_cmd'.")
    if meta.get("gripper_measured_semantics") != "mean_raw_finger_joint_position":
        raise ValueError(
            f"{meta_path} must declare "
            "gripper_measured_semantics='mean_raw_finger_joint_position'."
        )

    camera_dir = resolved_exp_dir / "camera_rgb"
    tactile_dir = resolved_exp_dir / "tactile_markers_rgb"
    image_streams = {
        "agentview": _indexed_pngs(camera_dir, "agentview"),
        "eye": _indexed_pngs(camera_dir, "eye"),
        "gsmini_left_markers_rgb": _indexed_pngs(tactile_dir, "gsmini_left_markers_rgb"),
        "gsmini_right_markers_rgb": _indexed_pngs(tactile_dir, "gsmini_right_markers_rgb"),
    }

    scalar_rows: dict[
        int,
        tuple[float, float, float, float, float, float, float, float, float],
    ] = {}
    with forces_path.open("r", encoding="utf-8") as force_file:
        for line_number, line in enumerate(force_file, start=1):
            if not line.strip():
                raise ValueError(f"forces.jsonl line {line_number} is blank.")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"forces.jsonl line {line_number} is invalid JSON: {exc}") from exc
            frame_index = payload.get("frame")
            if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
                raise ValueError(f"forces.jsonl line {line_number} field 'frame' must be a non-negative integer.")
            if frame_index in scalar_rows:
                raise ValueError(f"forces.jsonl contains duplicate frame index {frame_index}.")
            if payload.get("mode6_scalar_schema_version") != scalar_schema_version:
                raise ValueError(
                    f"forces.jsonl line {line_number} must declare "
                    f"mode6_scalar_schema_version={scalar_schema_version}."
                )
            required_fields = (
                "squeeze_pred",
                "squeeze_meas",
                "gripper_pred",
                "gripper_cmd",
                "gripper_meas",
                "gripper_joint_left",
                "gripper_joint_right",
            )
            missing_fields = [field for field in required_fields if field not in payload]
            if scalar_schema_version == 3:
                missing_fields.extend(
                    field
                    for field in ("squeeze_target_eff", "squeeze_meas_raw")
                    if field not in payload
                )
            if missing_fields:
                raise ValueError(
                    f"forces.jsonl line {line_number} is missing required fields {missing_fields}."
                )
            scalar_row = (
                _optional_finite_float(payload["squeeze_pred"], field="squeeze_pred", line_number=line_number),
                (
                    _optional_finite_float(
                        payload["squeeze_target_eff"],
                        field="squeeze_target_eff",
                        line_number=line_number,
                    )
                    if scalar_schema_version == 3
                    else float("nan")
                ),
                _optional_finite_float(payload["squeeze_meas"], field="squeeze_meas", line_number=line_number),
                (
                    _optional_finite_float(
                        payload["squeeze_meas_raw"],
                        field="squeeze_meas_raw",
                        line_number=line_number,
                    )
                    if scalar_schema_version == 3
                    else float("nan")
                ),
                _optional_finite_float(payload["gripper_pred"], field="gripper_pred", line_number=line_number),
                _optional_finite_float(payload["gripper_cmd"], field="gripper_cmd", line_number=line_number),
                _optional_finite_float(payload["gripper_meas"], field="gripper_meas", line_number=line_number),
                _optional_finite_float(
                    payload["gripper_joint_left"], field="gripper_joint_left", line_number=line_number
                ),
                _optional_finite_float(
                    payload["gripper_joint_right"], field="gripper_joint_right", line_number=line_number
                ),
            )
            gripper_meas = scalar_row[6]
            gripper_joint_mean = 0.5 * (scalar_row[7] + scalar_row[8])
            if math.isfinite(gripper_meas) and math.isfinite(gripper_joint_mean):
                if not math.isclose(
                    gripper_meas,
                    gripper_joint_mean,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        f"forces.jsonl line {line_number} gripper_meas does not equal "
                        "mean(gripper_joint_left, gripper_joint_right)."
                    )
            scalar_rows[frame_index] = scalar_row
    if not scalar_rows:
        raise ValueError(f"force trace contains no samples: {forces_path}")

    force_indices = tuple(sorted(scalar_rows))
    expected_indices = tuple(range(len(force_indices)))
    if force_indices != expected_indices:
        raise ValueError(
            "forces.jsonl frame indices must be contiguous from 0; "
            f"expected {expected_indices[:3]}...{expected_indices[-3:]}, "
            f"got {force_indices[:3]}...{force_indices[-3:]}."
        )
    force_index_set = set(force_indices)
    for stream_name, stream in image_streams.items():
        stream_indices = set(stream)
        if stream_indices != force_index_set:
            missing = sorted(force_index_set - stream_indices)
            extra = sorted(stream_indices - force_index_set)
            raise ValueError(
                f"{stream_name} frame indices do not match forces.jsonl; missing={missing}, extra={extra}."
            )

    pred = np.asarray([scalar_rows[index][0] for index in force_indices], dtype=np.float64)
    target_eff = np.asarray([scalar_rows[index][1] for index in force_indices], dtype=np.float64)
    meas = np.asarray([scalar_rows[index][2] for index in force_indices], dtype=np.float64)
    meas_raw = np.asarray([scalar_rows[index][3] for index in force_indices], dtype=np.float64)
    gripper_pred = np.asarray([scalar_rows[index][4] for index in force_indices], dtype=np.float64)
    gripper_cmd = np.asarray([scalar_rows[index][5] for index in force_indices], dtype=np.float64)
    gripper_meas = np.asarray([scalar_rows[index][6] for index in force_indices], dtype=np.float64)
    gripper_joint_left = np.asarray(
        [scalar_rows[index][7] for index in force_indices], dtype=np.float64
    )
    gripper_joint_right = np.asarray(
        [scalar_rows[index][8] for index in force_indices], dtype=np.float64
    )
    return Mode6Capture(
        exp_dir=resolved_exp_dir,
        prompt=prompt.strip(),
        adverb_used=adverb_used.strip(),
        frame_indices=force_indices,
        agentview_paths=tuple(image_streams["agentview"][index] for index in force_indices),
        eye_in_hand_paths=tuple(image_streams["eye"][index] for index in force_indices),
        left_tactile_paths=tuple(image_streams["gsmini_left_markers_rgb"][index] for index in force_indices),
        right_tactile_paths=tuple(image_streams["gsmini_right_markers_rgb"][index] for index in force_indices),
        scalar_schema_version=int(scalar_schema_version),
        squeeze_pred=pred,
        squeeze_target_eff=target_eff,
        squeeze_meas=meas,
        squeeze_meas_raw=meas_raw,
        gripper_pred=gripper_pred,
        gripper_cmd=gripper_cmd,
        gripper_meas=gripper_meas,
        gripper_joint_left=gripper_joint_left,
        gripper_joint_right=gripper_joint_right,
    )


def _force_axis_limits(*series: np.ndarray) -> tuple[float, float]:
    finite_parts = [values[np.isfinite(values)] for values in series]
    finite = np.concatenate(finite_parts) if finite_parts else np.asarray([])
    if finite.size == 0:
        return 0.0, 1.0
    low = min(0.0, float(np.min(finite)))
    high = max(1.0, float(np.max(finite)))
    span = max(1.0, high - low)
    return low - 0.05 * span, high + 0.10 * span


def _draw_force_segments(
    canvas: np.ndarray,
    values: np.ndarray,
    *,
    current_index: int,
    color: tuple[int, int, int],
    plot_box: tuple[int, int, int, int],
    y_limits: tuple[float, float],
) -> None:
    x0, y0, width, height = plot_box
    denominator = max(1, len(values) - 1)
    y_low, y_high = y_limits
    points: list[tuple[int, int]] = []
    for index in range(current_index + 1):
        value = values[index]
        if not np.isfinite(value):
            if len(points) >= 2:
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            points = []
            continue
        x = x0 + round(index / denominator * width)
        y = y0 + height - round((float(value) - y_low) / (y_high - y_low) * height)
        points.append((x, y))
    if len(points) >= 2:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    elif len(points) == 1:
        cv2.circle(canvas, points[0], 2, color, -1, cv2.LINE_AA)


def render_force_curve(
    squeeze_pred: Sequence[float],
    squeeze_meas: Sequence[float],
    *,
    squeeze_target_eff: Sequence[float] | None = None,
    squeeze_meas_raw: Sequence[float] | None = None,
    current_index: int,
    size: tuple[int, int],
) -> np.ndarray:
    """Render raw model target, effective target, and measured squeeze."""

    pred = np.asarray(squeeze_pred, dtype=np.float64)
    meas = np.asarray(squeeze_meas, dtype=np.float64)
    if pred.ndim != 1 or meas.ndim != 1 or len(pred) == 0 or len(pred) != len(meas):
        raise ValueError("squeeze_pred and squeeze_meas must be equal-length non-empty 1D sequences.")
    target_eff = (
        np.asarray(squeeze_target_eff, dtype=np.float64)
        if squeeze_target_eff is not None
        else np.full_like(pred, np.nan)
    )
    meas_raw = (
        np.asarray(squeeze_meas_raw, dtype=np.float64)
        if squeeze_meas_raw is not None
        else np.full_like(pred, np.nan)
    )
    if target_eff.ndim != 1 or len(target_eff) != len(pred):
        raise ValueError("squeeze_target_eff must match squeeze_pred length.")
    if meas_raw.ndim != 1 or len(meas_raw) != len(pred):
        raise ValueError("squeeze_meas_raw must match squeeze_pred length.")
    show_tracking_semantics = bool(
        np.any(np.isfinite(target_eff)) and np.any(np.isfinite(meas_raw))
    )
    displayed_meas = meas_raw if show_tracking_semantics else meas
    if not 0 <= int(current_index) < len(pred):
        raise ValueError(f"current_index must be in [0,{len(pred) - 1}], got {current_index}.")
    width, height = int(size[0]), int(size[1])
    if width < 240 or height < 180:
        raise ValueError(f"force plot size must be at least 240x180, got {size!r}.")

    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    left = max(54, round(width * 0.11))
    right = max(16, round(width * 0.03))
    top = max(50, round(height * 0.14))
    bottom = max(42, round(height * 0.14))
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_low, y_high = _force_axis_limits(pred, target_eff, displayed_meas)
    pred_color = (255, 150, 40)
    target_eff_color = (70, 70, 240)
    meas_color = (60, 150, 255)

    cv2.putText(
        canvas,
        "Squeeze force (N)",
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.38, min(0.65, width / 1000.0)),
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    legend_scale = max(0.28, min(0.42, width / 1500.0))
    legend_y = 42
    if show_tracking_semantics:
        legend_texts = ("MODEL RAW", "CTRL TARGET", "MEASURED RAW")
        legend_colors = (pred_color, target_eff_color, meas_color)
        legend_xs = (12, width // 3, 2 * width // 3)
    else:
        legend_texts = ("MODEL RAW (BLUE)", "MEASURED EMA (ORANGE)")
        legend_colors = (pred_color, meas_color)
        legend_xs = (12, width // 2)
    for legend_x, legend_text, legend_color in zip(
        legend_xs, legend_texts, legend_colors, strict=True
    ):
        cv2.line(
            canvas,
            (legend_x, legend_y - 4),
            (legend_x + 20, legend_y - 4),
            legend_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            legend_text,
            (legend_x + 25, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            legend_scale,
            legend_color,
            1,
            cv2.LINE_AA,
        )
    for grid_index in range(5):
        ratio = grid_index / 4.0
        y = top + round(ratio * plot_height)
        value = y_high - ratio * (y_high - y_low)
        cv2.line(canvas, (left, y), (left + plot_width, y), (55, 55, 55), 1)
        cv2.putText(
            canvas,
            f"{value:.1f}",
            (4, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (left, top), (left + plot_width, top + plot_height), (150, 150, 150), 1)
    plot_box = (left, top, plot_width, plot_height)
    _draw_force_segments(
        canvas,
        pred,
        current_index=int(current_index),
        color=pred_color,
        plot_box=plot_box,
        y_limits=(y_low, y_high),
    )
    _draw_force_segments(
        canvas,
        displayed_meas,
        current_index=int(current_index),
        color=meas_color,
        plot_box=plot_box,
        y_limits=(y_low, y_high),
    )
    if show_tracking_semantics:
        _draw_force_segments(
            canvas,
            target_eff,
            current_index=int(current_index),
            color=target_eff_color,
            plot_box=plot_box,
            y_limits=(y_low, y_high),
        )
    pred_value = pred[int(current_index)]
    meas_value = displayed_meas[int(current_index)]
    pred_text = "N/A" if not np.isfinite(pred_value) else f"{pred_value:.2f} N"
    meas_text = "N/A" if not np.isfinite(meas_value) else f"{meas_value:.2f} N"
    value_y = height - 12
    pred_label = "MODEL RAW" if width >= 480 else "MODEL"
    meas_label = (
        "MEASURED RAW" if show_tracking_semantics else "MEASURED EMA"
    )
    cv2.putText(
        canvas,
        f"{pred_label}: {pred_text}",
        (left, value_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        pred_color,
        1,
    )
    cv2.putText(
        canvas,
        f"{meas_label}: {meas_text}",
        (left + round(plot_width * 0.34), value_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        meas_color,
        1,
    )
    if show_tracking_semantics:
        target_value = target_eff[int(current_index)]
        target_text = (
            "N/A" if not np.isfinite(target_value) else f"{target_value:.2f} N"
        )
        cv2.putText(
            canvas,
            f"CTRL TARGET: {target_text}",
            (left + round(plot_width * 0.68), value_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            target_eff_color,
            1,
        )
    else:
        cv2.putText(
            canvas,
            f"frame {int(current_index)}/{len(pred) - 1}",
            (left + round(plot_width * 0.78), value_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (190, 190, 190),
            1,
        )
    return canvas


def _gripper_axis_limits(
    command_mm: np.ndarray,
    measured_mm: np.ndarray,
    predicted_mm: np.ndarray | None = None,
) -> tuple[float, float]:
    series = [command_mm, measured_mm]
    if predicted_mm is not None:
        series.append(predicted_mm)
    finite = np.concatenate(tuple(values[np.isfinite(values)] for values in series))
    if finite.size == 0:
        return -2.0, 42.0
    low = min(0.0, float(np.min(finite)))
    high = max(40.0, float(np.max(finite)))
    span = max(1.0, high - low)
    padded_low = low - 0.05 * span if low < 0.0 else 0.0
    return padded_low, high + 0.05 * span


def render_gripper_curve(
    gripper_cmd_m: Sequence[float],
    gripper_meas_m: Sequence[float],
    *,
    gripper_pred_m: Sequence[float] | None = None,
    current_index: int,
    size: tuple[int, int],
) -> np.ndarray:
    """Render model, controller, and measured per-finger joint positions."""

    command_mm = np.asarray(gripper_cmd_m, dtype=np.float64) * 1000.0
    measured_mm = np.asarray(gripper_meas_m, dtype=np.float64) * 1000.0
    predicted_mm = (
        None
        if gripper_pred_m is None
        else np.asarray(gripper_pred_m, dtype=np.float64) * 1000.0
    )
    if (
        command_mm.ndim != 1
        or measured_mm.ndim != 1
        or len(command_mm) == 0
        or len(command_mm) != len(measured_mm)
    ):
        raise ValueError("gripper_cmd_m and gripper_meas_m must be equal-length non-empty 1D sequences.")
    if predicted_mm is not None and (
        predicted_mm.ndim != 1 or len(predicted_mm) != len(command_mm)
    ):
        raise ValueError(
            "gripper_pred_m must be a 1D sequence with the same length as "
            "gripper_cmd_m when provided."
        )
    if not 0 <= int(current_index) < len(command_mm):
        raise ValueError(f"current_index must be in [0,{len(command_mm) - 1}], got {current_index}.")
    width, height = int(size[0]), int(size[1])
    if width < 240 or height < 180:
        raise ValueError(f"gripper plot size must be at least 240x180, got {size!r}.")

    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    left = max(54, round(width * 0.11))
    right = max(16, round(width * 0.03))
    top = max(50, round(height * 0.14))
    bottom = max(42, round(height * 0.14))
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_low, y_high = _gripper_axis_limits(command_mm, measured_mm, predicted_mm)
    predicted_color = (255, 220, 60)
    command_color = (80, 220, 80)
    measured_color = (220, 80, 220)

    cv2.putText(
        canvas,
        "Gripper per-finger joint position (mm)",
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.38, min(0.65, width / 1000.0)),
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    legend_scale = max(0.28, min(0.42, width / 1500.0))
    legend_y = 42
    if predicted_mm is None:
        legend_texts = ("CONTROLLER d_cmd (GREEN)", "MEASURED d_actual (MAGENTA)")
        legend_colors = (command_color, measured_color)
        legend_xs = (12, width // 2)
    else:
        legend_texts = (
            "MODEL d_pred (CYAN)",
            "CONTROLLER d_cmd (GREEN)",
            "MEASURED d_actual (MAGENTA)",
        )
        legend_colors = (predicted_color, command_color, measured_color)
        legend_xs = (12, width // 3, 2 * width // 3)
    for legend_x, legend_text, legend_color in zip(
        legend_xs, legend_texts, legend_colors, strict=True
    ):
        cv2.line(
            canvas,
            (legend_x, legend_y - 4),
            (legend_x + 20, legend_y - 4),
            legend_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            legend_text,
            (legend_x + 25, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            legend_scale,
            legend_color,
            1,
            cv2.LINE_AA,
        )

    for grid_index in range(5):
        ratio = grid_index / 4.0
        y = top + round(ratio * plot_height)
        value = y_high - ratio * (y_high - y_low)
        cv2.line(canvas, (left, y), (left + plot_width, y), (55, 55, 55), 1)
        cv2.putText(
            canvas,
            f"{value:.1f}",
            (4, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (left, top), (left + plot_width, top + plot_height), (150, 150, 150), 1)
    plot_box = (left, top, plot_width, plot_height)
    if predicted_mm is not None:
        _draw_force_segments(
            canvas,
            predicted_mm,
            current_index=int(current_index),
            color=predicted_color,
            plot_box=plot_box,
            y_limits=(y_low, y_high),
        )
    _draw_force_segments(
        canvas,
        command_mm,
        current_index=int(current_index),
        color=command_color,
        plot_box=plot_box,
        y_limits=(y_low, y_high),
    )
    _draw_force_segments(
        canvas,
        measured_mm,
        current_index=int(current_index),
        color=measured_color,
        plot_box=plot_box,
        y_limits=(y_low, y_high),
    )

    predicted_value = (
        float("nan") if predicted_mm is None else predicted_mm[int(current_index)]
    )
    command_value = command_mm[int(current_index)]
    measured_value = measured_mm[int(current_index)]
    predicted_text = "N/A" if not np.isfinite(predicted_value) else f"{predicted_value:.2f} mm"
    command_text = "N/A" if not np.isfinite(command_value) else f"{command_value:.2f} mm"
    measured_text = "N/A" if not np.isfinite(measured_value) else f"{measured_value:.2f} mm"
    value_y = height - 12
    value_xs = (left, left + round(plot_width * 0.34), left + round(plot_width * 0.68))
    if predicted_mm is not None:
        cv2.putText(
            canvas,
            f"d_pred: {predicted_text}",
            (value_xs[0], value_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            predicted_color,
            1,
        )
    cv2.putText(
        canvas,
        f"d_cmd: {command_text}",
        (value_xs[1] if predicted_mm is not None else left, value_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38 if predicted_mm is not None else 0.42,
        command_color,
        1,
    )
    cv2.putText(
        canvas,
        f"d_actual: {measured_text}",
        (
            value_xs[2]
            if predicted_mm is not None
            else left + round(plot_width * 0.44),
            value_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38 if predicted_mm is not None else 0.42,
        measured_color,
        1,
    )
    return canvas


def render_force_gripper_dashboard(
    squeeze_pred: Sequence[float],
    squeeze_meas: Sequence[float],
    gripper_cmd_m: Sequence[float],
    gripper_meas_m: Sequence[float],
    *,
    squeeze_target_eff: Sequence[float] | None = None,
    squeeze_meas_raw: Sequence[float] | None = None,
    gripper_pred_m: Sequence[float] | None = None,
    current_index: int,
    size: tuple[int, int],
) -> np.ndarray:
    """Stack synchronized squeeze and gripper-position curves in one panel."""

    width, height = int(size[0]), int(size[1])
    if width < 240 or height < 360:
        raise ValueError(f"dashboard size must be at least 240x360, got {size!r}.")
    top_height = height // 2
    bottom_height = height - top_height
    force_plot = render_force_curve(
        squeeze_pred,
        squeeze_meas,
        squeeze_target_eff=squeeze_target_eff,
        squeeze_meas_raw=squeeze_meas_raw,
        current_index=current_index,
        size=(width, top_height),
    )
    gripper_plot = render_gripper_curve(
        gripper_cmd_m,
        gripper_meas_m,
        gripper_pred_m=gripper_pred_m,
        current_index=current_index,
        size=(width, bottom_height),
    )
    dashboard = np.vstack((force_plot, gripper_plot))
    cv2.line(dashboard, (0, top_height), (width - 1, top_height), (120, 120, 120), 1)
    return dashboard


def _place_labeled_panel(
    canvas: np.ndarray,
    frame: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    label: str,
) -> None:
    x, y, width, height = box
    panel = letterbox_bgr(frame, (width, height))
    canvas[y : y + height, x : x + width] = panel
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (90, 90, 90), 1)
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + 28), (0, 0, 0), -1)
    cv2.putText(canvas, label, (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1)


def _draw_prompt(canvas: np.ndarray, prompt: str, adverb_used: str, *, y: int) -> None:
    text = str(prompt).strip()
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(1.0, max(0.55, canvas.shape[1] / 2100.0))
    thickness = 2
    while cv2.getTextSize(text, font, scale, thickness)[0][0] > canvas.shape[1] - 60 and scale > 0.45:
        scale -= 0.05
    x = 30
    adverb = str(adverb_used).strip()
    if not adverb:
        cv2.putText(canvas, text, (x, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)
        return
    match = re.search(re.escape(adverb), text, flags=re.IGNORECASE)
    if match is None:
        cv2.putText(canvas, text, (x, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)
        return
    before, highlighted, after = text[: match.start()], text[match.start() : match.end()], text[match.end() :]
    cv2.putText(canvas, before, (x, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)
    x += cv2.getTextSize(before, font, scale, thickness)[0][0]
    cv2.putText(canvas, highlighted, (x, y), font, scale, (60, 80, 255), thickness, cv2.LINE_AA)
    x += cv2.getTextSize(highlighted, font, scale, thickness)[0][0]
    cv2.putText(canvas, after, (x, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def compose_five_panel_frame(
    *,
    agentview_bgr: np.ndarray,
    eye_in_hand_bgr: np.ndarray,
    left_tactile_bgr: np.ndarray,
    right_tactile_bgr: np.ndarray,
    force_plot_bgr: np.ndarray,
    prompt: str,
    adverb_used: str = "",
    size: tuple[int, int] = (1920, 1080),
) -> np.ndarray:
    """Compose two cameras, two tactile views, and a force chart into one frame."""

    width, height = int(size[0]), int(size[1])
    if width < 640 or height < 360 or width % 2 or height % 2:
        raise ValueError(f"five-panel size must be even and at least 640x360, got {size!r}.")
    for name, frame in (
        ("agentview_bgr", agentview_bgr),
        ("eye_in_hand_bgr", eye_in_hand_bgr),
        ("left_tactile_bgr", left_tactile_bgr),
        ("right_tactile_bgr", right_tactile_bgr),
        ("force_plot_bgr", force_plot_bgr),
    ):
        _validate_bgr_image(name, frame)

    canvas = np.full((height, width, 3), 10, dtype=np.uint8)
    margin = max(8, round(width * 0.015))
    gap = max(8, round(width * 0.012))
    header_height = max(38, round(height * 0.075))
    _draw_prompt(canvas, prompt, adverb_used, y=max(28, header_height - 18))

    top_y = header_height + gap
    top_height = max(120, round(height * 0.44))
    top_width = min(top_height, round(width * 0.29))
    top_total_width = 2 * top_width + gap
    top_x = (width - top_total_width) // 2
    _place_labeled_panel(
        canvas,
        agentview_bgr,
        box=(top_x, top_y, top_width, top_height),
        label="Agentview",
    )
    _place_labeled_panel(
        canvas,
        eye_in_hand_bgr,
        box=(top_x + top_width + gap, top_y, top_width, top_height),
        label="Eye-in-hand",
    )

    bottom_y = top_y + top_height + gap
    bottom_height = height - bottom_y - margin
    available_width = width - 2 * margin - 2 * gap
    tactile_width = round(available_width * 0.27)
    chart_width = available_width - 2 * tactile_width
    _place_labeled_panel(
        canvas,
        left_tactile_bgr,
        box=(margin, bottom_y, tactile_width, bottom_height),
        label="Left GelSight markers",
    )
    _place_labeled_panel(
        canvas,
        right_tactile_bgr,
        box=(margin + tactile_width + gap, bottom_y, tactile_width, bottom_height),
        label="Right GelSight markers",
    )
    chart_x = margin + 2 * tactile_width + 2 * gap
    chart = letterbox_bgr(force_plot_bgr, (chart_width, bottom_height))
    canvas[bottom_y : bottom_y + bottom_height, chart_x : chart_x + chart_width] = chart
    cv2.rectangle(
        canvas,
        (chart_x, bottom_y),
        (chart_x + chart_width - 1, bottom_y + bottom_height - 1),
        (90, 90, 90),
        1,
    )
    return canvas


class H264FrameWriter:
    """Stream fixed-size BGR frames to FFmpeg without intermediate images."""

    def __init__(
        self,
        output_path: Path,
        *,
        fps: float,
        size: tuple[int, int],
        overwrite: bool = False,
    ) -> None:
        if not math.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {fps!r}.")
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(f"H.264 frame size must be positive and even, got {size!r}.")
        self.output_path = Path(output_path).expanduser().resolve()
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing video {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.output_path.with_name(f".{self.output_path.stem}.pending.mp4")
        if self.pending_path.exists():
            self.pending_path.unlink()
        self.fps = float(fps)
        self.size = (width, height)
        self.overwrite = bool(overwrite)
        self.frame_count = 0
        command = [
            "/usr/bin/ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(self.pending_path),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        image = _validate_bgr_image("video frame", frame)
        width, height = self.size
        if image.shape[:2] != (height, width):
            raise ValueError(f"video frame must be {width}x{height}, got {image.shape[1]}x{image.shape[0]}.")
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable.")
        self._process.stdin.write(np.ascontiguousarray(image).tobytes())
        self.frame_count += 1

    def finalize(self, *, expected_frames: int) -> Path:
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
        return_code = self._process.wait()
        if return_code != 0:
            if self.pending_path.exists():
                self.pending_path.unlink()
            raise RuntimeError(f"FFmpeg exited with code {return_code}: {stderr.strip()}")
        if self.frame_count != int(expected_frames):
            if self.pending_path.exists():
                self.pending_path.unlink()
            raise RuntimeError(f"encoded frames={self.frame_count} do not match expected_frames={expected_frames}.")
        self.pending_path.replace(self.output_path)
        return self.output_path

    def abort(self) -> None:
        """Terminate an incomplete encoder and remove only its pending file."""

        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        if self.pending_path.exists():
            self.pending_path.unlink()


def compose_side_by_side_frame(
    camera_frames: Iterable[tuple[str, np.ndarray]],
    overlay_lines: Iterable[str] = (),
) -> np.ndarray:
    """Return one BGR frame with RGB camera views placed side by side."""

    normalized: list[tuple[str, np.ndarray]] = []
    for name, frame in camera_frames:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"camera frame {name!r} must have shape (H,W,3), got {image.shape}."
            )
        if image.dtype != np.uint8:
            raise TypeError(
                f"camera frame {name!r} must be uint8 RGB, got {image.dtype}."
            )
        normalized.append((str(name), image))
    if not normalized:
        raise ValueError("at least one camera frame is required.")

    target_height = max(frame.shape[0] for _, frame in normalized)
    views: list[np.ndarray] = []
    for name, frame in normalized:
        if frame.shape[0] != target_height:
            width = max(1, round(frame.shape[1] * target_height / frame.shape[0]))
            frame = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr,
            name,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        views.append(bgr)

    canvas = np.concatenate(views, axis=1)
    for index, line in enumerate(overlay_lines):
        cv2.putText(
            canvas,
            str(line),
            (8, 46 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


class EpisodeVideoWriter:
    """Lazily write one MP4 and return an auditable completion descriptor."""

    def __init__(self, output_dir: Path, episode_index: int, fps: float) -> None:
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {fps!r}.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_index = int(episode_index)
        self.fps = float(fps)
        self.pending_path = self.output_dir / f"episode_{self.episode_index:03d}_pending.mp4"
        self._writer: cv2.VideoWriter | None = None
        self._frame_size: tuple[int, int] | None = None
        self.frame_count = 0
        self.error: str | None = None

    def write(
        self,
        camera_frames: Iterable[tuple[str, np.ndarray]],
        overlay_lines: Iterable[str] = (),
    ) -> None:
        if self.error is not None:
            return
        try:
            frame = compose_side_by_side_frame(camera_frames, overlay_lines)
            height, width = frame.shape[:2]
            frame_size = (width, height)
            if self._writer is None:
                self._frame_size = frame_size
                self._writer = cv2.VideoWriter(
                    str(self.pending_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    frame_size,
                )
                if not self._writer.isOpened():
                    raise RuntimeError(
                        f"failed to open MP4 writer at {self.pending_path}."
                    )
            elif frame_size != self._frame_size:
                raise ValueError(
                    f"video frame size changed from {self._frame_size} to {frame_size}."
                )
            self._writer.write(frame)
            self.frame_count += 1
        except Exception as exc:
            self.error = str(exc)

    def finalize(
        self,
        *,
        success: bool,
        end_reason: str,
        expected_frames: int,
    ) -> dict:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        outcome = "success" if success else f"failure_{_safe_name(end_reason)}"
        final_path = self.output_dir / (
            f"episode_{self.episode_index:03d}_{outcome}.mp4"
        )
        if self.error is None:
            if self.frame_count == 0 or not self.pending_path.is_file():
                self.error = "video contains no frames"
            elif self.frame_count != int(expected_frames):
                self.error = (
                    f"video frames={self.frame_count} do not match "
                    f"env_steps={expected_frames}"
                )
            elif final_path.exists():
                self.error = f"refusing to overwrite existing video {final_path}"
            else:
                self.pending_path.rename(final_path)

        completed = self.error is None and final_path.is_file()
        path = final_path if completed else self.pending_path
        return {
            "enabled": True,
            "status": "complete" if completed else "partial",
            "path": str(path.resolve()) if path.exists() else None,
            "frames": int(self.frame_count),
            "fps": self.fps,
            "error": self.error,
        }
