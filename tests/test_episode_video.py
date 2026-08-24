import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.episode_video import (  # noqa: E402
    EpisodeVideoWriter,
    _gripper_axis_limits,
    compose_five_panel_frame,
    compose_side_by_side_frame,
    letterbox_bgr,
    render_force_curve,
    render_force_gripper_dashboard,
    render_gripper_curve,
)


def test_compose_side_by_side_frame_accepts_different_camera_heights():
    main = np.zeros((48, 64, 3), dtype=np.uint8)
    wrist = np.full((24, 32, 3), 127, dtype=np.uint8)

    frame = compose_side_by_side_frame(
        [("agentview", main), ("wrist", wrist)], ["target_single=13 N"]
    )

    assert frame.dtype == np.uint8
    assert frame.shape == (48, 128, 3)


def test_episode_video_writer_records_one_frame_per_env_step(tmp_path):
    writer = EpisodeVideoWriter(tmp_path, episode_index=0, fps=20.0)
    main = np.zeros((48, 64, 3), dtype=np.uint8)
    wrist = np.full((48, 64, 3), 127, dtype=np.uint8)

    for step in range(4):
        writer.write(
            [("agentview", main), ("wrist", wrist)],
            [f"step={step}", "latched=1"],
        )

    descriptor = writer.finalize(
        success=True,
        end_reason="success",
        expected_frames=4,
    )

    assert descriptor["status"] == "complete"
    assert descriptor["frames"] == 4
    assert descriptor["error"] is None
    video_path = Path(descriptor["path"])
    assert video_path.name == "episode_000_success.mp4"

    capture = cv2.VideoCapture(str(video_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        assert capture.get(cv2.CAP_PROP_FPS) == 20.0
    finally:
        capture.release()


def test_episode_video_writer_marks_frame_count_mismatch_partial(tmp_path):
    writer = EpisodeVideoWriter(tmp_path, episode_index=2, fps=20.0)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    writer.write([("agentview", frame)])

    descriptor = writer.finalize(
        success=False,
        end_reason="max_inference_steps",
        expected_frames=2,
    )

    assert descriptor["status"] == "partial"
    assert "do not match" in descriptor["error"]


def test_letterbox_bgr_preserves_aspect_ratio():
    frame = np.full((20, 40, 3), 127, dtype=np.uint8)

    result = letterbox_bgr(frame, (40, 40))

    assert result.shape == (40, 40, 3)
    assert np.all(result[:10] == 0)
    assert np.all(result[10:30] == 127)
    assert np.all(result[30:] == 0)


def test_force_curve_only_reveals_samples_through_current_frame():
    pred = [0.0, 4.0, 8.0, 12.0]
    meas = [0.0, 2.0, 4.0, 6.0]

    first = render_force_curve(pred, meas, current_index=0, size=(480, 240))
    last = render_force_curve(pred, meas, current_index=3, size=(480, 240))

    pred_color = np.asarray([255, 150, 40], dtype=np.uint8)
    first_pred_pixels = np.count_nonzero(np.all(first == pred_color, axis=2))
    last_pred_pixels = np.count_nonzero(np.all(last == pred_color, axis=2))
    assert last_pred_pixels > first_pred_pixels


def test_force_curve_has_explicit_predicted_and_measured_color_legend():
    chart = render_force_curve([0.0, 4.0], [0.0, 2.0], current_index=1, size=(480, 240))

    pred_color = np.asarray([255, 150, 40], dtype=np.uint8)
    meas_color = np.asarray([60, 150, 255], dtype=np.uint8)
    pred_swatch = chart[34:43, 10:35]
    meas_swatch = chart[34:43, 238:265]
    assert np.any(np.all(pred_swatch == pred_color, axis=2))
    assert np.any(np.all(meas_swatch == meas_color, axis=2))


def test_force_curve_distinguishes_model_effective_target_and_raw_measurement():
    chart = render_force_curve(
        [10.0, 20.0],
        [2.0, 3.0],
        squeeze_target_eff=[19.0, 38.0],
        squeeze_meas_raw=[4.0, 5.0],
        current_index=1,
        size=(720, 240),
    )

    model_color = np.asarray([255, 150, 40], dtype=np.uint8)
    effective_color = np.asarray([70, 70, 240], dtype=np.uint8)
    measured_color = np.asarray([60, 150, 255], dtype=np.uint8)
    assert np.any(np.all(chart == model_color, axis=2))
    assert np.any(np.all(chart == effective_color, axis=2))
    assert np.any(np.all(chart == measured_color, axis=2))


def test_gripper_curve_only_reveals_samples_through_current_frame():
    command = [0.04, 0.03, 0.02, 0.01]
    measured = [0.04, 0.035, 0.025, 0.015]

    first = render_gripper_curve(command, measured, current_index=0, size=(480, 240))
    last = render_gripper_curve(command, measured, current_index=3, size=(480, 240))

    command_color = np.asarray([80, 220, 80], dtype=np.uint8)
    first_command_pixels = np.count_nonzero(np.all(first == command_color, axis=2))
    last_command_pixels = np.count_nonzero(np.all(last == command_color, axis=2))
    assert last_command_pixels > first_command_pixels


def test_gripper_curve_plots_model_command_and_measured_positions():
    chart = render_gripper_curve(
        [0.04, 0.02],
        [0.039, 0.025],
        gripper_pred_m=[0.041, 0.03],
        current_index=1,
        size=(720, 240),
    )

    predicted_color = np.asarray([255, 220, 60], dtype=np.uint8)
    command_color = np.asarray([80, 220, 80], dtype=np.uint8)
    measured_color = np.asarray([220, 80, 220], dtype=np.uint8)
    assert np.any(np.all(chart == predicted_color, axis=2))
    assert np.any(np.all(chart == command_color, axis=2))
    assert np.any(np.all(chart == measured_color, axis=2))


def test_gripper_axis_does_not_invent_negative_travel_for_nonnegative_data():
    low, high = _gripper_axis_limits(
        np.asarray([0.0, 20.0, 40.0]),
        np.asarray([1.0, 10.0, 30.0]),
    )

    assert low == 0.0
    assert high > 40.0


def test_force_gripper_dashboard_has_fixed_shape():
    dashboard = render_force_gripper_dashboard(
        [1.0, 2.0],
        [0.5, 1.5],
        [0.04, 0.02],
        [0.039, 0.025],
        gripper_pred_m=[0.041, 0.03],
        current_index=1,
        size=(480, 360),
    )

    assert dashboard.dtype == np.uint8
    assert dashboard.shape == (360, 480, 3)


def test_compose_five_panel_frame_has_fixed_output_shape():
    square = np.zeros((64, 64, 3), dtype=np.uint8)
    tactile = np.full((48, 64, 3), 127, dtype=np.uint8)
    force_plot = render_force_curve([1.0, 2.0], [0.5, 1.5], current_index=1, size=(320, 180))

    composed = compose_five_panel_frame(
        agentview_bgr=square,
        eye_in_hand_bgr=square,
        left_tactile_bgr=tactile,
        right_tactile_bgr=tactile,
        force_plot_bgr=force_plot,
        prompt="pick up the alphabet soup and place it in the basket",
        size=(640, 360),
    )

    assert composed.dtype == np.uint8
    assert composed.shape == (360, 640, 3)
