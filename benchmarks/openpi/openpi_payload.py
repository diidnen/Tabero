"""CPU-testable OpenPI camera payload construction and inference wiring."""

import cv2
import numpy as np

_PAYLOAD_ALIAS_KEYS = (
    "image",
    "wrist_image",
    "state",
    "prompt",
    "gripper_force",
    "tactile_image",
    "tactile_gripper_force",
    "tactile_marker_motion",
    "dsrl_raw_image",
    "dsrl_raw_wrist_image",
    "observation/image",
    "observation/wrist_image",
    "observation/state",
    "observation/gripper_force",
    "observation/tactile_image",
    "observation/tactile_gripper_force",
    "observation/tactile_marker_motion",
)


def _to_numpy(frame):
    return frame.detach().cpu().numpy() if hasattr(frame, "detach") else frame


def _to_uint8_rgb(image) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype in (np.float32, np.float64):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _resize_frames_with_padding(frames, target_image_size: tuple[int, int, int]) -> np.ndarray:
    """Match the existing OpenPI camera resize/padding protocol without Isaac imports."""
    frames = _to_numpy(frames)
    top_padding = (frames.shape[2] - frames.shape[1]) // 2
    frames = np.pad(
        frames,
        pad_width=((0, 0), (top_padding, top_padding), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    if frames.shape[1:] != target_image_size:
        frames = np.stack([cv2.resize(frame, target_image_size[:2]) for frame in frames])
    return frames


_DSRL_RAW_CAMERAS = {
    "agentview_cam": "dsrl_raw_image",
    "eye_in_hand_cam": "dsrl_raw_wrist_image",
}


def _capture_dsrl_raw_images(camera_frames):
    raw_images = {}
    for camera_name, frame in camera_frames:
        payload_key = _DSRL_RAW_CAMERAS.get(camera_name)
        if payload_key is None:
            continue
        if payload_key in raw_images:
            raise ValueError(f"send_dsrl_raw_image received duplicate {camera_name} frames.")
        raw_batch = _to_numpy(frame)
        if isinstance(raw_batch, np.ndarray) and raw_batch.shape[:1] == (1,):
            raw_image = raw_batch[0]
        else:
            raw_image = raw_batch
        # Official Libero cameras render at 512 square. Produce each DSRL-native
        # 256 frame independently of the standard OpenPI 224 preprocessing.
        if isinstance(raw_image, np.ndarray) and raw_image.shape == (512, 512, 3):
            raw_image = cv2.resize(
                raw_image, (256, 256), interpolation=cv2.INTER_AREA
            )
        raw_images[payload_key] = raw_image
    return raw_images


def _validate_dsrl_raw_image(raw_image, *, payload_key: str) -> None:
    if not isinstance(raw_image, np.ndarray):
        raise TypeError(
            f"send_dsrl_raw_image requires {payload_key} to be a numpy.ndarray, "
            f"but got {type(raw_image).__name__}."
        )
    if raw_image.dtype != np.uint8:
        raise TypeError(
            f"send_dsrl_raw_image requires {payload_key} to have dtype uint8, "
            f"but got {raw_image.dtype}."
        )
    if raw_image.shape != (256, 256, 3):
        raise ValueError(
            f"send_dsrl_raw_image requires {payload_key} to have shape (256, 256, 3), "
            f"but got {raw_image.shape}."
        )


def infer_openpi_step(
    client,
    *,
    camera_frames,
    target_image_size: tuple[int, int, int],
    base_payload: dict,
    send_dsrl_raw_image: bool,
    before_infer=None,
):
    """Build one production payload and pass that exact object to ``client.infer``."""
    camera_frames = tuple(camera_frames)
    raw_images = (
        _capture_dsrl_raw_images(camera_frames) if send_dsrl_raw_image else {}
    )
    if send_dsrl_raw_image:
        for payload_key in _DSRL_RAW_CAMERAS.values():
            _validate_dsrl_raw_image(
                raw_images.get(payload_key), payload_key=payload_key
            )

    resized_frames = tuple(_resize_frames_with_padding(frame, target_image_size) for _, frame in camera_frames)
    if len(resized_frames) < 2:
        raise ValueError("OpenPI inference requires at least two camera frames.")

    image = _to_uint8_rgb(np.squeeze(resized_frames[0], axis=0))
    wrist_image = _to_uint8_rgb(np.squeeze(resized_frames[1], axis=0))
    original_payload = dict(base_payload)
    element = base_payload
    element.clear()
    element["image"] = image
    element["wrist_image"] = wrist_image
    if "state" in original_payload:
        element["state"] = original_payload["state"]
    element["observation/image"] = image
    element["observation/wrist_image"] = wrist_image
    for key in ("observation/state", "prompt"):
        if key in original_payload:
            element[key] = original_payload[key]
    if send_dsrl_raw_image:
        element.update(raw_images)
    reserved_keys = {
        "image",
        "wrist_image",
        "state",
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
        "dsrl_raw_image",
        "dsrl_raw_wrist_image",
    }
    for key, value in original_payload.items():
        if key not in reserved_keys:
            element[key] = value

    for key in _PAYLOAD_ALIAS_KEYS:
        if key in element:
            element[key.encode("utf-8")] = element[key]

    if before_infer is not None:
        before_infer(resized_frames)
    return client.infer(element), resized_frames
