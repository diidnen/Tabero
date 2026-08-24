import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "benchmarks/openpi/openpi_inference_client.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeClient:
    def __init__(self, events=None):
        self.payload = None
        self.response = {"actions": np.zeros((2, 7), dtype=np.float32)}
        self.events = events

    def infer(self, payload):
        if self.events is not None:
            self.events.append("infer")
        self.payload = payload
        return self.response


def _infer_openpi_step():
    module = importlib.import_module("benchmarks.openpi.openpi_payload")
    return module.infer_openpi_step


def _base_payload():
    state = np.arange(7, dtype=np.float32)
    return {
        "state": state,
        "observation/state": state,
        "prompt": "pick up the object",
    }


def test_production_step_sends_224_images_and_two_raw_views_to_same_infer_payload():
    infer_openpi_step = _infer_openpi_step()
    events = []
    client = FakeClient(events)
    wrist = np.full((1, 512, 512, 3), 7, dtype=np.uint8)
    agentview = np.full((1, 512, 512, 3), 13, dtype=np.uint8)
    base_payload = _base_payload()

    def save_debug_frames(_resized_frames):
        events.append("debug")

    response, resized_frames = infer_openpi_step(
        client,
        camera_frames=(("eye_in_hand_cam", wrist), ("agentview_cam", agentview)),
        target_image_size=(224, 224, 3),
        base_payload=base_payload,
        send_dsrl_raw_image=True,
        before_infer=save_debug_frames,
    )

    assert response is client.response
    assert client.payload is base_payload
    assert events == ["debug", "infer"]
    assert client.payload["image"].shape == (224, 224, 3)
    assert client.payload["wrist_image"].shape == (224, 224, 3)
    assert client.payload["dsrl_raw_image"].shape == (256, 256, 3)
    assert client.payload["dsrl_raw_image"].dtype == np.uint8
    assert np.all(client.payload["dsrl_raw_image"] == 13)
    assert client.payload["dsrl_raw_wrist_image"].shape == (256, 256, 3)
    assert client.payload["dsrl_raw_wrist_image"].dtype == np.uint8
    assert np.all(client.payload["dsrl_raw_wrist_image"] == 7)
    assert client.payload[b"dsrl_raw_image"] is client.payload["dsrl_raw_image"]
    assert (
        client.payload[b"dsrl_raw_wrist_image"]
        is client.payload["dsrl_raw_wrist_image"]
    )
    assert client.payload[b"image"] is client.payload["image"]
    assert [frame.shape for frame in resized_frames] == [(1, 224, 224, 3), (1, 224, 224, 3)]


def test_production_step_default_payload_has_no_dsrl_raw_image():
    infer_openpi_step = _infer_openpi_step()
    client = FakeClient()
    agentview = np.full((1, 512, 512, 3), 13, dtype=np.uint8)
    wrist = np.full((1, 512, 512, 3), 7, dtype=np.uint8)

    base_payload = _base_payload()
    infer_openpi_step(
        client,
        camera_frames=(("agentview_cam", agentview), ("eye_in_hand_cam", wrist)),
        target_image_size=(224, 224, 3),
        base_payload=base_payload,
        send_dsrl_raw_image=False,
    )

    assert client.payload is base_payload
    string_keys = {key for key in client.payload if isinstance(key, str)}
    assert string_keys == {
        "image",
        "wrist_image",
        "state",
        "prompt",
        "observation/image",
        "observation/wrist_image",
        "observation/state",
    }
    assert {key for key in client.payload if isinstance(key, bytes)} == {key.encode("utf-8") for key in string_keys}
    assert client.payload["image"].shape == (224, 224, 3)
    assert client.payload["image"].dtype == np.uint8
    assert np.all(client.payload["image"] == 13)
    assert np.all(client.payload["wrist_image"] == 7)
    assert client.payload["observation/image"] is client.payload["image"]
    assert client.payload[b"observation/wrist_image"] is client.payload["wrist_image"]
    assert "dsrl_raw_image" not in client.payload
    assert b"dsrl_raw_image" not in client.payload
    assert "dsrl_raw_wrist_image" not in client.payload
    assert b"dsrl_raw_wrist_image" not in client.payload


@pytest.mark.parametrize(
    ("agentview", "message"),
    [
        (None, "numpy.ndarray"),
        (np.zeros((1, 256, 256, 3), dtype=np.float32), "dtype uint8"),
        (np.zeros((1, 224, 224, 3), dtype=np.uint8), r"shape \(256, 256, 3\)"),
    ],
)
def test_production_step_opt_in_strictly_validates_raw_agentview(agentview, message):
    infer_openpi_step = _infer_openpi_step()
    client = FakeClient()
    camera_frames = [] if agentview is None else [("agentview_cam", agentview)]
    camera_frames.append(("eye_in_hand_cam", np.zeros((1, 512, 512, 3), dtype=np.uint8)))

    with pytest.raises((TypeError, ValueError), match=message):
        infer_openpi_step(
            client,
            camera_frames=camera_frames,
            target_image_size=(224, 224, 3),
            base_payload=_base_payload(),
            send_dsrl_raw_image=True,
        )

    assert client.payload is None


@pytest.mark.parametrize(
    ("wrist", "message"),
    [
        (None, "dsrl_raw_wrist_image.*numpy.ndarray"),
        (
            np.zeros((1, 256, 256, 3), dtype=np.float32),
            "dsrl_raw_wrist_image.*dtype uint8",
        ),
        (
            np.zeros((1, 224, 224, 3), dtype=np.uint8),
            r"dsrl_raw_wrist_image.*shape \(256, 256, 3\)",
        ),
    ],
)
def test_production_step_opt_in_strictly_validates_raw_wrist(wrist, message):
    infer_openpi_step = _infer_openpi_step()
    client = FakeClient()
    camera_frames = [
        ("agentview_cam", np.zeros((1, 512, 512, 3), dtype=np.uint8))
    ]
    if wrist is not None:
        camera_frames.append(("eye_in_hand_cam", wrist))

    with pytest.raises((TypeError, ValueError), match=message):
        infer_openpi_step(
            client,
            camera_frames=camera_frames,
            target_image_size=(224, 224, 3),
            base_payload=_base_payload(),
            send_dsrl_raw_image=True,
        )

    assert client.payload is None


def test_production_step_raw_views_accept_native_256_without_resizing():
    infer_openpi_step = _infer_openpi_step()
    client = FakeClient()
    main = np.full((1, 256, 256, 3), 19, dtype=np.uint8)
    wrist = np.full((1, 256, 256, 3), 23, dtype=np.uint8)

    infer_openpi_step(
        client,
        camera_frames=(("agentview_cam", main), ("eye_in_hand_cam", wrist)),
        target_image_size=(224, 224, 3),
        base_payload=_base_payload(),
        send_dsrl_raw_image=True,
    )

    assert np.all(client.payload["dsrl_raw_image"] == 19)
    assert np.all(client.payload["dsrl_raw_wrist_image"] == 23)


@pytest.mark.parametrize("camera_name", ["agentview_cam", "eye_in_hand_cam"])
def test_production_step_rejects_duplicate_dsrl_camera(camera_name):
    infer_openpi_step = _infer_openpi_step()
    client = FakeClient()
    frame = np.zeros((1, 512, 512, 3), dtype=np.uint8)
    other_name = (
        "eye_in_hand_cam" if camera_name == "agentview_cam" else "agentview_cam"
    )

    with pytest.raises(ValueError, match=f"duplicate {camera_name}"):
        infer_openpi_step(
            client,
            camera_frames=((camera_name, frame), (camera_name, frame), (other_name, frame)),
            target_image_size=(224, 224, 3),
            base_payload=_base_payload(),
            send_dsrl_raw_image=True,
        )

    assert client.payload is None


def test_openpi_client_declares_dsrl_raw_image_opt_in_default_false():
    module = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "OpenpiClientArguments":
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and item.target.id == "send_dsrl_raw_image"
                ):
                    assert ast.literal_eval(item.value) is False
                    return

    pytest.fail("OpenpiClientArguments.send_dsrl_raw_image is missing")
