import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.episode_lift import EpisodeLiftTracker


def test_lift_requires_latched_sustained_height_increase():
    tracker = EpisodeLiftTracker(
        target_object="alphabet_soup_1",
        initial_height_m=0.90,
        height_threshold_m=0.03,
        hold_steps_required=3,
    )

    assert not tracker.update(
        step_index=0, object_height_m=0.94, contact_latched=False
    )["target_object_above_lift_threshold"]
    assert tracker.update(
        step_index=1, object_height_m=0.931, contact_latched=True
    )["target_object_lift_hold_count"] == 1
    assert tracker.update(
        step_index=2, object_height_m=0.935, contact_latched=True
    )["target_object_lift_hold_count"] == 2
    state = tracker.update(
        step_index=3, object_height_m=0.94, contact_latched=True
    )

    assert state["target_object_lifted"]
    assert tracker.summary()["first_above_threshold_step"] == 1
    assert tracker.summary()["lift_confirmed_step"] == 3
    assert tracker.summary()["max_height_delta_m"] == pytest.approx(0.04)


def test_lift_hold_resets_when_object_drops_below_threshold():
    tracker = EpisodeLiftTracker("alphabet_soup_1", 1.0, 0.03, 2)

    tracker.update(step_index=0, object_height_m=1.04, contact_latched=True)
    state = tracker.update(step_index=1, object_height_m=1.02, contact_latched=True)
    assert state["target_object_lift_hold_count"] == 0
    assert not state["target_object_lifted"]

    tracker.update(step_index=2, object_height_m=1.05, contact_latched=True)
    state = tracker.update(step_index=3, object_height_m=1.05, contact_latched=True)
    assert state["target_object_lifted"]
    assert tracker.summary()["lift_confirmed_step"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_object": "", "initial_height_m": 1.0},
        {"target_object": "x", "initial_height_m": math.nan},
        {
            "target_object": "x",
            "initial_height_m": 1.0,
            "height_threshold_m": 0.0,
        },
        {
            "target_object": "x",
            "initial_height_m": 1.0,
            "hold_steps_required": 0,
        },
    ],
)
def test_invalid_lift_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EpisodeLiftTracker(**kwargs)
