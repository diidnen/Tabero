from __future__ import annotations

import unittest

import math

from benchmarks.openpi.activeforcing_runtime import (
    ActiveForcingDecision,
    ActiveForcingExecutor,
    ResponseTargetForceModel,
    select_expected_utility_force,
    validate_native_13d_action,
)


class FakeAction:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.values = list(range(shape[-1]))


class FakeForcePositionAction:
    def __init__(self) -> None:
        self.calls: list[float] = []
        self.clears = 0
        self.status = {
            "mode": "native_policy",
            "status": "DISABLED",
            "requested_force_n": None,
            "measured_force_n": 0.0,
            "bilateral_contact": False,
        }

    def set_target_squeeze_force_n(self, target: float) -> None:
        self.calls.append(target)
        self.status.update(
            mode="newton",
            status="TRACKING",
            requested_force_n=target,
            measured_force_n=target - 0.1,
            bilateral_contact=True,
        )

    def clear_target_squeeze_force_n(self) -> None:
        self.clears += 1
        self.status.update(mode="native_policy", status="DISABLED", requested_force_n=None)

    def get_force_status(self) -> dict:
        return dict(self.status)


def test_scalar_decision_and_expected_utility_selection() -> None:
    decision = ActiveForcingDecision.from_model_output({"target_force_n": 4.0})
    assert decision.target_force_n == 4.0
    selected = select_expected_utility_force([2.0, 4.0], [0.2, 0.95], fmax_task_n=6.0)
    assert selected.target_force_n == 4.0


def test_scalar_interface_rejects_vector_and_nonfinite_values() -> None:
    with unittest.TestCase().assertRaises(TypeError):
        ActiveForcingDecision.from_model_output({"target_force_n": [4.0]})
    with unittest.TestCase().assertRaises(ValueError):
        ActiveForcingDecision.from_model_output({"target_force_n": math.inf})


def test_default_off_does_not_touch_native_term() -> None:
    term = FakeForcePositionAction()
    executor = ActiveForcingExecutor(term, enabled=False)
    executor.begin_episode()
    assert term.clears == 0
    assert executor.consume_model_output({"target_force_n": 4.0}) is None
    assert term.calls == []
    assert executor.telemetry()["force_loop_active"] is False


def test_enabled_executor_uses_one_native_force_loop_and_preserves_action() -> None:
    term = FakeForcePositionAction()
    executor = ActiveForcingExecutor(term, enabled=True)
    executor.begin_episode()
    action = FakeAction((13,))
    before = list(action.values)
    prepared = executor.prepare_action(action)
    executor.consume_model_output({"target_force_n": 4.0})
    executor.consume_model_output({"target_force_n": 4.0})
    assert prepared is action
    assert action.values == before
    assert action.values[:6] == before[:6]
    assert action.values[6:7] == before[6:7]
    assert action.values[7:13] == before[7:13]
    assert term.calls == [4.0]
    telemetry = executor.telemetry()
    assert {"F_des_N", "F_meas_N", "F_cmd", "execution_state", "force_loop_active", "contact_status"} <= telemetry.keys()
    assert telemetry["F_des_N"] == 4.0
    assert telemetry["force_loop_active"] is True


def test_response_model_reads_nested_scalar_without_changing_actions() -> None:
    response = {"actions": FakeAction((10, 13)), "activeforcing": {"target_force_n": 3.5}}
    decision = ActiveForcingDecision.from_model_output(
        ResponseTargetForceModel().predict_target_force_n(inference_response=response, context={})
    )
    assert decision.target_force_n == 3.5
    assert response["actions"].shape == (10, 13)


def test_native_action_dimension_validation() -> None:
    validate_native_13d_action(FakeAction((4, 13)))
    with unittest.TestCase().assertRaises(ValueError):
        validate_native_13d_action(FakeAction((4, 12)))


if __name__ == "__main__":
    _tests = [
        test_scalar_decision_and_expected_utility_selection,
        test_scalar_interface_rejects_vector_and_nonfinite_values,
        test_default_off_does_not_touch_native_term,
        test_enabled_executor_uses_one_native_force_loop_and_preserves_action,
        test_response_model_reads_nested_scalar_without_changing_actions,
        test_native_action_dimension_validation,
    ]
    for _test in _tests:
        _test()
    print(f"PASS activeforcing runtime smoke ({len(_tests)} tests)")
