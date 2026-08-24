import json
from pathlib import Path

import run_rlt_step_sweep as sweep


def test_sweep_only_runs_step5() -> None:
    assert sweep.STEPS == (5,)


def _write_result(path: Path, task: int, success: int, total: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {"num_total_experiments": 50},
                "results": {
                    f"libero_object_task{task}": {
                        "task_suite": "libero_object",
                        "task_id": task,
                        "status": "completed",
                        "successful_experiments": success,
                        "total_experiments": total,
                        "success_rate": 100.0 * success / total,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_completed_result_requires_exactly_50_episodes(tmp_path: Path) -> None:
    complete = tmp_path / "complete.json"
    partial = tmp_path / "partial.json"
    _write_result(complete, task=0, success=31)
    _write_result(partial, task=0, success=4, total=5)

    assert sweep.load_completed_result(complete, task=0)["successful_experiments"] == 31
    assert sweep.load_completed_result(partial, task=0) is None


def test_completed_result_rejects_non_integer_counts(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    _write_result(malformed, task=0, success=31)
    payload = json.loads(malformed.read_text(encoding="utf-8"))
    payload["results"]["libero_object_task0"]["successful_experiments"] = 31.0
    malformed.write_text(json.dumps(payload), encoding="utf-8")

    assert sweep.load_completed_result(malformed, task=0) is None


def test_summary_uses_historical_baseline_and_excludes_task4_delta(
    tmp_path: Path,
) -> None:
    for task, success in enumerate((10, 20, 30, 40, 50, 45)):
        _write_result(
            tmp_path / f"task{task}" / "step10" / "success_rates_test.json",
            task,
            success,
        )

    summary = sweep.build_summary(tmp_path, tasks=range(6), steps=(10,))
    row = summary["rows"][0]

    assert row["task4_delta_pp"] is None
    assert row["task5_delta_pp"] == 28.0
    assert row["macro_sr_all_tasks"] == 65.0
    assert row["macro_delta_baseline_tasks_pp"] == -8.0
    assert summary["baselines"]["4"] is None


def test_report_records_required_experiment_provenance(tmp_path: Path) -> None:
    for task, success in enumerate((1, 2, 3, 4, 5, 6)):
        _write_result(
            tmp_path / f"task{task}" / "step5" / "success_rates_test.json",
            task,
            success,
        )

    report = tmp_path / "report.md"
    sweep.write_summary(tmp_path, report_path=report)
    text = report.read_text(encoding="utf-8")

    required_fragments = (
        "## 评测参数",
        "## 权重与输入",
        "0b196114d78726e839573e101f75636179b056ba6b2fd1a66f52f6f87684580c",
        "e421623fd09cbd842f3ec57ca4276e72a364c355830e6b5ad824081b175ef6fe",
        "## 启动命令",
        "--rlt-bundle",
        "--sim-device cuda:1",
        "## Task 5 专项与 Task 0-4 保持",
        "## 失败、重试与中断记录",
        "aborted_runs/task0_step5_failed_attempts_before_clean_restart_20260725",
        "Step 10/20/30",
        "不计入本轮 Step 5 结论",
        "step5_only_sweep.log",
    )
    for fragment in required_fragments:
        assert fragment in text
    assert "运行状态：已完成 6/6 个任务" in text
    assert "Task 0-4 Step 5 绝对 macro SR：6.00%（5/5 任务完成）" in text
    assert "\\n" not in text
    assert "Tabero/benchmarks/datasets/libero/assembled_hdf5" in text
    assert "以下 server/client 命令从 workspace 根目录运行" in text
    assert "结果 JSON 只保存任务级聚合字段，不包含逐 episode 结构化数组" in text


def test_default_report_is_mirrored_to_legacy_path(
    tmp_path: Path, monkeypatch,
) -> None:
    for task in range(6):
        _write_result(
            tmp_path / f"task{task}" / "step5" / "success_rates_test.json",
            task,
            task,
        )
    canonical = tmp_path / "canonical.md"
    legacy = tmp_path / "legacy.md"
    monkeypatch.setattr(sweep, "REPORT", canonical)
    monkeypatch.setattr(sweep, "LEGACY_REPORT", legacy)

    sweep.write_summary(tmp_path)

    assert canonical.read_bytes() == legacy.read_bytes()
