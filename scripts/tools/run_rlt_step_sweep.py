#!/usr/bin/env python3
"""Run and summarize the Tabero RLT step 5 evaluation on tasks 0-5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


STEPS = (5,)
TASKS = tuple(range(6))
BASELINES: dict[int, int | None] = {0: 31, 1: 47, 2: 44, 3: 12, 4: None, 5: 31}
TASK_NAMES = {
    0: "alphabet soup",
    1: "cream cheese",
    2: "salad dressing",
    3: "BBQ sauce",
    4: "ketchup",
    5: "tomato sauce",
}

WORKSPACE = Path(__file__).resolve().parents[3]
TABERO = WORKSPACE / "Tabero"
T2VLA = WORKSPACE / "T2-VLA"
RLINF = WORKSPACE / "RLinf"
BASE_MODEL = WORKSPACE / "models/pi0_lora_tacfield_tabero_safetensors"
STAGE1 = (
    WORKSPACE
    / "results/tabero_rlt_stage1_20260721_tactile_fixed"
    / "tabero_rlt_stage1_tacfield_tactile_fixed/checkpoints/global_step_2000"
    / "actor/model_state_dict/trainable_weights.pt"
)
STAGE2_ROOT = (
    WORKSPACE
    / "results/tabero_rlt_stage2_task5_firm_allgpu_lowload_formal_20260722"
    / "tabero_rlt_stage2_task5_firm/checkpoints"
)
DEFAULT_OUTPUT = (
    WORKSPACE
    / "results/tabero_rlt_stage2_task5_firm_allgpu_lowload_formal_20260722"
    / "evaluations/rlt_step_sweep_20260725"
)
STEP5_BUNDLE = (
    WORKSPACE
    / "results/tabero_rlt_stage2_task5_firm_allgpu_lowload_formal_20260722"
    / "exports/global_step_5"
)
# Follow the Record archive convention: full_time-taskid-adverb-train_method-annotation.
REPORT = WORKSPACE / "Record/experiments/2026-07-25_task0-5_firm_rlt_step5-evaluation.md"
LEGACY_REPORT = WORKSPACE / "Record/experiments/2026-07-25_rlt_training_step_task0_5_success_curve.md"
HDF5 = TABERO / "benchmarks/datasets/libero/assembled_hdf5"
STEP5_ACTOR_SHA256 = "0b196114d78726e839573e101f75636179b056ba6b2fd1a66f52f6f87684580c"
STEP5_ENCODER_SHA256 = "e421623fd09cbd842f3ec57ca4276e72a364c355830e6b5ad824081b175ef6fe"


def _result_candidates(result_dir: Path) -> list[Path]:
    return sorted(result_dir.glob("success_rates_*.json"), key=lambda path: path.stat().st_mtime)


def load_completed_result(path: Path, task: int) -> dict[str, Any] | None:
    """Return a validated 50-episode task result, or None when incomplete."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["results"][f"libero_object_task{task}"]
        raw_success = result["successful_experiments"]
        raw_total = result["total_experiments"]
        if type(raw_success) is not int or type(raw_total) is not int:
            return None
        success = raw_success
        total = raw_total
        rate = float(result["success_rate"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if result.get("status") != "completed" or total != 50 or not 0 <= success <= total:
        return None
    if abs(rate - 100.0 * success / total) > 0.011:
        return None
    return result


def find_completed_result(result_dir: Path, task: int) -> tuple[Path, dict[str, Any]] | None:
    for path in reversed(_result_candidates(result_dir)):
        result = load_completed_result(path, task)
        if result is not None:
            return path, result
    return None


def build_summary(output_root: Path, tasks: Iterable[int] = TASKS, steps: Iterable[int] = STEPS) -> dict[str, Any]:
    """Build the machine-readable sweep summary from validated task JSON files."""
    task_list = list(tasks)
    rows: list[dict[str, Any]] = []
    for step in steps:
        row: dict[str, Any] = {"step": step}
        rates: list[float] = []
        deltas: list[float] = []
        for task in task_list:
            found = find_completed_result(output_root / f"task{task}" / f"step{step}", task)
            success = total = rate = delta = None
            result_path = None
            if found:
                path, result = found
                success = int(result["successful_experiments"])
                total = int(result["total_experiments"])
                rate = float(result["success_rate"])
                result_path = str(path)
                rates.append(rate)
                baseline = BASELINES.get(task)
                if baseline is not None:
                    delta = rate - 2.0 * baseline
                    deltas.append(delta)
            row.update(
                {
                    f"task{task}_success": success,
                    f"task{task}_total": total,
                    f"task{task}_sr": rate,
                    f"task{task}_delta_pp": delta,
                    f"task{task}_result": result_path,
                }
            )
        row["completed_tasks"] = len(rates)
        row["macro_sr_all_tasks"] = sum(rates) / len(rates) if rates else None
        row["macro_delta_baseline_tasks_pp"] = sum(deltas) / len(deltas) if deltas else None
        rows.append(row)
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "suite": "libero_object",
            "control_mode": "tactile",
            "episodes_per_task": 50,
            "seed": 11,
            "prompt_seed": 0,
            "prompt_adverbs": ["firmly", "tightly"],
            "num_success_steps": 8,
            "replan_steps": 10,
            "fixed_hdf5_reset_states": True,
        },
        "baseline_source": "Record/experiments/2026-07-15_all_task_step0_pytorch_baseline.md",
        "baselines": {str(task): (None if value is None else value / 50.0 * 100.0) for task, value in BASELINES.items()},
        "rows": rows,
    }


def _fmt(value: Any, suffix: str = "") -> str:
    return "N/A" if value is None else f"{float(value):.2f}{suffix}"


def _display_path(path: Path) -> str:
    absolute = path if path.is_absolute() else (Path.cwd() / path).absolute()
    try:
        return str(absolute.relative_to(WORKSPACE))
    except ValueError:
        return str(absolute)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(output_root: Path, report_path: Path | None = None) -> dict[str, Any]:
    summary = build_summary(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = ["step", "completed_tasks"]
    for task in TASKS:
        fields.extend((f"task{task}_success", f"task{task}_total", f"task{task}_sr", f"task{task}_delta_pp"))
    fields.extend(("macro_sr_all_tasks", "macro_delta_baseline_tasks_pp"))
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["rows"])

    actor_path = STEP5_BUNDLE / "rlt_actor.safetensors"
    encoder_path = STEP5_BUNDLE / "rlt_encoder.safetensors"
    manifest_path = STEP5_BUNDLE / "manifest.json"
    actor_sha256 = _sha256(actor_path)
    encoder_sha256 = _sha256(encoder_path)
    if actor_sha256 != STEP5_ACTOR_SHA256 or encoder_sha256 != STEP5_ENCODER_SHA256:
        raise ValueError("Step 5 bundle hash changed since preflight validation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = summary["rows"][0]
    completed_retention_rates = [
        row[f"task{task}_sr"]
        for task in range(5)
        if row[f"task{task}_sr"] is not None
    ]
    retention_macro = (
        sum(completed_retention_rates) / len(completed_retention_rates)
        if completed_retention_rates
        else None
    )
    baseline_retention_deltas = [
        row[f"task{task}_delta_pp"]
        for task in range(4)
        if row[f"task{task}_delta_pp"] is not None
    ]
    retention_delta = (
        sum(baseline_retention_deltas) / len(baseline_retention_deltas)
        if baseline_retention_deltas
        else None
    )
    shell_continuation = " " + chr(92) + chr(10)
    server_command = shell_continuation.join(
        (
            "CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "T2-VLA/.venv/bin/python T2-VLA/scripts/serve_policy.py",
            "  --port 18122",
            f"  --rlt-bundle {_display_path(STEP5_BUNDLE)}",
            "  policy:checkpoint",
            "  --policy.config pi0_lora_tacfield_tabero",
            f"  --policy.dir {_display_path(BASE_MODEL)}",
        )
    )
    client_command = shell_continuation.join(
        (
            "conda run --no-capture-output -n tabero python -u",
            "  Tabero/scripts/tools/run_task_evaluations.py",
            "  --policy-model openpi --control-mode tactile",
            "  --server-host 127.0.0.1 --server-port 18122",
            "  --num-total-experiments 50 --num-success-steps 8 --replan-steps 10",
            "  --task-suites libero_object --task-ids TASK_ID",
            f"  --hdf5-folder {_display_path(HDF5)} --require-hdf5",
            "  --output-dir RESULT_DIR --output-format both",
            "  --seed 11 --prompt-seed 0 --prompt-adverbs firmly tightly",
            "  --sim-device cuda:1 --sim-kit-args=--/renderer/activeGpu=1 --headless",
        )
    )

    lines = [
        "# RLT 训练步数 Task 0-5 成功率曲线",
        "",
        f"- 更新时间：{summary['generated_at']}",
        "- 模型：`models/pi0_lora_tacfield_tabero_safetensors` + Task 5 Firm RLT actor",
        f"- 原始结果：`{_display_path(output_root)}`",
        "- 范围：仅 RLT Step 5，依次评测 Task 0→5",
        f"- 运行状态：已完成 {row['completed_tasks']}/6 个任务；N/A 仅表示无正式历史 baseline。",
        "",
        "## 评测参数",
        "",
        "- suite：`libero_object`；tasks：0-5；control mode：`tactile`。",
        "- 每任务 50 episodes；`seed=11`；`prompt_seed=0`；prompt adverbs：`firmly tightly`。",
        "- `num_success_steps=8`；`replan_steps=10`；object suite 最大推理步数为 30。",
        "- 强制从每个任务 HDF5 的 50 个固定 reset states 启动；headless。",
        "- Policy server 使用物理 GPU0；Isaac client 与 renderer 使用物理 GPU1；全程串行。",
        "- 本轮没有重新训练或修改权重，只评测已导出的 RLT Step 5。",
        "",
        "## 权重与输入",
        "",
        f"- baseline：`{_display_path(BASE_MODEL)}`。",
        f"- RLT bundle：`{_display_path(STEP5_BUNDLE)}`。",
        f"- manifest：`{_display_path(manifest_path)}`，`stage2_global_step={manifest['stage2_global_step']}`。",
        f"- actor：`{_display_path(actor_path)}`；8 tensors；SHA-256 `{actor_sha256}`。",
        f"- encoder：`{_display_path(encoder_path)}`；31 tensors；SHA-256 `{encoder_sha256}`。",
        f"- Stage 1 来源：`{manifest['stage1_checkpoint']}`。",
        f"- Stage 2 manifest 记录的训练时来源：`{manifest['stage2_checkpoint']}`。原始 checkpoint 已清理，本轮以导出 bundle 及哈希为权威输入。",
        f"- 固定 reset-state 数据集目录：`{_display_path(HDF5)}`。",
        "- 评测前校验通过：manifest step=5；actor 8 tensors、encoder 31 tensors；shape/dtype 合法且数值均 finite。",
        "",
        "## 训练来源与参数",
        "",
        "- 本文记录的是已训练权重的评测；本轮未重新训练。",
        "- Stage 2 训练配置：`RLinf/examples/tabero/tabero_rlt_stage2_ac_task5_firm.yaml`；训练目标为 `libero_object/task 5` 的 Firm prompt。",
        "- 正式训练运行至 global step 100，RLT Step 5 bundle 来自其中的 global step 5；训练过程、8-GPU 拓扑、rollout/learner 指标和 checkpoint 审计见 `Record/experiments/2026-07-22_task5_firm_rlt_stage2_actor_critic.md`。",
        "- 训练日志：`results/tabero_rlt_stage2_task5_firm_allgpu_lowload_formal_20260722/metrics.log`；TensorBoard：`results/tabero_rlt_stage2_task5_firm_allgpu_lowload_formal_20260722/tensorboard`。",
        "",
        "## 启动命令",
        "",
        "以下 server/client 命令从 workspace 根目录运行；编排入口单独从 `Tabero/` 运行。",
        "",
        "编排入口（从 `Tabero/` 运行）：",
        "",
        "```bash",
        "python scripts/tools/run_rlt_step_sweep.py --phase run",
        "```",
        "",
        "每个任务独立启动的 server 命令：",
        "",
        "```bash",
        server_command,
        "```",
        "",
        "每个任务使用的 client 命令（`TASK_ID` 依次为 0-5，`RESULT_DIR` 为对应 `taskN/step5`）：",
        "",
        "```bash",
        client_command,
        "```",
        "",
        "## 历史 Step 0",
        "",
        "Step 0 引用 `Record/experiments/2026-07-15_all_task_step0_pytorch_baseline.md`，",
        "不是本轮同批次配对数据。Task 4 无该正式 baseline，因此 delta 保持 N/A。",
        "",
        "| Task | Object | Step 0 Firm |",
        "|---:|---|---:|",
    ]
    for task in TASKS:
        baseline = summary["baselines"][str(task)]
        lines.append(f"| {task} | {TASK_NAMES[task]} | {_fmt(baseline, '%')} |")
    lines.extend(("", "## RLT 结果", ""))
    for task in TASKS:
        lines.extend((f"### Task {task}: {TASK_NAMES[task]}", "", "| Step | Success | SR | vs Step 0 |", "|---:|---:|---:|---:|"))
        for row in summary["rows"]:
            success = row[f"task{task}_success"]
            total = row[f"task{task}_total"]
            count = "N/A" if success is None else f"{success}/{total}"
            lines.append(f"| {row['step']} | {count} | {_fmt(row[f'task{task}_sr'], '%')} | {_fmt(row[f'task{task}_delta_pp'], 'pp')} |")
        lines.append("")
    lines.extend(("## 跨任务汇总", "", "| Step | 完成任务 | 六任务绝对 Macro SR | 五个有基线任务 Macro Delta |", "|---:|---:|---:|---:|"))
    for row in summary["rows"]:
        lines.append(f"| {row['step']} | {row['completed_tasks']}/6 | {_fmt(row['macro_sr_all_tasks'], '%')} | {_fmt(row['macro_delta_baseline_tasks_pp'], 'pp')} |")
    lines.extend(
        (
            "",
            "## Task 5 专项与 Task 0-4 保持",
            "",
            f"- Task 5（训练目标）Step 5 SR：{_fmt(row['task5_sr'], '%')}；相对历史 Step 0：{_fmt(row['task5_delta_pp'], 'pp')}。",
            f"- Task 0-4 Step 5 绝对 macro SR：{_fmt(retention_macro, '%')}（{len(completed_retention_rates)}/5 任务完成）。",
            f"- Task 0-3（有正式 baseline）macro delta：{_fmt(retention_delta, 'pp')}；Task 4 因无正式 baseline 不进入该 delta。",
            "",
            "## 解释约束",
            "",
            "Task 5 是 RLT 训练目标；Task 0-4 用于衡量跨任务保持或灾难性退化。",
            "由于 Step 0 来自历史批次，百分点变化用于趋势参考，不声明严格配对统计显著性。",
            "结果 JSON 只保存任务级聚合字段，不包含逐 episode 结构化数组；50-episode 完整性由 `status=completed`、`total_experiments=50`、成功率算术一致性、client 日志和 `evaluation_completed` 事件共同验证。",
            "",
            "## 运行记录",
            "",
            f"- 编排日志：`{_display_path(output_root / 'step5_only_sweep.log')}`。",
            f"- 事件日志：`{_display_path(output_root / 'events.jsonl')}`。",
            f"- 机器可读汇总：`{_display_path(output_root / 'summary.json')}`、`{_display_path(output_root / 'summary.csv')}`。",
            "- 每个 `taskN/stepM/` 独立保存 server/client 日志与 evaluator JSON。",
            "",
            "| Task | Result JSON | Client log | Server log |",
            "|---:|---|---|---|",
        )
    )
    for task in TASKS:
        result_path = row[f"task{task}_result"]
        result_display = "N/A" if result_path is None else f"`{_display_path(Path(result_path))}`"
        result_dir = output_root / f"task{task}" / "step5"
        lines.append(
            f"| {task} | {result_display} | `{_display_path(result_dir / 'client.log')}` | `{_display_path(result_dir / 'server.log')}` |"
        )
    lines.extend(
        (
            "",
            "## 失败、重试与中断记录",
            "",
            "以下失败尝试均未生成通过结构校验的 50-episode 完整结果，不计入正式统计：",
            "",
            "1. Task 0 / Step 5 首次尝试运行约 287 秒并推进至 episode 9，policy server 以 `-9` 退出，client 收到 WebSocket `ConnectionClosedError`。",
            "2. Task 0 / Step 5 第二、三次尝试均因 client 使用错误 Python 环境立即失败：`ModuleNotFoundError: isaaclab`。",
            f"3. 上述三次无效尝试已隔离至 `{_display_path(output_root / 'aborted_runs/task0_step5_failed_attempts_before_clean_restart_20260725')}`。",
            f"4. scope 改为仅 Step 5 前，Task 0 的 Step 40 已启动后中断，日志隔离至 `{_display_path(output_root / 'aborted_runs/task0_step40_interrupted_before_step5_20260725')}`。",
            "5. scope 改动前已完成的 Task 0 Step 10/20/30 均为 0/50；这些结果不计入本轮 Step 5 结论，也不进入 `summary.json`/`summary.csv`。",
            "",
            "正式重启后只跳过通过 `status=completed`、`total_experiments=50`、成功数/成功率一致性校验的结果。",
        )
    )
    report_text = "\n".join(lines) + "\n"
    report_targets = (REPORT, LEGACY_REPORT) if report_path is None else (report_path,)
    for target in dict.fromkeys(report_targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report_text, encoding="utf-8")
    return summary


def _event(output_root: Path, event: str, **details: Any) -> None:
    record = {"timestamp": datetime.now().astimezone().isoformat(), "event": event, **details}
    with (output_root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def preflight() -> None:
    required = [
        BASE_MODEL / "model.safetensors",
        STAGE1,
        HDF5,
        T2VLA / ".venv/bin/python",
        Path("/data/home/sim6g/anaconda3/envs/tabero/bin/python"),
    ]
    required.extend(
        (STEP5_BUNDLE / name)
        for name in ("manifest.json", "rlt_encoder.safetensors", "rlt_actor.safetensors")
    )
    missing = [str(path) for path in required if not path.exists()]
    for task in TASKS:
        if not list(HDF5.glob(f"libero_object_task{task}_*_demo.hdf5")):
            missing.append(f"HDF5 task {task} under {HDF5}")
    if missing:
        raise FileNotFoundError("Preflight missing:\n" + "\n".join(missing))
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 18122)) == 0:
            raise RuntimeError("Port 18122 is already in use")


def export_bundles(output_root: Path) -> None:
    exporter = RLINF / "rlinf/utils/ckpt_convertor/export_tabero_rlt_for_t2vla.py"
    for step in STEPS:
        bundle = output_root / "bundles" / f"step{step}"
        command = [
            sys.executable,
            str(exporter),
            "--stage1-checkpoint", str(STAGE1),
            "--stage2-checkpoint", str(STAGE2_ROOT / f"global_step_{step}/actor/model_state_dict/full_weights.pt"),
            "--output-dir", str(bundle),
            "--base-model", str(BASE_MODEL),
            "--stage2-global-step", str(step),
        ]
        subprocess.run(command, check=True, cwd=RLINF)
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if manifest["stage2_global_step"] != step or manifest["encoder_tensor_count"] <= 0 or manifest["actor_tensor_count"] <= 0:
            raise ValueError(f"Invalid exported manifest for step {step}")
        _event(output_root, "bundle_exported", step=step, manifest=str(bundle / "manifest.json"))


def _wait_for_port(port: int, process: subprocess.Popen[Any], timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited with code {process.returncode}")
        with socket.socket() as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                time.sleep(2.0)
                return
        time.sleep(1.0)
    raise TimeoutError(f"Policy server did not listen on port {port}")


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_sweep(output_root: Path) -> None:
    for task in TASKS:
        for step in STEPS:
            result_dir = output_root / f"task{task}" / f"step{step}"
            
            print(f"=== Running evaluation for task {task} ({TASK_NAMES[task]}) step {step} ===")
            
            if find_completed_result(result_dir, task):
                _event(output_root, "evaluation_skipped_complete", task=task, step=step)
                print(f"Task {task} step {step} already has a complete result, skipping.")
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            server_log_path = result_dir / "server.log"
            client_log_path = result_dir / "client.log"
            server_env = os.environ.copy()
            server_env.update({"CUDA_VISIBLE_DEVICES": "0", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"})
            bundle = STEP5_BUNDLE if step == 5 else output_root / "bundles" / f"step{step}"
            server_command = [
                str(T2VLA / ".venv/bin/python"), str(T2VLA / "scripts/serve_policy.py"),
                "--port", "18122", "--rlt-bundle", str(bundle),
                "policy:checkpoint", "--policy.config", "pi0_lora_tacfield_tabero", "--policy.dir", str(BASE_MODEL),
            ]
            server_log = server_log_path.open("a", encoding="utf-8")
            server = subprocess.Popen(server_command, cwd=T2VLA, env=server_env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            
            print(f"Starting policy server for task {task} step {step}, logging to {server_log_path}")
            print(f"Server command: {' '.join(server_command)}")
            
            try:
                _wait_for_port(18122, server)
                _event(output_root, "evaluation_started", task=task, step=step, server_pid=server.pid)
                client_command = [
                    "conda", "run", "--no-capture-output", "-n", "tabero", "python", "-u",
                    "scripts/tools/run_task_evaluations.py",
                    "--policy-model", "openpi", "--control-mode", "tactile",
                    "--server-host", "127.0.0.1", "--server-port", "18122",
                    "--num-total-experiments", "50", "--num-success-steps", "8", "--replan-steps", "10",
                    "--task-suites", "libero_object", "--task-ids", str(task),
                    "--hdf5-folder", str(HDF5), "--require-hdf5",
                    "--output-dir", str(result_dir), "--output-format", "both",
                    "--seed", "11", "--prompt-seed", "0", "--prompt-adverbs", "firmly", "tightly",
                    "--sim-device", "cuda:1", "--sim-kit-args=--/renderer/activeGpu=1", "--headless",
                ]
                client_env = os.environ.copy()
                client_env.pop("CUDA_VISIBLE_DEVICES", None)
                
                print(f"Starting task evaluation for task {task} step {step}, logging to {client_log_path}")
                print(f"Client command: {' '.join(client_command)}")
                
                with client_log_path.open("a", encoding="utf-8") as client_log:
                    completed = subprocess.run(client_command, cwd=TABERO, env=client_env, stdout=client_log, stderr=subprocess.STDOUT)
                found = find_completed_result(result_dir, task)
                if found is None:
                    _event(output_root, "evaluation_failed", task=task, step=step, return_code=completed.returncode)
                    raise RuntimeError(f"Task {task} step {step} did not produce a complete 50-episode JSON")
                _, result = found
                _event(output_root, "evaluation_completed", task=task, step=step, success=result["successful_experiments"], total=50, return_code=completed.returncode)
                write_summary(output_root)
            finally:
                _stop_process_group(server)
                server_log.close()
                time.sleep(3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("all", "preflight", "export", "run", "summarize"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.phase in ("all", "preflight"):
        preflight()
        _event(output_root, "preflight_completed")
    if args.phase in ("all", "export"):
        export_bundles(output_root)
        _event(output_root, "all_bundles_exported")
    if args.phase in ("all", "run"):
        run_sweep(output_root)
    if args.phase in ("all", "summarize"):
        write_summary(output_root)


if __name__ == "__main__":
    main()
