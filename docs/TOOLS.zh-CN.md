# Tools（统一工具说明：收集 / 测试 / 可视化）

[English](TOOLS.md) | 中文

本目录的脚本很多，但逻辑上可以分为三类：

- **（1）数据收集**：从开源轨迹 replay 再采集、或手动遥操作录制
- **（2）数据测试**：回放/评估成功率/统计指标（尽量 headless + 自动批跑）
- **（3）可视化与其它工具**：LeRobot 数据预览、力/混合控制调试、上传等

> 说明：根目录 [`README.md`](../README.md) 只做“总览”。这里重点写根目录没强调的“tools 逻辑”和“每个脚本怎么用”。

---

## 通用约定（强烈建议先看）

### 1）统一路径/环境变量约定（避免把输入输出搞混）

- **输入：开源轨迹源（assembled）**
  - `HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5`
- **输出：replay 再采集（写出）**
  - `OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos`
  - `OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets`
- **输入：replay 再采集后的数据（读入，评估/回放用）**
  - `REPLAYED_DEMOS_DIR=/path/to/replayed_demos`
  - 通常：`export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"`
- **输入：手动录制数据（record）**
  - 单文件：`RECORDED_DEMOS_PATH=/path/to/single_recorded_demo.hdf5`
  - 目录模式：`RECORDED_DEMOS_DIR=/path/to/recorded_demos`

建议：**不要把输入目录和输出目录指向同一个位置**。

### 2）批跑逻辑（重要）

- `replay_demos_with_camera.py` / `replay_demos.py`：当你传了多个 `--task_suite`，或只传 `--task_suite` 不传 `--task_id` 时，会**通过子进程**逐个 (suite, task_id) 调用自己，避免单进程反复重建 Isaac/Kit 导致不稳定。
- `run_data_evaluations.py`：会对每个 task 反复调用 `replay_demos.py`（子进程），解析 stdout 统计 success/metrics，直到累计 episode 达到 `max_episodes`。
- `run_task_evaluations.py`：会对每个 task 调用 OpenPI（或其它 policy）推理脚本（子进程），解析 stdout 得到成功率、逐 experiment step 计数与力学指标。

### 3）可选 Libero 光照随机化

默认不启用光照随机化，以保持 replay / evaluation 可复现。如果需要在每次 `env.reset()` 时随机化 `/World/light` 的 DomeLight 强度、颜色和 HDR 天空纹理，在 replay 或 evaluation 命令里加 `--randomize_light`：

```bash
python scripts/tools/replay_demos.py \
  --task Isaac-Libero-Franka-Replay-Camera-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --randomize_light \
  --headless
```

运行逻辑：脚本先把 `LIBERO_RANDOMIZE_LIGHT=1` 写入环境变量；随后 `setup_task_objects()` 设置 `TASK_SUITE/TASK_ID`；`parse_env_cfg()` 实例化 Libero cfg；`EventCfgFrankaPanda` 只在该 flag 开启时注册 `randomize_light` reset 事件。ContactForce / Tactile Libero 环境继承同一套基础 Franka Libero event 配置，不需要重复写事件。

批量评测脚本也支持同一参数，并会透传给子进程：

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode tactile \
  --task_suites libero_goal \
  --task_ids 1 \
  --randomize_light \
  --headless
```

### 4）“精简指令 vs 全指令”怎么理解

- **精简指令**：最少参数、适合快速跑通（默认依赖当前 shell 的环境变量/默认值）
- **全指令**：把关键环境变量、输出路径、与常用可选参数都显式写出来，便于复现与团队协作

---

## （1）数据收集

### `set_replay_env.sh`（一键配置 profile）

- **作用**：给当前 shell 设置上面那组环境变量（输入/输出目录等）。

精简指令：

```bash
source scripts/tools/set_replay_env.sh tabero_force_gentle
```

全指令（更通用：不依赖脚本硬编码路径，推荐直接 export）：

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets
export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"
export USE_TABERO_TASKS=0
```

---

### `replay_demos_with_camera.py`（结构化回放/再采集：最常用）

- **作用**：读取开源轨迹（通常来自 `HDF5_TRAJ_SOURCE_DIR`），在 Isaac 中回放，并写出新的：
  - `replayed_demos/*.hdf5`（成功 episode）
  - 可选：相机视频 `video_datasets/.../videos/*.mp4`
  - 可选：触觉视频 `video_datasets/.../tactile_outputs/*.mp4`
- **重要**：支持 `--recorder_type`，其中 `7dpf` 会把 Force(6) 写进 action（得到 13D）。

精简指令（来自你最近的实际运行方式）：

```bash
python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
  --task_suite libero_goal \
  --task_id 2 \
  --dump_data \
  --recorder_type 7dpf \
  --video \
  --headless
```

全指令（推荐模板：路径与关键参数都写清楚）：

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets

python scripts/tools/replay_demos_with_camera.py \
  --task Isaac-Libero-Franka-Replay-Camera-Tactile-v0 \
  --task_suite libero_goal \
  --task_id 2 \
  --num_envs 1 \
  --video \
  --camera_view_list agentview eye_in_hand \
  --tactile_sensor_list gsmini_left gsmini_right \
  --tactile_output_type tactile_rgb \
  --recorder_type 7dpf \
  --dump_data \
  --headless
```

---

### `record_demos.py`（手动遥操作录制）

- **作用**：用键盘/SpaceMouse 等遥操作设备录制 demo，输出一个 HDF5。
- **注意**：LIBERO 任务如果传了 `--task_suite` / `--task_id`，需要先设置 `HDF5_TRAJ_SOURCE_DIR`。`spacemouse` 还要求本机连接了 SpaceMouse；没有设备时请改用 `--teleop_device keyboard`。

精简指令（最小可用，避免默认无限录制）：

```bash
source scripts/tools/set_replay_env.sh inference

python scripts/tools/record_demos.py \
  --task Isaac-Libero-Franka-IK-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --teleop_device keyboard \
  --num_demos 1 \
  --dataset_file ./output/manual_demo.hdf5
```

全指令（常用模板：明确 teleop 设备、任务选择与输出）：

```bash
source scripts/tools/set_replay_env.sh inference

python scripts/tools/record_demos.py \
  --task Isaac-Libero-Franka-IK-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --teleop_device spacemouse \
  --step_hz 30 \
  --num_demos 5 \
  --num_success_steps 10 \
  --recorder_type 8d2 \
  --dataset_file ./output/manual_demo.hdf5
```

---

## （2）数据测试

### `replay_demos.py`（轻量回放/验证：debug 用）

- **作用**：回放一个 HDF5（或按 suite/task 自动遍历），常用于：
  - 验证某个 demo 能否正常复现
  - 调试 headless/相机/力观测
  - 可选：`--validate_states` 做状态一致性验证（仅 `--num_envs 1`）

精简指令（回放单个文件）：

```bash
python scripts/tools/replay_demos.py --task Isaac-Libero-Franka-Replay-Camera-v0 --dataset_file /path/to/demo.hdf5 --demo_id 0
```

全指令（按 suite/task 回放 + 可选校验 + headless）：

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos

python scripts/tools/replay_demos.py \
  --task Isaac-Libero-Franka-Replay-Camera-ContactForce-v0 \
  --task_suite libero_goal \
  --task_id 5 \
  --num_envs 1 \
  --validate_states \
  --dump_data \
  --headless
```

---

### `run_data_evaluations.py`（数据回放评估：成功率/指标）

- **作用**：批量评估 `REPLAYED_DEMOS_DIR` 下的数据质量。
- **逻辑**：
  - 对每个 task 从 `REPLAYED_DEMOS_DIR` 解析出对应的 `{suite}_task{id}_*_demo.hdf5`
  - 子进程调用 `replay_demos.py` 回放
  - 解析 stdout 统计 success / Hybrid metrics
  - 累计回放 episode 达到 `max_episodes` 后输出 JSON/TXT 表格

精简指令（跑一组 suite，headless）：

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/run_data_evaluations.py --control_mode tactile --headless
```

全指令（明确 suite/task、episode 数、输出目录、以及需要的环境）：

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos

python scripts/tools/run_data_evaluations.py \
  --task_suites libero_10 libero_spatial libero_object libero_goal \
  --task_ids 0 1 2 \
  --control_mode tactile \
  --max_episodes 50 \
  --num_envs 1 \
  --output_dir ./evaluation_results \
  --output_format both \
  --replay_script scripts/tools/replay_demos.py \
  --headless
```

---

### `run_task_evaluations.py`（策略推理评估：OpenPI / 其它）

- **作用**：对每个 task 启动推理脚本（默认 OpenPI client），统计成功率，并解析逐 experiment 的 step 与力学指标；OpenPI 路径会记录成功和失败 experiment，逐步轨迹接口也仅覆盖 OpenPI。
- **逻辑**：
  - 子进程调用 `benchmarks/openpi/openpi_inference_client.py`
  - 从 stdout 解析 “Success rate / Hybrid metrics”

精简指令（OpenPI + diffik，跑一个 task）：

```bash
python scripts/tools/run_task_evaluations.py --policy_model openpi --control_mode diffik --task_suites libero_goal --task_ids 1 --num_total_experiments 5 --headless
```

全指令（显式指定 server、数据 reset 源、debug 等）：

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode tactile \
  --server_host 127.0.1.1 \
  --server_port 8000 \
  --task_suites libero_goal libero_10 \
  --task_ids 0 1 2 \
  --num_total_experiments 50 \
  --num_success_steps 8 \
  --max_inference_steps 80 \
  --replan_steps 10 \
  --hdf5_folder /path/to/libero/assembled_hdf5 \
  --debug_mode 0 \
  --output_dir ./evaluation_results \
  --output_format both \
  --record-step-traces \
  --step-trace-dir ./evaluation_results/my_step_traces \
  --headless
```

`--record-step-traces` 是可选开关，默认关闭。启用但没有指定 `--step-trace-dir` 时，轨迹默认写入
`<output_dir>/step_traces_<model>_<timestamp>/<suite>_task<ID>.jsonl`。只指定目录但未启用开关会直接报错。该 JSONL 仅记录力与索引，不保存图像，也不依赖 `debug_mode=6`。

Step 语义：

- `env_steps` 只统计策略控制阶段实际完成的 `env.step()` 调用，不包含 reset/setup。
- `inference_chunks` 每次有效策略请求加一，因此应满足 `env_steps <= inference_chunks * replan_steps`。
- JSONL 每行对应一个 `env.step()`，包含 experiment/step/chunk 索引、预测与实测左右指局部 3D 力、squeeze/ap 及接触标记。

JSON 保留已有 task 级力字段，并新增 `metrics_status`、`metrics_warnings`、`step_statistics`、`force_metric_episode_counts`、`episodes` 和 `step_trace`。`status` 仍只表示成功率解析状态；指标不完整只会把 `metrics_status` 设为 `partial`。TXT 保留原 task 汇总，同时新增逐 experiment 的 step/覆盖率表和八项力指标表；缺失值统一写 `N/A`。

八项 task 级力指标仍只汇总成功 experiment。每个成功或失败 episode 都记录 `squeeze_avg_pred`、`squeeze_avg_meas`、`squeeze_max_pred`、`squeeze_max_meas`、`ap_avg_pred`、`ap_avg_meas`、`ap_max_pred`、`ap_max_meas`。其中 `max` 仍是非零力帧中最大 Top 5% 的均值，不是单帧最大值。tactile/hybrid 要求预测与实测完整覆盖；binary 允许预测力缺失；diffik/osc 的力状态是 `not_applicable`，但仍记录 step。

---

### `export_mode6_five_panel_video.py`（mode-6 五画面视频导出）

将已有 OpenPI `debug_mode=6` experiment 采集数据离线拼成一个五画面视频：

1. agentview；
2. eye-in-hand；
3. 左侧 GelSight `markers_rgb`；
4. 右侧 GelSight `markers_rgb`；
5. 上下排列、随时间逐帧展开的模型原始 squeeze 目标 / 控制器有效目标 / 实测 squeeze，
   以及模型原始 `d_pred` / 最终控制器 `d_cmd` / 实测 `d_actual` 曲线。

这是纯离线后处理：不会启动 Isaac 或策略 server，也不会改写原评测。四路 PNG
和 `forces.jsonl` 必须具有从 0 开始、连续且完全一致的帧索引。成功 episode 的 debug
帧数可能比 `env_steps` 少 1；导出器保留真实采集边界，不伪造 terminal 帧。

```bash
python scripts/tools/export_mode6_five_panel_video.py \
  --capture-exp-dir /path/to/capture_mode6/libero_object/task_0/none/<timestamp>/exp_000 \
  --step-trace /path/to/traces/libero_object_task0.jsonl \
  --output /path/to/combined_5view_force_gripper_curve.mp4 \
  --fps 20 \
  --width 1920 \
  --height 1080
```

输出默认为 1920×1080、20 FPS、H.264/YUV420P。工具会用 FFprobe 审计 codec、像素格式、
分辨率、帧率和帧数，并默认写出 `combined_5view_force_curve.manifest.json`。若视频或
manifest 已存在，需要显式传入 `--overwrite` 才会覆盖。

力语义必须区分：mode-6 scalar schema v3 分别记录模型原始 `squeeze_pred`、经过前馈或
override 后的控制器有效目标 `squeeze_target_eff`、控制器 EMA `squeeze_meas`，以及未滤波
实测值 `squeeze_meas_raw`。视频用蓝线画模型原始目标、红线画控制器有效目标、橙线画 raw
实测力。raw 实测值还必须经过 grasp/contact/reward-valid gate 后，才会进入
`trajectory_mean_measured_squeeze`。对于 scalar schema v2 的旧采集，导出器默认自动查找
`<run>/traces/<suite>_task<ID>.jsonl`，补齐控制器有效目标与 raw 实测值；非标准目录可用
`--step-trace` 显式指定。使用前会逐帧核对 trace 与 capture 的
`squeeze_pred/squeeze_meas`，再读取 `effective_squeeze_target_n` 和
`reward_squeeze_meas_raw`。找不到匹配 trace 时会失败，不会静默退回旧的 EMA 两曲线。

夹爪语义单独审计。mode-6 scalar schema v2 同时保存模型原始 `d_pred`、经过力反馈修正并
截断后的最终 `d_cmd`、左右指 post-step 原始关节位置，以及
`d_actual=mean(q_left,q_right)`。视频默认用青线画 `d_pred`、绿线画 `d_cmd`、紫线画
`d_actual`，单位为 mm。loader 会验证 `d_actual` 与左右关节均值一致，并拒绝把带符号
policy observation `[q_left,-q_right]` 求平均所得的旧 capture。manifest schema v5 明确
记录这三条已绘制的夹爪曲线。

---

### `raw_data_retention_analysis.py`（“成功数据留存率”统计）

- **作用**：统计 `REPLAYED_DEMOS_DIR` 下每个 HDF5 里成功 demo 的数量（以 `/data/demo_*` 计），并与期望值对比。

精简指令：

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/raw_data_retention_analysis.py
```

全指令：

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos

python scripts/tools/raw_data_retention_analysis.py \
  --task_suites libero_10 libero_spatial libero_goal libero_object \
  --expected_episodes_per_file 50 \
  --output_dir ./evaluation_results
```

---

## （3）可视化和其他工具

### `force_debug_playground.py`（力调试：回放 + 局部力曲线/保存）

- **作用**：回放某条 demo，并从 `obs["policy"]["gripper_net_force"]` 提取左右指局部力 `(2,3)` 进行打印/可视化/保存。
- **适用**：ContactForce / Tactile 环境（通过 `--env_variant` 选择）。

精简指令（headless 保存曲线图，更适合服务器）：

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/force_debug_playground.py --env_variant contactforce --task_suite libero_goal --task_id 5 --demo_id 0 --save_plot ./force.png --headless
```

全指令（打印更密 + 保存 npz + 限制步数）：

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos

python scripts/tools/force_debug_playground.py \
  --env_variant tactile \
  --task_suite libero_goal \
  --task_id 5 \
  --demo_id 0 \
  --num_steps 300 \
  --print_every 5 \
  --save_npz ./force_series.npz \
  --save_plot ./force.png \
  --headless
```

---

### `upload_lerobot_to_hf.py`（上传 LeRobot 数据到 Hugging Face）

精简指令（默认只上传 `data/` + `meta/`）：

```bash
python scripts/tools/upload_lerobot_to_hf.py --local-path /path/to/lerobot_dataset_root --repo-id your_username/your_dataset
```

全指令（显式控制仓库类型与是否私有）：

```bash
python scripts/tools/upload_lerobot_to_hf.py \
  --local-path /path/to/lerobot_dataset_root \
  --repo-id your_username/your_dataset \
  --repo-type dataset \
  --private
```

如果要上传公开数据集，省略 `--private`。如果 Hugging Face 仓库已经存在且不希望脚本创建仓库，可加 `--no-create-repo`。
