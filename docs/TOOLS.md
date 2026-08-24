# Tools (Collection / Evaluation / Visualization)

English | [中文](TOOLS.zh-CN.md)

This page is the central reference for scripts under `scripts/tools/`. The root [`README.md`](../README.md) is only a high-level overview; this document explains what each tool does and how to run it.

## Common Conventions

### 1. Path And Environment Variables

- **Input: source trajectories**
  - `HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5`
- **Output: replay recollection**
  - `OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos`
  - `OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets`
- **Input: replayed demos for replay/evaluation**
  - `REPLAYED_DEMOS_DIR=/path/to/replayed_demos`
  - Usually: `export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"`
- **Input: manually recorded demos**
  - Single file: `RECORDED_DEMOS_PATH=/path/to/single_recorded_demo.hdf5`
  - Directory mode: `RECORDED_DEMOS_DIR=/path/to/recorded_demos`

Keep input directories and output directories separate.

### 2. Batch Execution

- `replay_demos_with_camera.py` / `replay_demos.py`: if you pass multiple `--task_suite` values, or pass `--task_suite` without `--task_id`, the script launches child processes for each `(suite, task_id)` pair. This avoids repeatedly rebuilding Isaac/Kit environments in one process.
- `run_data_evaluations.py`: launches `replay_demos.py` for each task, parses stdout, and aggregates success/metric results until `max_episodes` is reached.
- `run_task_evaluations.py`: launches the OpenPI or other policy inference client for each task and parses stdout for success rate, per-experiment step counts, and force metrics.

### 3. Optional Libero Light Randomization

Libero DomeLight randomization is disabled by default. Add `--randomize_light` to replay or evaluation commands when you want reset-time randomization of intensity, color, and HDR sky texture.

```bash
python scripts/tools/replay_demos.py \
  --task Isaac-Libero-Franka-Replay-Camera-v0 \
  --task_suite libero_goal \
  --task_id 1 \
  --randomize_light \
  --headless
```

The CLI flag sets `LIBERO_RANDOMIZE_LIGHT=1` before the environment cfg is parsed. `EventCfgFrankaPanda` then registers the `randomize_light` reset event; tactile and contact-force Libero environments inherit the same event wiring from the base Franka Libero cfg.

Batch wrappers also accept the same flag and pass it to child processes:

```bash
python scripts/tools/run_task_evaluations.py \
  --policy_model openpi \
  --control_mode tactile \
  --task_suites libero_goal \
  --task_ids 1 \
  --randomize_light \
  --headless
```

### 4. Short Commands Versus Full Commands

- **Short commands** use minimal arguments and rely on the current shell environment/defaults.
- **Full commands** spell out important paths and common options, which is better for reproducibility and team use.

## 1. Data Collection

### `set_replay_env.sh`

Sets profile-based environment variables in the current shell.

Short command:

```bash
source scripts/tools/set_replay_env.sh tabero_force_gentle
```

Full/manual exports:

```bash
export HDF5_TRAJ_SOURCE_DIR=/path/to/libero/assembled_hdf5
export OUTPUT_REPLAYED_DEMOS_DIR=/path/to/output/replayed_demos
export OUTPUT_REPLAYED_VIDEOS_DIR=/path/to/output/video_datasets
export REPLAYED_DEMOS_DIR="$OUTPUT_REPLAYED_DEMOS_DIR"
export USE_TABERO_TASKS=0
```

### `replay_demos_with_camera.py`

Reads source trajectories, replays them in Isaac, and writes:

- `replayed_demos/*.hdf5` for successful episodes
- optional camera videos under `video_datasets/.../videos/*.mp4`
- optional tactile videos under `video_datasets/.../tactile_outputs/*.mp4`

`--recorder_type 7dpf` writes Force(6) into actions, producing 13D actions.

Short command:

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

Full command:

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

### `record_demos.py`

Records manual demos with keyboard or SpaceMouse and writes one HDF5 file.

For LIBERO tasks with `--task_suite` / `--task_id`, set `HDF5_TRAJ_SOURCE_DIR` first. SpaceMouse also requires the device to be connected; use `--teleop_device keyboard` if no SpaceMouse is available.

Short command:

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

Full command:

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

## 2. Data Evaluation

### `replay_demos.py`

Lightweight replay/validation for an HDF5 demo. Use it to:

- verify that a demo can replay correctly
- debug headless/camera/force observations
- optionally run `--validate_states` with `--num_envs 1`

Short command:

```bash
python scripts/tools/replay_demos.py --task Isaac-Libero-Franka-Replay-Camera-v0 --dataset_file /path/to/demo.hdf5 --demo_id 0
```

Full command:

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

### `run_data_evaluations.py`

Batch-evaluates replayed data quality from `REPLAYED_DEMOS_DIR`.

It resolves each task HDF5 from `REPLAYED_DEMOS_DIR`, launches `replay_demos.py` as a child process, parses stdout for success/Hybrid metrics, and writes JSON/TXT summaries.

Short command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/run_data_evaluations.py --control_mode tactile --headless
```

Full command:

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

### `run_task_evaluations.py`

Runs policy inference evaluation for each task, usually through the OpenPI client, then parses success rate plus structured per-experiment step and force metrics from stdout. On the OpenPI path, per-experiment summaries are always enabled and include both successful and failed experiments; the optional step-trace interface is OpenPI-only.

Short command:

```bash
python scripts/tools/run_task_evaluations.py --policy_model openpi --control_mode diffik --task_suites libero_goal --task_ids 1 --num_total_experiments 5 --headless
```

Full command:

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

`--record-step-traces` is optional and disabled by default. When it is enabled without `--step-trace-dir`, traces are written under
`<output_dir>/step_traces_<model>_<timestamp>/<suite>_task<ID>.jsonl`. Passing `--step-trace-dir` without enabling traces is rejected. This force-only JSONL is independent of `debug_mode=6` and never stores images.

Step semantics:

- `env_steps` counts only completed policy-control `env.step()` calls; reset and setup are excluded.
- `inference_chunks` increments once for each valid policy request. Therefore `env_steps <= inference_chunks * replan_steps`.
- Each JSONL row represents one `env.step()` and includes experiment/step/chunk indices, predicted and measured left/right local 3D finger forces, squeeze/ap values, and contact flags.

The JSON result keeps the existing task-level force fields unchanged and adds `metrics_status`, `metrics_warnings`, `step_statistics`, `force_metric_episode_counts`, `episodes`, and `step_trace`. `status` remains the success-rate status; incomplete metric records only change `metrics_status` to `partial`. The TXT result retains the task summary and adds per-experiment step/coverage and eight-force-metric tables, using `N/A` for unavailable values.

The eight force metrics remain success-only at task level. Each episode records the same fields for success or failure: `squeeze_avg_pred`, `squeeze_avg_meas`, `squeeze_max_pred`, `squeeze_max_meas`, `ap_avg_pred`, `ap_avg_meas`, `ap_max_pred`, and `ap_max_meas`. The `max` metrics are the mean of the largest 5% of nonzero-force frames, not a single-frame maximum. Tactile/hybrid expects full predicted and measured coverage; binary may omit predicted force; diffik/osc reports force as `not_applicable` while still recording steps.

### `export_mode6_five_panel_video.py`

Composes an existing OpenPI `debug_mode=6` experiment capture into one five-panel video:

1. agentview;
2. eye-in-hand;
3. left GelSight `markers_rgb`;
4. right GelSight `markers_rgb`;
5. stacked cumulative curves for raw model squeeze target / controller-effective target /
   measured squeeze and raw model `d_pred` / final controller `d_cmd` / measured `d_actual`.

This is offline post-processing: it does not start Isaac or the policy server and does not
modify the recorded evaluation. The four PNG streams and `forces.jsonl` must contain exactly
the same contiguous frame indices starting at zero. A successful episode may have one fewer
debug frame than `env_steps`; the exporter preserves that recorded boundary and never
synthesizes a terminal frame.

```bash
python scripts/tools/export_mode6_five_panel_video.py \
  --capture-exp-dir /path/to/capture_mode6/libero_object/task_0/none/<timestamp>/exp_000 \
  --step-trace /path/to/traces/libero_object_task0.jsonl \
  --output /path/to/combined_5view_force_gripper_curve.mp4 \
  --fps 20 \
  --width 1920 \
  --height 1080
```

The output is H.264/YUV420P. The tool verifies codec, pixel format, dimensions, frame rate,
and frame count with FFprobe, then writes `combined_5view_force_curve.manifest.json` by
default. Existing video or manifest files are rejected unless `--overwrite` is passed.

Force semantics are deliberately explicit. Mode-6 scalar schema v3 records raw model
`squeeze_pred`, controller-effective `squeeze_target_eff` after feed-forward/override,
controller EMA `squeeze_meas`, and unfiltered `squeeze_meas_raw`. The video plots the raw
model target in blue, effective controller target in red, and raw measurement in orange.
The raw measurement becomes `trajectory_mean_measured_squeeze` only after the separate
grasp/contact/reward-valid gate. For scalar-schema-v2 captures, the exporter defaults to all
three force series by auto-discovering `<run>/traces/<suite>_task<ID>.jsonl`; `--step-trace`
overrides that location. It verifies the trace and capture `squeeze_pred/squeeze_meas` values
frame by frame before using `effective_squeeze_target_n` and `reward_squeeze_meas_raw`. Export
fails rather than silently reverting to the historical EMA curve when the matching trace is
unavailable.

Gripper semantics are independently audited. Mode-6 scalar schema v2 records raw model
`d_pred`, final force-corrected and clamped `d_cmd`, both raw post-step finger joint positions,
and `d_actual=mean(q_left,q_right)`. By default the video plots `d_pred` in cyan, `d_cmd` in
green, and `d_actual` in magenta, in millimetres. The loader checks that `d_actual` equals the
two-joint mean and rejects legacy captures that averaged the signed policy observation
`[q_left,-q_right]`. Manifest schema v5 records these three plotted gripper series.

### `raw_data_retention_analysis.py`

Counts successful demos under each HDF5 in `REPLAYED_DEMOS_DIR` and compares the count with the expected number of episodes.

Short command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/raw_data_retention_analysis.py
```

Full command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos

python scripts/tools/raw_data_retention_analysis.py \
  --task_suites libero_10 libero_spatial libero_goal libero_object \
  --expected_episodes_per_file 50 \
  --output_dir ./evaluation_results
```

## 3. Visualization And Utilities

### `force_debug_playground.py`

Replays one demo and extracts local left/right finger forces from `obs["policy"]["gripper_net_force"]` as `(2, 3)` values for printing, plotting, or saving.

Use it with ContactForce / Tactile environments via `--env_variant`.

Short command:

```bash
export REPLAYED_DEMOS_DIR=/path/to/replayed_demos
python scripts/tools/force_debug_playground.py --env_variant contactforce --task_suite libero_goal --task_id 5 --demo_id 0 --save_plot ./force.png --headless
```

Full command:

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

### `upload_lerobot_to_hf.py`

Uploads a local LeRobot dataset to Hugging Face Hub. By default, it only uploads `data/` and `meta/`.

Short command:

```bash
python scripts/tools/upload_lerobot_to_hf.py --local-path /path/to/lerobot_dataset_root --repo-id your_username/your_dataset
```

Full command:

```bash
python scripts/tools/upload_lerobot_to_hf.py \
  --local-path /path/to/lerobot_dataset_root \
  --repo-id your_username/your_dataset \
  --repo-type dataset \
  --private
```

Omit `--private` for a public dataset. If the Hugging Face repository already exists and you do not want the script to create it, add `--no-create-repo`.
