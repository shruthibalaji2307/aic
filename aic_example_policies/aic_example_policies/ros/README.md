# ROS Policy Run Commands

Quick reference for launching the simulator and running each policy.

---

## Terminal 1 — Simulator + Engine

Always start this first. It launches Gazebo, the robot controller, the Zenoh
router, and `aic_engine` all in one go.

```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

### Use a custom config file

The engine defaults to `sample_config.yaml` (3 trials). Pass
`aic_engine_config_file` to use a different file.

> **Important:** the gripper finger width is baked into the URDF at launch time
> via `cable_type`. SFP and SC plugs have different widths, so **run SFP and SC
> trials in separate sessions** using the split configs below.

```bash
# Built-in sample (default — 3 trials: 2× SFP + 1× SC)
/entrypoint.sh ground_truth:=true start_aic_engine:=true

# 35 SFP trials (randomized board pose, NIC rail, translation, yaw)
/entrypoint.sh ground_truth:=true start_aic_engine:=true \
    aic_engine_config_file:=/home/shruthi/ws_aic/src/fork/aic/aic_engine/config/test_config_sfp.yaml

# 15 SC trials — must pass cable_type so gripper closes correctly on SC plug
/entrypoint.sh ground_truth:=true start_aic_engine:=true \
    cable_type:=sfp_sc_cable_reversed \
    aic_engine_config_file:=/home/shruthi/ws_aic/src/fork/aic/aic_engine/config/test_config_sc.yaml

# Absolute path to any custom config
/entrypoint.sh ground_truth:=true start_aic_engine:=true \
    aic_engine_config_file:=/path/to/my_config.yaml
```

> **Note:** `AIC_RESULTS_DIR` controls where `aic_engine` writes
> `scoring.yaml` and bag files (default: `~/aic_results`).
> Set it before running if you want results in a different location:
> ```bash
> export AIC_RESULTS_DIR=~/aic_results/my_run
> /entrypoint.sh ground_truth:=true start_aic_engine:=true ...
> ```

---

## Terminal 2 — Policy

All policy commands use `pixi run ros2 run aic_model aic_model`.
`use_sim_time:=true` is always required.

---

### CheatCodeBenchmark

Runs the unmodified CheatCode policy and prints a live score tally after each
trial plus a full tier-breakdown report at the end. Use this to establish a
CheatCode baseline to compare against `DataCollectorCheatCode`.

**Minimal (unlimited trials, no report):**
```bash
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCodeBenchmark
```

**Fixed number of episodes with report (typical usage):**
```bash
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCodeBenchmark \
    -p num_episodes:=3
```

**Full argument reference:**
```bash
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCodeBenchmark \
    -p num_episodes:=50         # episodes before printing report; -1 = unlimited (default: -1)
```

| Parameter | Default | Description |
|---|---|---|
| `num_episodes` | `-1` | Number of episodes to run before printing the report. `-1` = run forever, no report. |

**Scores come from `/rosout`** — no file dependency. Live per-trial totals
print immediately after each trial; the full T1/T2/T3 breakdown prints at the
end once `aic_engine` logs it.

---

### DataCollectorCheatCode

Runs CheatCode while recording observations and actions into a
[LeRobot](https://github.com/huggingface/lerobot) HuggingFace dataset.
Adds compliant insertion (force-triggered) and early-exit (seated detection)
on top of the base policy.

**Minimal (record unlimited episodes):**
```bash
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.DataCollectorCheatCode
```

**Record N episodes and print a dataset quality report:**
```bash
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.DataCollectorCheatCode \
    -p num_episodes:=10 \
    -p report_dataset_metrics:=true
```

**Full argument reference:**
```bash
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.DataCollectorCheatCode \
    -p num_episodes:=50                                    # default: -1 (unlimited) \
    -p dataset_repo_id:=local/cheatcode_demos              # default: "local/cheatcode_demos" \
    -p dataset_root:=/path/to/datasets                     # default: ~/.cache/huggingface/lerobot \
    -p image_scaling:=0.25                                 # default: 0.25  →  256×288 px \
    -p recording_fps:=20                                   # default: 20 Hz \
    -p report_dataset_metrics:=true                        # default: false \
    -p scoring_yaml_path:=/home/shruthi/aic_results/scoring.yaml  # default: ~/aic_results/scoring.yaml
```

| Parameter | Default | Description |
|---|---|---|
| `num_episodes` | `-1` | Episodes to record. `-1` = unlimited (policy keeps running but frames still saved). |
| `dataset_repo_id` | `"local/cheatcode_demos"` | HuggingFace dataset repo ID. `local/` prefix keeps it local. |
| `dataset_root` | `~/.cache/huggingface/lerobot` | Root directory for datasets. |
| `image_scaling` | `0.25` | Scale factor for camera images. `0.25` → 256×288 px per camera. |
| `recording_fps` | `20` | Recording framerate (must match the dataset's declared fps). |
| `report_dataset_metrics` | `false` | Print per-episode dataset quality report when `num_episodes` is reached. |
| `scoring_yaml_path` | `~/aic_results/scoring.yaml` | Path to `aic_engine`'s scoring file for backfilling scores into the report. |

**Recorded data per frame (32-dim state, 6-dim action, 3× RGB images):**
- State: TCP pose (7) + TCP velocity (6) + TCP error (6) + joint positions (7) + wrist wrench (6)
- Action: TCP linear velocity (x, y, z) + angular velocity (x, y, z)
- Images: `left_camera`, `center_camera`, `right_camera`

---

### Other Provided Policies (upstream, read-only)

These ship with the repo and run the same way — just swap the `policy` argument.

```bash
# Wave the arm (smoke test — no insertion)
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.WaveArm

# Run a trained ACT checkpoint
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.RunACT \
    -p checkpoint_path:=/path/to/checkpoint

# Wall-touch / wall-press (contact characterization)
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.WallToucher

pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.WallPresser

# Unmodified CheatCode (no scoring)
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCode
```

---

## Quick-Start: 3-Episode Benchmark Run

```bash
# Terminal 1
/entrypoint.sh ground_truth:=true start_aic_engine:=true

# Terminal 2 (once Terminal 1 is up)
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCodeBenchmark \
    -p num_episodes:=3
```

## Quick-Start: 35-Episode SFP Benchmark

```bash
# Terminal 1
/entrypoint.sh ground_truth:=true start_aic_engine:=true \
    aic_engine_config_file:=/home/shruthi/ws_aic/src/fork/aic/aic_engine/config/test_config_sfp.yaml

# Terminal 2
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCodeBenchmark \
    -p num_episodes:=35
```

## Quick-Start: 15-Episode SC Benchmark

```bash
# Terminal 1  (cable_type controls gripper finger width for SC plug)
/entrypoint.sh ground_truth:=true start_aic_engine:=true \
    cable_type:=sfp_sc_cable_reversed \
    aic_engine_config_file:=/home/shruthi/ws_aic/src/fork/aic/aic_engine/config/test_config_sc.yaml

# Terminal 2
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCodeBenchmark \
    -p num_episodes:=15
```

## Quick-Start: Collect 35 SFP Demo Episodes

```bash
# Terminal 1
/entrypoint.sh ground_truth:=true start_aic_engine:=true \
    aic_engine_config_file:=/home/shruthi/ws_aic/src/fork/aic/aic_engine/config/test_config_sfp.yaml

# Terminal 2
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.DataCollectorCheatCode \
    -p num_episodes:=35 \
    -p dataset_repo_id:=local/cheatcode_demos \
    -p report_dataset_metrics:=true
```
