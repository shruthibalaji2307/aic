# aic_data_collection_pipeline

Utility package to automate qualification-style task config generation and run a
LeRobot recording workflow around the existing commands from
`aic_utils/lerobot_robot_aic/README.md`.

This package does not modify `aic_utils`; it orchestrates existing tools:

- **`aic-generate-qualification-config`** — writes randomized engine YAML from `aic_engine/config/sample_config.yaml`.
- **`aic-run-lerobot-pipeline`** — full host-colcon mode *or* **`--eval-in-container`** when sim runs in **`aic_eval`**.

Typical split:

- **Container:** Zenoh + Gazebo + `aic_engine` via **`/entrypoint.sh`**.
- **Host (Pixi):** `aic_model` (CheatCode) and **`lerobot-record`**.

## Recommended workflow: Distrobox + `aic_eval` (authors’ default)

The [Getting Started](../docs/getting_started.md) flow runs **Gazebo + Zenoh + engine** inside the **`ghcr.io/intrinsic-dev/aic/aic_eval`** image. Your **host** Pixi environment is used for **`aic_model`** and **`lerobot-record`**, which talk to the container over **Zenoh** (Pixi’s `pixi_env_setup.sh` already sets `RMW_IMPLEMENTATION` and `ZENOH_CONFIG_OVERRIDE` for that).

The container **`/entrypoint.sh`** forwards all arguments to `ros2 launch aic_bringup aic_gz_bringup.launch.py`, so you can pass a generated engine config with `aic_engine_config_file:=...`. Use an **absolute path** that exists **inside** the container. With Distrobox, your home directory is usually the same path as on the host (e.g. `/home/you/...`).

### Sequential commands (smoke test: one random trial + record)

**Terminal A — host:** generate config (same repo path you already use):

```bash
cd ~/ws_aic/src/aic
pixi run aic-generate-qualification-config \
  --template-config ~/ws_aic/src/aic/aic_engine/config/sample_config.yaml \
  --output-config ~/ws_aic/src/aic/tmp/qualification_random_1.yaml \
  --seed 7 \
  --num-trials 1 \
  --mode random
```

**Terminal B — inside Distrobox `aic_eval`:** start sim + engine with **ground truth** (needed for CheatCode) and your YAML:

```bash
export DBX_CONTAINER_MANAGER=docker   # if not already set
distrobox enter -r aic_eval

/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/YOUR_USER/ws_aic/src/aic/tmp/qualification_random_1.yaml
```

Replace `YOUR_USER` with your Linux username. Use the **same** absolute path you used in Terminal A.

**Terminal C — host:** record with LeRobot (simulation is already running; **do not** let this script start Gazebo):

**`--dataset-repo-id`** is optional with **`--eval-in-container`**: if omitted, the pipeline uses **`local/aic_cable_insert`** (disk-only). Pass **`--dataset-repo-id your_user/your-dataset`** when you want a specific Hub id or name. **Do not** use `<...>` in bash.

```bash
cd ~/ws_aic/src/aic
pixi run aic-run-lerobot-pipeline \
  --workspace-dir ~/ws_aic/src/aic \
  --engine-config ~/ws_aic/src/aic/tmp/qualification_random_1.yaml \
  --eval-in-container \
  --launch-cheatcode-on-host \
  --dataset-repo-id MY_HF_USERNAME/aic-cable-insert-smoke \
  --dataset-single-task "insert cable plug into target port" \
  --teleop-type aic_keyboard_ee \
  --robot-teleop-target-mode cartesian \
  --robot-teleop-frame-id base_link \
  --dataset-private \
  --startup-wait-sec 8
```

- `--eval-in-container`: only runs `lerobot-record` on the host (no `aic_bringup` in Pixi).
- `--launch-cheatcode-on-host`: starts CheatCode via `pixi run ros2 run aic_model ...` before recording so the engine sees `aic_model` within its timeout.

If you prefer to start CheatCode yourself (as in Getting Started step 3), omit `--launch-cheatcode-on-host` and run:

```bash
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
```

before `lerobot-record`, then use only `--eval-in-container` on the pipeline.

### Task board / cable never spawn, but `lerobot-record` works

`lerobot-record` only needs camera + controller topics on the host ROS graph. **Spawning the board and cable is entirely driven by `aic_engine` inside the container**, after it successfully loads your YAML and finds the participant **`aic_model`** lifecycle node. If those steps fail, you can still get images (sometimes black/stale) while nothing meaningful happens in sim.

Checklist:

1. **`aic_engine_config_file` must exist inside the container** at the path you pass. It must be the **same machine user** as your real home (Distrobox usually mirrors `/home/<you>/...`). A path like `/home/pvrohin/...` will **not** work if your user is `untangled`. Verify:

   ```bash
   # inside aic_eval
   echo $HOME
   ls -la /home/YOUR_USER/ws_aic/src/aic/tmp/qualification_random_1.yaml
   ```

2. **Start order and timeout:** Launch **`/entrypoint.sh` first** and wait until Gazebo is up and the engine is waiting for the model. Then start **`aic_model` on the host within the engine discovery window** (see Getting Started; the engine retries waiting for `aic_model`). Use **`--launch-cheatcode-on-host`** or start `aic_model` manually **before** the engine gives up.

3. **One ROS graph:** Host and container must share the same **Zenoh** router (the entrypoint starts `rmw_zenohd`). Run host commands from **`cd .../ws_aic/src/aic` + `pixi run ...`** so `pixi_env_setup.sh` sets `RMW_IMPLEMENTATION` / Zenoh peer settings. If `aic_model` is not visible to the engine, trials won’t start.

4. **Sanity checks (host, Pixi env):**

   ```bash
   pixi run ros2 node list | grep aic_model
   ```

   After the engine should start a trial, you should see logs in the **container** terminal about trials / task board. If the YAML path was wrong, fix it and restart `/entrypoint.sh`.

**During `lerobot-record`:** Right Arrow = next episode, Left Arrow = redo episode, ESC = stop.

**Outputs:** Engine/scoring artifacts go under `$HOME/aic_results/` in the environment that runs the engine (often the container’s home if that is where you launched `/entrypoint.sh`). Dataset files follow LeRobot’s usual cache / Hub behavior.

### Local dataset path (LeRobot)

For `repo_id` like `pvrohin/sample_dataset`, LeRobot stores data under:

`~/.cache/huggingface/lerobot/pvrohin/sample_dataset`

### If you see `FileExistsError` on that folder

A **second** recording with the same `--dataset-repo-id` tries to **create** a new dataset at the same path and fails. Either:

- **Append** to the existing run (recommended):

  ```bash
  ... aic-run-lerobot-pipeline ... --lerobot-resume
  ```

- **Or** delete the folder and record fresh:

  ```bash
  rm -rf ~/.cache/huggingface/lerobot/pvrohin/sample_dataset
  ```

If `lerobot-record` crashed after a failed `create()`, LeRobot may still try `push_to_hub` with a missing `dataset` object; fix the folder issue first, then re-run.

### If you see `FileNotFoundError: ... meta/info.json` or `RevisionNotFoundError` when using `--lerobot-resume`

That means the directory exists under `~/.cache/huggingface/lerobot/<repo_id>/` but **no successful dataset was ever written** (e.g. a failed `create()` left an empty or partial folder). **`--lerobot-resume` only works when `meta/info.json` exists.**

Remove the broken directory and record again **without** `--lerobot-resume`:

```bash
rm -rf ~/.cache/huggingface/lerobot/pvrohin/sample_dataset
# then run aic-run-lerobot-pipeline WITHOUT --lerobot-resume
```

The pipeline now checks this before starting `lerobot-record` and prints the same hint.

### Noisy `aic_model` / TF errors when stopping

When `lerobot-record` exits first, the pipeline stops `aic_model` with SIGINT; you may still see rclpy/TF thread tracebacks during teardown. That is usually harmless. The pipeline waits longer on SIGINT before SIGTERM/kill.

---

## Advanced: colcon evaluation stack on the host (no container)

If you built the full workspace from [Building from Source](../docs/build_eval.md), you can run the **all-in-one** pipeline with:

```bash
--ros-setup-bash /path/to/your/colcon/install/setup.bash
```

That mode runs Zenoh, `aic_model`, `ros2 launch aic_bringup ...`, and `pixi run lerobot-record` from your machine. The default **Pixi-only** workspace does **not** include `aic_bringup`; without `--ros-setup-bash` or `--eval-in-container`, the script exits with a hint.

---

## Important note about action labeling

`lerobot-record` is fundamentally teleop-driven. This workflow is useful and
convenient for data collection, but your recorded action labels are generated by
the active teleop stream. If you want guaranteed "policy action" supervision
for autonomous runs, use a policy-action recorder/converter path instead of
teleop action labels.

---

## Sequential commands: install and test this package

1) Workspace root:

```bash
cd ~/ws_aic/src/aic
```

2) Dependencies:

```bash
pixi install
```

3) Install this package into the Pixi env:

```bash
pixi reinstall ros-kilted-aic-data-collection-pipeline
```

4) Check CLIs:

```bash
pixi run aic-generate-qualification-config --help
pixi run aic-run-lerobot-pipeline --help
```

---

## Push dataset to Hugging Face

```bash
pixi run hf auth login
```

Then add `--dataset-push-to-hub` to `aic-run-lerobot-pipeline` (with the same Distrobox / `--eval-in-container` flags you use for local recording).

---

## Scale up to many randomized tasks

Generate configs:

```bash
pixi run aic-generate-qualification-config \
  --template-config ~/ws_aic/src/aic/aic_engine/config/sample_config.yaml \
  --output-config ~/ws_aic/src/aic/tmp/qualification_random_50.yaml \
  --seed 123 \
  --num-trials 50 \
  --mode alternating
```

Restart **`/entrypoint.sh`** in `aic_eval` with `aic_engine_config_file:=` pointing at that YAML (or split into multiple runs). On the host, run recording with `--eval-in-container` as above for each session.
