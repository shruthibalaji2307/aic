# CLAUDE.md — ROS Example Policies

## Scope of work

**Only these files in this directory are user-authored and subject to modification:**
- `DataCollectorCheatCode.py`
- `CheatCodeBenchmark.py`
- `admittance.py`

All other `.py` files in this directory (`CheatCode.py`, `WaveArm.py`, `RunACT.py`, `GentleGiant.py`, `SpeedDemon.py`, `WallPresser.py`, `WallToucher.py`, `__init__.py`) are provided by the upstream `aic` repo and **must not be touched**.

---

## Repository context

This directory lives inside the [AI for Industry Challenge (AIC)](https://github.com/anthropics/aic) toolkit at:

```
aic/
└── aic_example_policies/
    └── aic_example_policies/
        └── ros/          ← you are here
```

The toolkit is a robotics competition framework for cable-insertion tasks. The main packages relevant to policy development are:

| Package | Purpose |
|---|---|
| `aic_model` | Base `Policy` class all policies inherit from |
| `aic_interfaces` | ROS message types (`Task`, `Observation`, `MotionUpdate`, etc.) |
| `aic_example_policies` | Example/baseline policies (this package) |

---

## Policy pattern

Every policy inherits from `aic_model.policy.Policy` and implements `insert_cable()`:

```python
from aic_model.policy import Policy

class MyPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        ...
        return True  # True = success
```

### Key base-class methods
- `self.get_logger()` — ROS logger
- `self.sleep_for(sec)` — simulation-aware sleep
- `self.set_pose_target(move_robot, pose, frame_id="base_link")` — Cartesian pose command
- `self.set_cartesian_twist_target(twist)` — velocity command
- `self._parent_node._tf_buffer.lookup_transform(...)` — TF frame lookup

### Callback contracts
| Callback | Returns | Notes |
|---|---|---|
| `get_observation()` | `Observation \| None` | images, TCP pose/vel/error, joint states, wrench |
| `move_robot(motion_update=..., joint_motion_update=...)` | — | send Cartesian or joint motion targets |
| `send_feedback(str)` | — | status string visible in the evaluation UI |

### Observation fields (key ones)
```
obs.controller_state.tcp_pose          # geometry_msgs/Pose
obs.controller_state.tcp_velocity      # geometry_msgs/Twist
obs.controller_state.tcp_error         # float[6]
obs.joint_states.position              # float[7]
obs.wrist_wrench.wrench.force          # geometry_msgs/Vector3
obs.wrist_wrench.wrench.torque         # geometry_msgs/Vector3
obs.center_image / left_image / right_image  # sensor_msgs/Image
```

### Impedance parameters (passed inside `MotionUpdate`)
- `stiffness`: 6-dim `[X, Y, Z, Rx, Ry, Rz]`
- `damping`: 6-dim `[X, Y, Z, Rx, Ry, Rz]`
- `wrench_feedback_gains`: 6-dim — typically `[0.5, 0.5, 0.5, 0, 0, 0]`

---

## File summaries

### `DataCollectorCheatCode.py`
Extends `CheatCode` (repo-provided) to record demonstration data into a [LeRobot](https://github.com/huggingface/lerobot) HuggingFace dataset with an improved contact-triggered insertion strategy.

#### Insertion strategy (4 phases, all recorded at 20 Hz)

1. **Approach** – `_compute_aligned_pose()` + SLERP over 100 steps (5 s) to `APPROACH_Z_OFFSET = 0.2 m` above the port.  Pure feedforward from TF: no integrators.
2. **Descent** – straight-down at `DESCENT_STEP_M = 0.0005 m/step` until `|force| > CONTACT_FORCE_THRESHOLD = 3.0 N` (or `z_offset < −15 mm`).  Compliant impedance activates automatically below `COMPLIANT_Z_OFFSET_GATE = 0.05 m`.
3. **Rotational search** (`_rotational_search()`) – once contact is detected, the commanded XY traces a growing spiral centred on the port (radius 0 → `SEARCH_MAX_RADIUS = 2 mm` over `SEARCH_MAX_REVOLUTIONS = 5` revolutions, `SEARCH_ANGLE_STEP = 0.2 rad`).  Compliant Z-axis keeps the plug pressed against the surface; when the tip aligns with the slot it drops in naturally.  The 2 mm radius bound keeps the plug on the circuit board.
4. **Stabilise** – 2 s hold for physics and aic_engine scoring to settle.

#### Why this approach beats CheatCode baseline
- CheatCode relies entirely on GT XY alignment; any GT error → plug never finds slot.
- The rotational search explores a small area around the GT position, tolerating small errors.
- Compliant impedance means contact forces guide insertion rather than fighting them.

#### Key constants
| Constant | Value | Notes |
|---|---|---|
| `APPROACH_Z_OFFSET` | `0.2` m | Hover height above port |
| `DESCENT_STEP_M` | `0.0005` m | 10 mm/s at 20 Hz |
| `CONTACT_FORCE_THRESHOLD` | `3.0` N | Contact detection |
| `COMPLIANT_Z_OFFSET_GATE` | `0.05` m | Enable compliant below this |
| `SEARCH_MAX_RADIUS` | `0.002` m | Board-safe XY bound |
| `SEARCH_RADIUS_STEP` | `0.0004` m/rev | Spiral growth rate |
| `SEARCH_ANGLE_STEP` | `0.2` rad | ~11.5° per step |
| `SEARCH_MAX_REVOLUTIONS` | `5` | Max spiral revolutions |

#### `_compute_aligned_pose()` vs `CheatCode.calc_gripper_pose()`
`_compute_aligned_pose()` is pure feedforward (gripper–plug-tip offset from TF, no integrators).  CheatCode's version adds XY integrators to compensate for persistent GT error — this was removed because the rotational search handles that correction instead.

ROS parameters declared in `__init__`:
| Parameter | Default | Description |
|---|---|---|
| `dataset_repo_id` | `"local/cheatcode_demos"` | HuggingFace dataset id |
| `dataset_root` | `~/.cache/huggingface/lerobot` | Local dataset path |
| `image_scaling` | `0.25` | Downscale factor for images |
| `recording_fps` | `20` | Target recording framerate |
| `num_episodes` | `-1` | Max episodes (-1 = unlimited) |
| `report_dataset_metrics` | `False` | Print per-frame dataset stats |
| `scoring_yaml_path` | `~/aic_results/scoring.yaml` | Path to backfill final scores |

State vector (32 dims): TCP pose (7) + TCP velocity (6) + TCP error (6) + joint positions (7) + wrench (6).
Action vector (6 dims): linear velocity (x, y, z) + angular velocity (x, y, z).

### `CheatCodeBenchmark.py`
Extends `CheatCode` (repo-provided) with no behavioural changes — it delegates `insert_cable()` to `super()` unmodified. Its only additions are score tracking and a terminal report at the end of a run, intended to establish a quantified CheatCode baseline for comparison against `DataCollectorCheatCode`.

Key behaviours:
- **Score capture**: subscribes to `/rosout` and captures per-trial total scores from the `✓ Trial 'trial_N' completed successfully! Score: X` messages that aic_engine logs between trials (while the lifecycle node is still active).
- **Benchmark report**: prints a colour-coded table with per-trial total scores, average, range, and high-score rate (>80).
- **Report saved to pwd**: writes an ANSI-stripped copy to `cheatcode_report_<run_id>.txt` in the working directory.
- **Auto-exit**: fires `os._exit(0)` after 2 s report delay + 1 s flush (10 s safety timer), matching DataCollectorCheatCode behaviour.

ROS parameters:
| Parameter | Default | Description |
|---|---|---|
| `num_episodes` | `-1` | Episodes to run before printing report (-1 = unlimited, no report) |

Usage:
```bash
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCodeBenchmark \
    -p num_episodes:=10
```

### `admittance.py`
A standalone admittance controller built on top of `pyatk.tools.Admittance`. Wraps the PyATK low-level impedance loop and exposes a clean async `start()` interface.

```python
class AdmittanceController:
    def __init__(self, robot, force_sensor, world_T_tcp, user_cb_func):
        ...
    async def start(self):
        ...
```

The user callback signature:
```python
def user_cb_func(controller, force, moment, world_T_tcp, elapsed) -> bool:
    # return True to stop the controller
```

Hardware-aware: uses slower M/kp/max_v parameters when running on a real robot controller vs. simulation.

---

## Quaternion conventions

- `transforms3d` quaternion format: `(w, x, y, z)`
- `geometry_msgs/Quaternion` format: `(x, y, z, w)`
- SLERP via `transforms3d._gohlketransforms.quaternion_slerp`
- Composition via `transforms3d._gohlketransforms.quaternion_multiply`
