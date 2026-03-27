#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Data-collecting wrapper around CheatCode with compliant insertion.

Runs CheatCode (ground-truth cable insertion) while recording every
observation and the corresponding robot velocity into a LeRobot dataset.
Adds a contact-triggered rotational search on top of the base policy so
the plug can reliably find the slot even when the GT pose has small error.

Insertion strategy
------------------
1. **Approach** – smooth SLERP from current position to ``APPROACH_Z_OFFSET``
   above the port (identical to CheatCode).
2. **Descend** – 1 s settle at hover + port TF re-fetch, then straight-down
   at ``DESCENT_STEP_M`` per step.  CheatCode's XY integrators accumulate
   plug-tip→port error and correct commanded XY on every step.  Exits when
   baseline-subtracted contact force exceeds ``CONTACT_FORCE_THRESHOLD``
   or ``z_offset < -0.015``.
3. **Rotational search** – once contact is detected the impedance params
   switch to compliant (low stiffness + wrench feedback) and the commanded
   XY traces a growing spiral (max ``SEARCH_MAX_RADIUS = 2 mm``) around the
   port centre.  The search radius is bounded so the plug can never wander
   off the circuit board.  The compliant Z-axis keeps the plug pressed
   against the surface; when the tip aligns with the slot it drops in.
4. **Stabilise** – 2 s wait for physics to settle; aic_engine scores here.

Usage (with the sim launched with ``ground_truth:=true``):

    pixi run ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=aic_example_policies.ros.DataCollectorCheatCode

ROS parameters (set via ``--ros-args -p <name>:=<value>``):
    dataset_repo_id        – local repo-id for the dataset
                             (default: "local/cheatcode_demos")
    dataset_root           – root directory where datasets are stored
                             (default: ~/.cache/huggingface/lerobot)
    image_scaling          – scale factor applied to camera images before saving
                             (default: 0.25, → 256×288 px per camera)
    recording_fps          – target recording FPS, must match dataset fps
                             (default: 20)
    num_episodes           – number of episodes to record; -1 for unlimited
                             (default: -1)
    report_dataset_metrics – print a formatted dataset quality report after
                             all episodes are collected (requires num_episodes > 0)
                             (default: False)
    scoring_yaml_path      – path to aic_engine's scoring.yaml for the report
                             (default: ~/aic_results/scoring.yaml)

Example – record 10 episodes and show the report at the end::

    pixi run ros2 run aic_model aic_model --ros-args \\
        -p use_sim_time:=true \\
        -p policy:=aic_example_policies.ros.DataCollectorCheatCode \\
        -p num_episodes:=10 \\
        -p report_dataset_metrics:=true
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
import cv2
import numpy as np

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3, Wrench
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from .CheatCode import CheatCode

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1152

OBSERVATION_STATE_NAMES = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.x",
    "tcp_pose.orientation.y",
    "tcp_pose.orientation.z",
    "tcp_pose.orientation.w",
    "tcp_velocity.linear.x",
    "tcp_velocity.linear.y",
    "tcp_velocity.linear.z",
    "tcp_velocity.angular.x",
    "tcp_velocity.angular.y",
    "tcp_velocity.angular.z",
    "tcp_error.x",
    "tcp_error.y",
    "tcp_error.z",
    "tcp_error.rx",
    "tcp_error.ry",
    "tcp_error.rz",
    "joint_positions.0",
    "joint_positions.1",
    "joint_positions.2",
    "joint_positions.3",
    "joint_positions.4",
    "joint_positions.5",
    "joint_positions.6",
    "wrench.force.x",
    "wrench.force.y",
    "wrench.force.z",
    "wrench.torque.x",
    "wrench.torque.y",
    "wrench.torque.z",
]

ACTION_NAMES = [
    "linear.x",
    "linear.y",
    "linear.z",
    "angular.x",
    "angular.y",
    "angular.z",
]


def _build_features(image_h: int, image_w: int) -> dict:
    """Build the LeRobot features dict matching AICRobotAICController."""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(OBSERVATION_STATE_NAMES),),
            "names": list(OBSERVATION_STATE_NAMES),
        },
        "observation.images.left_camera": {
            "dtype": "video",
            "shape": (image_h, image_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.center_camera": {
            "dtype": "video",
            "shape": (image_h, image_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_camera": {
            "dtype": "video",
            "shape": (image_h, image_w, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": list(ACTION_NAMES),
        },
    }


def _ros_image_to_numpy(raw_img, scale: float) -> np.ndarray:
    """Convert a sensor_msgs/Image to a scaled uint8 numpy array (H, W, 3)."""
    img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
        raw_img.height, raw_img.width, 3
    )
    if scale != 1.0:
        img_np = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img_np


def _extract_state(obs: Observation) -> np.ndarray:
    """Extract the 32-dim state vector from an Observation message."""
    tcp_pose = obs.controller_state.tcp_pose
    tcp_vel = obs.controller_state.tcp_velocity
    wrench = obs.wrist_wrench.wrench
    return np.array(
        [
            tcp_pose.position.x,
            tcp_pose.position.y,
            tcp_pose.position.z,
            tcp_pose.orientation.x,
            tcp_pose.orientation.y,
            tcp_pose.orientation.z,
            tcp_pose.orientation.w,
            tcp_vel.linear.x,
            tcp_vel.linear.y,
            tcp_vel.linear.z,
            tcp_vel.angular.x,
            tcp_vel.angular.y,
            tcp_vel.angular.z,
            *obs.controller_state.tcp_error,
            *obs.joint_states.position[:7],
            wrench.force.x,
            wrench.force.y,
            wrench.force.z,
            wrench.torque.x,
            wrench.torque.y,
            wrench.torque.z,
        ],
        dtype=np.float32,
    )


def _extract_action(obs: Observation) -> np.ndarray:
    """Extract the 6-dim velocity action from the robot's observed TCP velocity."""
    tcp_vel = obs.controller_state.tcp_velocity
    return np.array(
        [
            tcp_vel.linear.x,
            tcp_vel.linear.y,
            tcp_vel.linear.z,
            tcp_vel.angular.x,
            tcp_vel.angular.y,
            tcp_vel.angular.z,
        ],
        dtype=np.float32,
    )


class DataCollectorCheatCode(CheatCode):
    """CheatCode with compliant rotational-search insertion and LeRobot recording.

    Insertion phases
    ----------------
    approach  – SLERP from current TCP to APPROACH_Z_OFFSET above port (5 s)
    descent   – straight-down at DESCENT_STEP_M/step; stops when contact force
                exceeds CONTACT_FORCE_THRESHOLD or z_offset falls below −15 mm
    search    – spiral XY pattern (compliant impedance) bounded by
                SEARCH_MAX_RADIUS = 2 mm so the plug stays on the board
    stabilise – 2 s for physics/scoring to settle

    Recording
    ---------
    One frame per get_observation() call at each loop step (~20 Hz).
    State: 32-dim (TCP pose/vel/error + joints + wrench)
    Action: 6-dim TCP velocity
    Images: left, centre, right cameras (scaled by image_scaling)
    """

    # ── Insertion tuning ─────────────────────────────────────────────────────
    APPROACH_Z_OFFSET = 0.2          # m above port face for initial hover
    DESCENT_STEP_M = 0.0005          # m per descent step (20 Hz → 10 mm/s)
    CONTACT_FORCE_THRESHOLD = 8.0    # N – triggers compliant / search mode
    XY_I_GAIN = 0.15                 # integrator gain (matches CheatCode)

    # ── Rotational search ────────────────────────────────────────────────────
    SEARCH_MAX_RADIUS = 0.002        # m from port centre (board-safe boundary)
    SEARCH_RADIUS_STEP = 0.0004      # m radius growth per full revolution
    SEARCH_ANGLE_STEP = 0.2          # rad (~11.5°) per step
    SEARCH_MAX_REVOLUTIONS = 5

    # ── Wrench baseline ──────────────────────────────────────────────────────
    WRENCH_BASELINE_SAMPLES = 10
    GRIPPER_PAYLOAD_MASS = 1.0       # kg
    GRAVITY = 9.81                   # m/s²

    # ── Compliant impedance params ───────────────────────────────────────────
    COMPLIANT_STIFFNESS = np.diag([80.0, 80.0, 120.0, 40.0, 40.0, 40.0])
    COMPLIANT_DAMPING   = np.diag([50.0, 50.0,  50.0, 20.0, 20.0, 20.0])
    COMPLIANT_WRENCH_GAINS = [0.5, 0.5, 0.3, 0.2, 0.2, 0.2]

    def __init__(self, parent_node):
        super().__init__(parent_node)

        parent_node.declare_parameter("dataset_repo_id", "local/cheatcode_demos")
        parent_node.declare_parameter("dataset_root", "")
        parent_node.declare_parameter("image_scaling", 0.25)
        parent_node.declare_parameter("recording_fps", 20)
        parent_node.declare_parameter("num_episodes", -1)
        parent_node.declare_parameter("report_dataset_metrics", False)
        parent_node.declare_parameter(
            "scoring_yaml_path",
            str(Path.home() / "aic_results" / "scoring.yaml"),
        )

        self._repo_id: str = (
            parent_node.get_parameter("dataset_repo_id").get_parameter_value().string_value
        )
        root_str: str = (
            parent_node.get_parameter("dataset_root").get_parameter_value().string_value
        )
        self._root: Path | None = Path(root_str) if root_str else None
        self._image_scaling: float = (
            parent_node.get_parameter("image_scaling").get_parameter_value().double_value
        )
        self._recording_fps: int = (
            parent_node.get_parameter("recording_fps").get_parameter_value().integer_value
        )
        self._num_episodes: int = (
            parent_node.get_parameter("num_episodes").get_parameter_value().integer_value
        )
        self._report_metrics: bool = (
            parent_node.get_parameter("report_dataset_metrics").get_parameter_value().bool_value
        )
        self._scoring_yaml_path: Path = Path(
            parent_node.get_parameter("scoring_yaml_path").get_parameter_value().string_value
        )

        scaled_h = int(IMAGE_HEIGHT * self._image_scaling)
        scaled_w = int(IMAGE_WIDTH * self._image_scaling)
        self._features = _build_features(scaled_h, scaled_w)

        self._dataset = None
        self._run_id = time.strftime("%Y%m%d_%H%M%S")
        self._trial_index = 0
        self._cached_scoring: dict[str, Any] = {}

        # Per-trial state (reset by _reset_trial_state)
        self._last_wrench_force = np.zeros(3)
        self._wrench_baseline = np.zeros(3)
        self._wrench_baseline_samples: list[np.ndarray] = []
        self._wrench_baseline_ready = False
        self._in_compliant_mode = False
        self._insertion_phase = "idle"
        self._current_z_offset = self.APPROACH_Z_OFFSET

    # ── Dataset management ────────────────────────────────────────────────────

    def _ensure_dataset(self) -> None:
        """Lazily create or resume the dataset on first use."""
        if self._dataset is not None:
            return

        import shutil
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = (self._root / self._repo_id) if self._root is not None else None
        dataset_dir = root if root is not None else (
            Path.home() / ".cache" / "huggingface" / "lerobot" / self._repo_id
        )

        can_resume = False
        info_path = dataset_dir / "meta" / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            existing_shape = tuple(
                info.get("features", {}).get("observation.state", {}).get("shape", [])
            )
            expected_shape = self._features["observation.state"]["shape"]
            if existing_shape == expected_shape:
                can_resume = True
            else:
                self.get_logger().warn(
                    f"Existing dataset has observation.state shape {existing_shape}, "
                    f"expected {expected_shape}. Removing incompatible dataset."
                )
                shutil.rmtree(dataset_dir)

        if can_resume:
            try:
                self._dataset = LeRobotDataset(repo_id=self._repo_id, root=root)
                self._dataset.start_image_writer(num_processes=0, num_threads=4)
                self.get_logger().info(
                    f"Resumed LeRobot dataset '{self._repo_id}' at {self._dataset.root} "
                    f"({self._dataset.meta.total_episodes} existing episodes)"
                )
            except Exception as exc:
                self.get_logger().warn(
                    f"Could not resume dataset: {exc}. Creating fresh."
                )
                shutil.rmtree(dataset_dir)
                can_resume = False

        if not can_resume:
            create_kwargs: dict = {
                "repo_id": self._repo_id,
                "fps": self._recording_fps,
                "features": self._features,
                "robot_type": "ur5e_aic",
                "use_videos": True,
                "image_writer_processes": 0,
                "image_writer_threads": 4,
            }
            if root is not None:
                create_kwargs["root"] = root
            self._dataset = LeRobotDataset.create(**create_kwargs)
            self.get_logger().info(
                f"Created LeRobot dataset '{self._repo_id}' at {self._dataset.root}"
            )

    # ── Pose calculation ──────────────────────────────────────────────────────

    def _compute_aligned_pose(
        self,
        port_transform: Transform,
        cable_tip_frame: str,
        z_offset: float = 0.2,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
    ) -> Pose:
        """Compute the gripper pose that places the plug tip at the port.

        Pure feedforward from TF: offset = gripper_xyz − plug_tip_xyz.
        Target gripper position = port_xyz + [0, 0, z_offset] + offset.
        Orientation: SLERP from current gripper orientation toward the
        orientation that aligns the plug frame with the port frame.
        """
        tf_buffer = self._parent_node._tf_buffer
        plug_tf = tf_buffer.lookup_transform("base_link", cable_tip_frame, Time())
        gripper_tf = tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())

        plug_xyz = np.array([
            plug_tf.transform.translation.x,
            plug_tf.transform.translation.y,
            plug_tf.transform.translation.z,
        ])
        gripper_xyz = np.array([
            gripper_tf.transform.translation.x,
            gripper_tf.transform.translation.y,
            gripper_tf.transform.translation.z,
        ])
        port_xyz = np.array([
            port_transform.translation.x,
            port_transform.translation.y,
            port_transform.translation.z,
        ])

        offset = gripper_xyz - plug_xyz
        target_xyz = port_xyz + np.array([0.0, 0.0, z_offset]) + offset
        blend_xyz = position_fraction * target_xyz + (1.0 - position_fraction) * gripper_xyz

        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        q_plug = (
            plug_tf.transform.rotation.w,
            plug_tf.transform.rotation.x,
            plug_tf.transform.rotation.y,
            plug_tf.transform.rotation.z,
        )
        q_gripper = (
            gripper_tf.transform.rotation.w,
            gripper_tf.transform.rotation.x,
            gripper_tf.transform.rotation.y,
            gripper_tf.transform.rotation.z,
        )
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        q_target = quaternion_multiply(q_diff, q_gripper)
        q_result = quaternion_slerp(q_gripper, q_target, slerp_fraction)

        return Pose(
            position=Point(x=float(blend_xyz[0]), y=float(blend_xyz[1]), z=float(blend_xyz[2])),
            orientation=Quaternion(w=q_result[0], x=q_result[1], y=q_result[2], z=q_result[3]),
        )

    # ── Compliant mode ────────────────────────────────────────────────────────

    def _should_use_compliant_mode(self) -> bool:
        """Return True when compliant impedance params should be used.

        Always True during the rotational search phase.  During descent,
        only activates below COMPLIANT_Z_OFFSET_GATE with baseline ready
        and contact force exceeding threshold.
        """
        if self._insertion_phase == "search":
            return True
        if self._insertion_phase != "descent":
            return False
        if not self._wrench_baseline_ready:
            return False
        return float(np.linalg.norm(self._last_wrench_force)) > self.CONTACT_FORCE_THRESHOLD

    def set_pose_target(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        frame_id: str = "base_link",
    ) -> None:
        """Send a pose target, switching to compliant impedance when contact detected."""
        if self._should_use_compliant_mode():
            force_mag = float(np.linalg.norm(self._last_wrench_force))
            if not self._in_compliant_mode:
                self.get_logger().info(
                    f"Contact detected ({force_mag:.1f} N, "
                    f"z_off={self._current_z_offset:.4f}), switching to compliant mode"
                )
                self._in_compliant_mode = True

            motion_update = MotionUpdate(
                header=Header(
                    frame_id=frame_id,
                    stamp=self._parent_node.get_clock().now().to_msg(),
                ),
                pose=pose,
                target_stiffness=self.COMPLIANT_STIFFNESS.flatten(),
                target_damping=self.COMPLIANT_DAMPING.flatten(),
                feedforward_wrench_at_tip=Wrench(
                    force=Vector3(x=0.0, y=0.0, z=0.0),
                    torque=Vector3(x=0.0, y=0.0, z=0.0),
                ),
                wrench_feedback_gains_at_tip=self.COMPLIANT_WRENCH_GAINS,
                trajectory_generation_mode=TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION,
                ),
            )
            try:
                move_robot(motion_update=motion_update)
            except Exception as ex:
                self.get_logger().info(f"move_robot exception: {ex}")
        else:
            if self._in_compliant_mode:
                self.get_logger().info(
                    f"Contact lost ({float(np.linalg.norm(self._last_wrench_force)):.1f} N), "
                    "returning to stiff mode"
                )
                self._in_compliant_mode = False
            super().set_pose_target(move_robot, pose, frame_id)

    # ── Recording ─────────────────────────────────────────────────────────────

    def _record_frame(self, obs: Observation, task_string: str) -> None:
        """Append one frame to the dataset episode buffer."""
        frame = {
            "observation.state": _extract_state(obs),
            "observation.images.left_camera": _ros_image_to_numpy(
                obs.left_image, self._image_scaling
            ),
            "observation.images.center_camera": _ros_image_to_numpy(
                obs.center_image, self._image_scaling
            ),
            "observation.images.right_camera": _ros_image_to_numpy(
                obs.right_image, self._image_scaling
            ),
            "action": _extract_action(obs),
            "task": task_string,
        }
        self._dataset.add_frame(frame)

    # ── Wrench ────────────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
        """Unit quaternion (w, x, y, z) → 3×3 body-to-base rotation matrix."""
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])

    def _gravity_force_in_sensor_frame(self, obs: Observation) -> np.ndarray:
        """Expected gravitational force on the F/T sensor (sensor frame)."""
        q = obs.controller_state.tcp_pose.orientation
        R = self._quat_to_rotation_matrix(q.w, q.x, q.y, q.z)
        gravity_base = np.array([0.0, 0.0, -self.GRAVITY * self.GRIPPER_PAYLOAD_MASS])
        return R.T @ gravity_base

    def _update_wrench(self, obs: Observation) -> None:
        """Update _last_wrench_force with gravity compensation and baseline removal."""
        w = obs.wrist_wrench.wrench
        raw = np.array([w.force.x, w.force.y, w.force.z])
        compensated = raw - self._gravity_force_in_sensor_frame(obs)

        if not self._wrench_baseline_ready:
            self._wrench_baseline_samples.append(compensated.copy())
            if len(self._wrench_baseline_samples) >= self.WRENCH_BASELINE_SAMPLES:
                self._wrench_baseline = np.mean(self._wrench_baseline_samples, axis=0)
                self._wrench_baseline_ready = True
                self.get_logger().info(
                    f"Wrench baseline captured: {self._wrench_baseline}"
                )

        self._last_wrench_force = compensated - self._wrench_baseline

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _task_to_string(task: Task) -> str:
        return (
            f"Insert {task.plug_type} plug ({task.cable_name}/{task.plug_name}) "
            f"into {task.port_type} port ({task.target_module_name}/{task.port_name})"
        )

    def _save_episode_metadata(
        self, episode_idx: int, frame_count: int, task_string: str
    ) -> None:
        """Append per-episode metadata so runs can be correlated with scoring.yaml."""
        meta_path = self._dataset.root / "meta" / "episodes_scoring.jsonl"
        entry = {
            "episode_index": episode_idx,
            "run_id": self._run_id,
            "trial_index": self._trial_index,
            "task": task_string,
            "frame_count": frame_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(meta_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.get_logger().info(
            f"Saved episode metadata: run={self._run_id} "
            f"trial={self._trial_index} episode={episode_idx}"
        )

    def _reset_trial_state(self) -> None:
        """Reset all per-trial mutable state before each insertion attempt."""
        self._last_wrench_force = np.zeros(3)
        self._wrench_baseline = np.zeros(3)
        self._wrench_baseline_samples = []
        self._wrench_baseline_ready = False
        self._in_compliant_mode = False
        self._insertion_phase = "idle"
        self._current_z_offset = self.APPROACH_Z_OFFSET
        # Reset CheatCode XY integrators for each trial
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0

    # ── Dataset metrics report ────────────────────────────────────────────────

    _BOLD   = "\033[1m"
    _GREEN  = "\033[92m"
    _RED    = "\033[91m"
    _YELLOW = "\033[93m"
    _CYAN   = "\033[96m"
    _DIM    = "\033[2m"
    _RESET  = "\033[0m"

    def _load_scoring_yaml(self) -> dict[str, Any]:
        """Load aic_engine scoring.yaml, returning {} on failure."""
        try:
            with open(self._scoring_yaml_path) as f:
                return yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError) as exc:
            self.get_logger().warn(f"Could not load scoring.yaml: {exc}")
            return {}

    @staticmethod
    def _ordered_engine_trials(scoring: dict[str, Any]) -> list[dict[str, Any]]:
        """Return per-trial score dicts in trial_1, trial_2, ... order."""
        pairs: list[tuple[int, dict[str, Any]]] = []
        for k, v in scoring.items():
            if not k.startswith("trial_"):
                continue
            suffix = k[6:]
            if not suffix.isdigit():
                continue
            if not isinstance(v, dict):
                continue
            pairs.append((int(suffix), v))
        pairs.sort(key=lambda x: x[0])
        return [d for _, d in pairs]

    def _load_episode_metadata(self) -> list[dict[str, Any]]:
        """Load episodes_scoring.jsonl from the dataset's meta directory."""
        meta_path = self._dataset.root / "meta" / "episodes_scoring.jsonl"
        entries: list[dict[str, Any]] = []
        if not meta_path.exists():
            return entries
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _backfill_scores(self) -> int:
        """Read scoring.yaml and merge scores into episodes_scoring.jsonl."""
        if self._dataset is None:
            return 0

        meta_path = self._dataset.root / "meta" / "episodes_scoring.jsonl"
        if not meta_path.exists():
            return 0

        scoring = self._load_scoring_yaml()
        if not scoring:
            return 0

        for k, v in scoring.items():
            if k.startswith("trial_"):
                self._cached_scoring[k] = v

        episodes = self._load_episode_metadata()
        updated = 0

        for entry in episodes:
            if "tier_1_score" in entry:
                continue
            if entry.get("run_id") != self._run_id:
                continue
            trial_key = f"trial_{entry.get('trial_index', -1)}"
            trial_data = self._cached_scoring.get(trial_key)
            if trial_data is None:
                continue
            entry["tier_1_score"] = float(trial_data.get("tier_1", {}).get("score", 0))
            entry["tier_2_score"] = float(trial_data.get("tier_2", {}).get("score", 0))
            entry["tier_3_score"] = float(trial_data.get("tier_3", {}).get("score", 0))
            updated += 1

        if updated > 0:
            with open(meta_path, "w") as f:
                for entry in episodes:
                    f.write(json.dumps(entry) + "\n")
            self.get_logger().info(
                f"Backfilled scores for {updated} episode(s) from {self._scoring_yaml_path}"
            )

        return updated

    def _print_dataset_report(self) -> None:
        """Print a richly-formatted dataset quality report to the terminal."""
        B, G, R, Y, C, D, X = (
            self._BOLD, self._GREEN, self._RED,
            self._YELLOW, self._CYAN, self._DIM, self._RESET,
        )
        W = 60
        SEP = f"{D}{'─' * W}{X}"

        all_episodes = self._load_episode_metadata()
        episodes = [e for e in all_episodes if e.get("run_id") == self._run_id]
        yaml_scoring = self._load_scoring_yaml()
        scoring: dict[str, Any] = {**yaml_scoring, **self._cached_scoring}
        engine_trials_ordered = self._ordered_engine_trials(scoring)
        num_trials_scored = len(engine_trials_ordered)

        lines: list[str] = []
        lines.append("")
        lines.append(f"{B}{C}{'═' * W}{X}")
        lines.append(f"{B}{C}  DATASET COLLECTION REPORT{X}")
        lines.append(f"{B}{C}{'═' * W}{X}")

        # ── Run info ──────────────────────────────────────────────────────────
        lines.append(SEP)
        lines.append(f"{B}  Run Info{X}")
        lines.append(SEP)
        lines.append(f"  Run ID:            {self._run_id}")
        lines.append(f"  Dataset:           {self._repo_id}")
        lines.append(f"  Dataset root:      {self._dataset.root}")
        lines.append(f"  Recording FPS:     {self._recording_fps}")
        lines.append(f"  Trials executed:   {self._trial_index}")

        run_eps = len(episodes)
        lines.append(f"  Episodes recorded: {run_eps}")

        # ── Frame statistics ──────────────────────────────────────────────────
        frame_counts = [e["frame_count"] for e in episodes if "frame_count" in e]
        lines.append(SEP)
        lines.append(f"{B}  Frame Statistics{X}")
        lines.append(SEP)
        avg_frames = 0.0
        std_frames = 0.0
        if frame_counts:
            total_frames = sum(frame_counts)
            avg_frames = total_frames / len(frame_counts)
            min_frames = min(frame_counts)
            max_frames = max(frame_counts)
            std_frames = (
                sum((x - avg_frames) ** 2 for x in frame_counts) / len(frame_counts)
            ) ** 0.5
            total_dur_s = total_frames / max(self._recording_fps, 1)

            lines.append(f"  Total frames:      {total_frames:,}")
            lines.append(f"  Total duration:    {total_dur_s:.1f}s ({total_dur_s / 60:.1f}min)")
            lines.append(f"  Avg per episode:   {avg_frames:.1f}")
            lines.append(f"  Min / Max:         {min_frames} / {max_frames}")
            lines.append(f"  Std dev:           {std_frames:.1f}")

            short_threshold = avg_frames * 0.3
            short_eps = [e for e in episodes if e.get("frame_count", 0) < short_threshold]
            if short_eps:
                lines.append(
                    f"  {Y}⚠  {len(short_eps)} episode(s) with "
                    f"<{short_threshold:.0f} frames (possible failures){X}"
                )
            else:
                lines.append(f"  {G}✓  All episodes have consistent frame counts{X}")
        else:
            lines.append(f"  {Y}No frame data available.{X}")

        # ── Per-episode scores ────────────────────────────────────────────────
        lines.append(SEP)
        lines.append(f"{B}  Per-Episode Scores{X}")
        lines.append(SEP)

        scored_episodes = [e for e in episodes if "tier_1_score" in e]
        if scored_episodes:
            lines.append(
                f"  {D}{'Ep':>3s}  {'Trial':>5s}  {'T1':>5s}  {'T2':>5s}  {'T3':>5s}  Task{X}"
            )
            for e in scored_episodes:
                t1 = e.get("tier_1_score", 0)
                t2 = e.get("tier_2_score", 0)
                t3 = e.get("tier_3_score", 0)
                c1 = G if t1 >= 0.8 else (Y if t1 >= 0.4 else R)
                c2 = G if t2 >= 0.8 else (Y if t2 >= 0.4 else R)
                c3 = G if t3 >= 0.8 else (Y if t3 >= 0.4 else R)
                lines.append(
                    f"  {e.get('episode_index', '?'):>3}  "
                    f"{e.get('trial_index', '?'):>5}  "
                    f"{c1}{t1:>5.2f}{X}  {c2}{t2:>5.2f}{X}  {c3}{t3:>5.2f}{X}  "
                    f"{D}{e.get('task', '')[:30]}{X}"
                )
        else:
            unscored = len(episodes) - len(scored_episodes)
            if unscored > 0:
                lines.append(f"  {Y}Scores not yet available for {unscored} episode(s).{X}")
            else:
                lines.append(f"  {Y}No episodes recorded.{X}")

        # ── AIC scoring aggregate ─────────────────────────────────────────────
        lines.append(SEP)
        lines.append(f"{B}  AIC Scoring (Aggregate){X}")
        lines.append(SEP)

        if num_trials_scored == 0:
            lines.append(f"  {Y}No scoring data found at {self._scoring_yaml_path}{X}")
        else:
            lines.append(f"  Trials scored:     {num_trials_scored}")

            tier1_scores: list[float] = []
            tier2_scores: list[float] = []
            tier3_scores: list[float] = []
            category_scores: dict[str, list[float]] = {}

            for trial in engine_trials_ordered:
                tier1_scores.append(float(trial.get("tier_1", {}).get("score", 0)))
                tier2_scores.append(float(trial.get("tier_2", {}).get("score", 0)))
                tier3_scores.append(float(trial.get("tier_3", {}).get("score", 0)))
                for cat, info in trial.get("tier_2", {}).get("categories", {}).items():
                    category_scores.setdefault(cat, []).append(float(info.get("score", 0)))

            def _score_color(val: float) -> str:
                return G if val >= 0.8 else (Y if val >= 0.4 else R)

            for label, scores in [
                ("Tier 1 (validation)", tier1_scores),
                ("Tier 2 (quality)",    tier2_scores),
                ("Tier 3 (completion)", tier3_scores),
            ]:
                if not scores:
                    continue
                avg = sum(scores) / len(scores)
                lines.append(
                    f"  {label:.<30s} "
                    f"{_score_color(avg)}{B}{avg:.2f}{X}  "
                    f"(range {min(scores):.2f}–{max(scores):.2f})"
                )

            if category_scores:
                lines.append(f"\n  {D}Tier-2 category breakdown:{X}")
                for cat in sorted(category_scores):
                    vals = category_scores[cat]
                    avg = sum(vals) / len(vals)
                    lines.append(f"    {cat:.<28s} {_score_color(avg)}{avg:.2f}{X}")

            successes = sum(1 for s in tier3_scores if s > 0)
            total_t = len(tier3_scores)
            rate = successes / total_t if total_t else 0.0
            lines.append("")
            lines.append(
                f"  {B}Insertion success rate: "
                f"{_score_color(rate)}{successes}/{total_t} ({rate * 100:.0f}%){X}"
            )

        # ── Data quality checks ───────────────────────────────────────────────
        lines.append(SEP)
        lines.append(f"{B}  Data Quality Checks{X}")
        lines.append(SEP)

        checks_passed = 0
        checks_total = 0

        checks_total += 1
        if run_eps == self._trial_index:
            lines.append(
                f"  {G}✓{X}  Episodes match trials executed ({run_eps}/{self._trial_index})"
            )
            checks_passed += 1
        else:
            lines.append(
                f"  {R}✗{X}  Episode count mismatch: "
                f"{run_eps} recorded vs {self._trial_index} executed"
            )

        checks_total += 1
        zero_eps = [e for e in episodes if e.get("frame_count", 0) == 0]
        if not zero_eps:
            lines.append(f"  {G}✓{X}  No zero-frame episodes")
            checks_passed += 1
        else:
            lines.append(f"  {R}✗{X}  {len(zero_eps)} episode(s) have zero frames")

        checks_total += 1
        if frame_counts and avg_frames > 0:
            cv = std_frames / avg_frames
            if cv < 0.5:
                lines.append(f"  {G}✓{X}  Frame count consistency (CV={cv:.2f})")
                checks_passed += 1
            else:
                lines.append(
                    f"  {Y}⚠{X}  High frame count variance "
                    f"(CV={cv:.2f}) — check for interrupted episodes"
                )
        else:
            lines.append(f"  {D}–  Not enough data for consistency check{X}")

        checks_total += 1
        if num_trials_scored >= self._trial_index and self._trial_index > 0:
            lines.append(
                f"  {G}✓{X}  Scoring data covers all {self._trial_index} trials"
            )
            checks_passed += 1
        elif num_trials_scored > 0:
            lines.append(
                f"  {Y}⚠{X}  Scoring data covers {num_trials_scored}/{self._trial_index} trials"
            )
        else:
            lines.append(f"  {Y}⚠{X}  No scoring data available for validation")

        clr = G if checks_passed == checks_total else Y
        lines.append(f"\n  {clr}{B}{checks_passed}/{checks_total} checks passed{X}")

        lines.append(f"{B}{C}{'═' * W}{X}")
        lines.append("")

        print("\n".join(lines), flush=True)

    # ── Insertion logic ───────────────────────────────────────────────────────

    def _rotational_search(
        self,
        port_transform: Transform,
        contact_z_offset: float,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        episode_task: str,
        recording_enabled: bool,
        frame_count: int,
    ) -> int:
        """Spiral search to find the port slot after contact is made.

        Commands a growing spiral of XY offsets around the port centre
        while keeping the compliant Z-axis pressed against the surface.
        Uses calc_gripper_pose (same formula as descent) to avoid any
        Z discontinuity when transitioning from descent to search.
        The bounded radius (SEARCH_MAX_RADIUS = 2 mm) ensures the plug
        never wanders outside the circuit board.

        Returns the updated frame_count.
        """
        steps_per_rev = max(1, int(2 * np.pi / self.SEARCH_ANGLE_STEP))
        total_steps = int(self.SEARCH_MAX_REVOLUTIONS * steps_per_rev)

        self._insertion_phase = "search"
        self.get_logger().info(
            f"Rotational search: {total_steps} steps over "
            f"{self.SEARCH_MAX_REVOLUTIONS} revolutions, "
            f"max_r={self.SEARCH_MAX_RADIUS * 1000:.1f} mm, "
            f"z_offset={contact_z_offset:.4f} m"
        )
        send_feedback("Rotational search for slot...")

        angle = 0.0
        for step in range(total_steps):
            revolution = step / steps_per_rev
            radius = min(revolution * self.SEARCH_RADIUS_STEP, self.SEARCH_MAX_RADIUS)
            angle += self.SEARCH_ANGLE_STEP

            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)

            # Build a copy of port_transform with the XY search offset applied.
            # Z and orientation stay at the corrected port values.
            # Use calc_gripper_pose (same Z formula as descent) so there is
            # no commanded height jump at the start of the search.
            # reset_xy_integrator=True keeps integrators frozen — the XY
            # correction is already baked into port_transform.
            search_transform = Transform(
                translation=Vector3(
                    x=port_transform.translation.x + dx,
                    y=port_transform.translation.y + dy,
                    z=port_transform.translation.z,
                ),
                rotation=Quaternion(
                    x=port_transform.rotation.x,
                    y=port_transform.rotation.y,
                    z=port_transform.rotation.z,
                    w=port_transform.rotation.w,
                ),
            )

            try:
                pose = self.calc_gripper_pose(
                    search_transform,
                    z_offset=contact_z_offset,
                    reset_xy_integrator=True,
                )
                self.set_pose_target(move_robot, pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF during search step {step}: {ex}")

            obs = get_observation()
            if obs is not None:
                self._update_wrench(obs)
                if recording_enabled:
                    self._record_frame(obs, episode_task)
                    frame_count += 1

            self.sleep_for(0.05)

            if step % steps_per_rev == 0:
                force_mag = float(np.linalg.norm(self._last_wrench_force))
                self.get_logger().info(
                    f"Search rev {revolution:.1f}: "
                    f"r={radius * 1000:.1f} mm, "
                    f"force={force_mag:.1f} N"
                )

        self.get_logger().info("Rotational search complete.")
        return frame_count

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info("DataCollectorCheatCode.insert_cable() enter")
        self._reset_trial_state()
        self._task = task
        self._ensure_dataset()
        self._backfill_scores()

        episode_task = self._task_to_string(task)
        recording_enabled = (
            self._num_episodes < 0 or self._trial_index < self._num_episodes
        )
        if not recording_enabled:
            self.get_logger().info(
                f"Episode limit reached ({self._num_episodes}), running without recording"
            )

        frame_count = 0

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        # Wait for TF frames (same as CheatCode)
        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Port TF lookup failed: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        # ── Phase 1: Approach ──────────────────────────────────────────────
        # SLERP from current TCP pose to APPROACH_Z_OFFSET above port (5 s).
        # Uses CheatCode's calc_gripper_pose with reset_xy_integrator=True so
        # the XY integrators don't accumulate noise during interpolation.
        send_feedback("Approaching port...")
        self._insertion_phase = "approach"
        self._current_z_offset = self.APPROACH_Z_OFFSET

        for i in range(100):
            frac = i / 100.0
            try:
                pose = self.calc_gripper_pose(
                    port_transform,
                    slerp_fraction=frac,
                    position_fraction=frac,
                    z_offset=self.APPROACH_Z_OFFSET,
                    reset_xy_integrator=True,
                )
                self.set_pose_target(move_robot, pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF during approach step {i}: {ex}")

            obs = get_observation()
            if obs is not None:
                self._update_wrench(obs)
                if recording_enabled:
                    self._record_frame(obs, episode_task)
                    frame_count += 1

            self.sleep_for(0.05)

        # ── Settle + re-lookup port TF ─────────────────────────────────────
        # Give the robot and board physics time to settle before descent.
        # Re-fetch port TF — the board may have micro-settled since spawn.
        send_feedback("Settling at hover...")
        self.sleep_for(1.0)
        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time(),
            )
            port_transform = port_tf_stamped.transform
            self.get_logger().info("Port TF re-fetched after approach settle.")
        except TransformException as ex:
            self.get_logger().warn(f"Port TF re-lookup failed ({ex}), using original.")

        # ── Phase 2: Descent ───────────────────────────────────────────────
        # Straight-down at DESCENT_STEP_M per step (10 mm/s at 20 Hz).
        # XY integrators accumulate plug-tip→port error and correct the
        # commanded XY on every step (CheatCode strategy).
        # Exits when contact force exceeds CONTACT_FORCE_THRESHOLD
        # or z_offset falls below −15 mm (full descent limit).
        send_feedback("Descending to contact...")
        self._insertion_phase = "descent"
        z_offset = float(self.APPROACH_Z_OFFSET)
        contact_detected = False

        while z_offset > -0.015:
            z_offset -= self.DESCENT_STEP_M
            self._current_z_offset = z_offset

            try:
                pose = self.calc_gripper_pose(port_transform, z_offset=z_offset)
                self.set_pose_target(move_robot, pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF during descent: {ex}")

            obs = get_observation()
            if obs is not None:
                self._update_wrench(obs)
                if recording_enabled:
                    self._record_frame(obs, episode_task)
                    frame_count += 1

            self.sleep_for(0.05)

            if (
                self._wrench_baseline_ready
                and float(np.linalg.norm(self._last_wrench_force)) > self.CONTACT_FORCE_THRESHOLD
            ):
                contact_detected = True
                send_feedback(
                    f"Contact at z_offset={z_offset:.3f} m — starting rotational search"
                )
                self.get_logger().info(
                    f"Contact detected: force={float(np.linalg.norm(self._last_wrench_force)):.1f} N, "
                    f"z_offset={z_offset:.4f} m"
                )
                break

        # ── Phase 3: Rotational search (if contact was made) ───────────────
        # Compliant impedance + spiral XY pattern to find and enter the slot.
        # The search is centred on the integrator-corrected port XY so the
        # spiral starts from the best-known plug alignment, not raw GT.
        # If contact was NOT detected we simply let the full descent stand.
        if contact_detected:
            corrected_port_transform = Transform(
                translation=Vector3(
                    x=port_transform.translation.x + self.XY_I_GAIN * self._tip_x_error_integrator,
                    y=port_transform.translation.y + self.XY_I_GAIN * self._tip_y_error_integrator,
                    z=port_transform.translation.z,
                ),
                rotation=Quaternion(
                    x=port_transform.rotation.x,
                    y=port_transform.rotation.y,
                    z=port_transform.rotation.z,
                    w=port_transform.rotation.w,
                ),
            )
            self.get_logger().info(
                f"Search centre: integrator correction "
                f"dx={self.XY_I_GAIN * self._tip_x_error_integrator * 1000:.2f} mm, "
                f"dy={self.XY_I_GAIN * self._tip_y_error_integrator * 1000:.2f} mm"
            )
            frame_count = self._rotational_search(
                port_transform=corrected_port_transform,
                contact_z_offset=z_offset,
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                episode_task=episode_task,
                recording_enabled=recording_enabled,
                frame_count=frame_count,
            )

        # ── Phase 4: Stabilise ─────────────────────────────────────────────
        # Hold position and record while physics and aic_engine settle.
        # 5 s matches CheatCode's stabilisation window.
        send_feedback("Stabilising...")
        self._insertion_phase = "idle"
        for _ in range(100):  # 5 s at 20 Hz
            obs = get_observation()
            if obs is not None:
                self._update_wrench(obs)
                if recording_enabled:
                    self._record_frame(obs, episode_task)
                    frame_count += 1
            self.sleep_for(0.05)

        # ── Episode bookkeeping ────────────────────────────────────────────
        self._trial_index += 1

        if frame_count > 0:
            self._dataset.save_episode()
            ep_idx = self._dataset.meta.total_episodes - 1
            self.get_logger().info(
                f"Saved episode {ep_idx} with {frame_count} frames to {self._dataset.root}"
            )
            self._save_episode_metadata(ep_idx, frame_count, episode_task)
        elif recording_enabled:
            self.get_logger().warn("No frames were recorded for this episode.")

        self.get_logger().info(
            f"DataCollectorCheatCode.insert_cable() done. "
            f"Trial {self._trial_index}"
            f"/{self._num_episodes if self._num_episodes > 0 else '∞'}, "
            f"total recorded episodes: {self._dataset.meta.total_episodes}"
        )

        is_final_episode = (
            self._num_episodes > 0 and self._trial_index >= self._num_episodes
        )
        if self._report_metrics and is_final_episode:
            self._wait_and_backfill_final_score()
            self._print_dataset_report()

        if is_final_episode:
            self.get_logger().info(
                "All requested episodes collected. Exiting after 8 s."
            )
            import threading
            threading.Timer(8.0, lambda: os._exit(0)).start()

        return True

    def _wait_and_backfill_final_score(
        self, timeout_sec: float = 15.0, poll_interval: float = 1.0,
    ) -> None:
        """Poll scoring.yaml until the last trial's score appears."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self._backfill_scores()
            episodes = self._load_episode_metadata()
            if episodes and all("tier_1_score" in e for e in episodes):
                self.get_logger().info("All episode scores backfilled successfully.")
                return
            elapsed = timeout_sec - (deadline - time.monotonic())
            self.get_logger().info(
                f"Waiting for scoring.yaml (trial {self._trial_index}) "
                f"— {elapsed:.0f} s / {timeout_sec:.0f} s"
            )
            time.sleep(poll_interval)

        self._backfill_scores()
        self.get_logger().warn(
            f"Timed out waiting for all scores after {timeout_sec} s. "
            "Some episodes may be missing score data."
        )
