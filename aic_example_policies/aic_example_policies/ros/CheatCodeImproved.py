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


import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

QuaternionTuple = tuple[float, float, float, float]

# Softer Cartesian impedance during insertion/descent only (vs Policy.set_pose_target
# defaults 90/90/90 translational, 50/50/50 rotational). Reduces jamming on
# port chamfers by allowing slight compliance in X/Y; Z kept slightly stiffer
# than X/Y to retain downward motion.
_INSERT_STIFFNESS = [45.0, 45.0, 55.0, 35.0, 35.0, 35.0]
_INSERT_DAMPING = [38.0, 38.0, 42.0, 18.0, 18.0, 18.0]

# Plug-in-gripper readiness: max distance (m) from gripper/tcp to plug tip
# that we consider "cable is properly attached and settled." Grasp offsets in
# sample configs are ~0.04 m; allow headroom for cable flex.
_PLUG_GRIPPER_MAX_DIST = 0.10
_PLUG_READY_CONSECUTIVE = 5  # consecutive OK readings required
_PLUG_READY_TIMEOUT_SEC = 15.0

# Small circular XY dither ("wiggle") applied during descent when close to
# the port mouth.  Helps the plug slide off the chamfer/lip into the opening.
_WIGGLE_Z_THRESHOLD = 0.02  # activate when z_offset drops below this (m)
_WIGGLE_AMPLITUDE = 0.0008  # radius of the circular motion (m)
_WIGGLE_PERIOD_STEPS = 40   # descent steps for one full circle


class CheatCodeImproved(Policy):
    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        super().__init__(parent_node)

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        """Wait for a TF frame to become available."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... -- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def _wait_for_plug_in_gripper(
        self, cable_tip_frame: str
    ) -> bool:
        """Block until plug tip is close to gripper/tcp for several consecutive
        readings, meaning the cable is attached and settled.  Returns False on
        timeout."""
        start = self.time_now()
        timeout = Duration(seconds=_PLUG_READY_TIMEOUT_SEC)
        consecutive_ok = 0
        while (self.time_now() - start) < timeout:
            try:
                tf = self._parent_node._tf_buffer.lookup_transform(
                    "gripper/tcp",
                    cable_tip_frame,
                    Time(),
                )
                dx = tf.transform.translation.x
                dy = tf.transform.translation.y
                dz = tf.transform.translation.z
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                if dist <= _PLUG_GRIPPER_MAX_DIST:
                    consecutive_ok += 1
                    if consecutive_ok >= _PLUG_READY_CONSECUTIVE:
                        self.get_logger().info(
                            f"Plug settled in gripper (dist={dist:.4f} m, "
                            f"{consecutive_ok} consecutive OK readings)."
                        )
                        return True
                else:
                    if consecutive_ok > 0:
                        self.get_logger().info(
                            f"Plug–gripper distance reset ({dist:.4f} m > "
                            f"{_PLUG_GRIPPER_MAX_DIST} m), waiting..."
                        )
                    consecutive_ok = 0
            except TransformException:
                consecutive_ok = 0
            self.sleep_for(0.05)
        self.get_logger().error(
            f"Plug did not settle in gripper within {_PLUG_READY_TIMEOUT_SEC}s"
        )
        return False

    def _lookup_port_transform(self, port_frame: str) -> Transform | None:
        """Latest port pose in base_link; None if lookup fails."""
        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
            return port_tf_stamped.transform
        except TransformException as ex:
            self.get_logger().warn(f"Port transform lookup failed: {ex}")
            return None

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        """Find the gripper pose that results in plug alignment."""
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )

        self.get_logger().info(
            f"pfrac: {position_fraction:.3} xy_error: {tip_x_error:0.3} {tip_y_error:0.3}   integrators: {self._tip_x_error_integrator:.3} , {self._tip_y_error_integrator:.3}"
        )

        i_gain = 0.15

        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(
                x=blend_xyz[0],
                y=blend_xyz[1],
                z=blend_xyz[2],
            ),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"CheatCodeImproved.insert_cable() task: {task}")
        self._task = task

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        # Wait for both the port and cable tip TFs to become available.
        # These come via ground_truth and may not be immediate.
        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        # Ensure the plug is actually attached and settled in the gripper
        # before commanding any motion (avoids bad first targets from stale
        # or pre-attach TF).
        if not self._wait_for_plug_in_gripper(cable_tip_frame):
            self.get_logger().error("Aborting: plug not settled in gripper.")
            return False

        z_offset = 0.2

        # Over five seconds, smoothly interpolate from the current position to
        # a position above the port. Advance the schedule only after a successful
        # command so TF drops do not desync motion from the interpolation progress.
        t = 0
        while t < 100:
            interp_fraction = t / 100.0
            port_transform = self._lookup_port_transform(port_frame)
            if port_transform is None:
                self.sleep_for(0.05)
                continue
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
                self.sleep_for(0.05)
                continue
            t += 1
            self.sleep_for(0.05)

        # Descend until the cable is inserted into the port. Only step z_offset
        # after a successful command so skipped ticks do not advance "virtual" depth.
        # A small circular XY dither is added near the port mouth to help the
        # plug find the opening if it lands on the chamfer.
        descent_step = 0
        while True:
            if z_offset < -0.015:
                break

            next_z = z_offset - 0.0005
            port_transform = self._lookup_port_transform(port_frame)
            if port_transform is None:
                self.sleep_for(0.05)
                continue
            try:
                pose = self.calc_gripper_pose(port_transform, z_offset=next_z)
                if next_z < _WIGGLE_Z_THRESHOLD:
                    angle = 2.0 * np.pi * descent_step / _WIGGLE_PERIOD_STEPS
                    pose.position.x += _WIGGLE_AMPLITUDE * np.cos(angle)
                    pose.position.y += _WIGGLE_AMPLITUDE * np.sin(angle)
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=_INSERT_STIFFNESS,
                    damping=_INSERT_DAMPING,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                self.sleep_for(0.05)
                continue

            z_offset = next_z
            descent_step += 1
            self.get_logger().info(f"z_offset: {z_offset:0.5}")
            self.sleep_for(0.05)

        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(5.0)

        self.get_logger().info("CheatCodeImproved.insert_cable() exiting...")
        return True
