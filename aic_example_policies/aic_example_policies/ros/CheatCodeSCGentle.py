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

"""Slower, more compliant CheatCode variant for SC plugs; SFP unchanged.

Use when the base CheatCode trajectory collides with the board or fails to seat
the SC connector at difficult rail translations. Tune class attributes if needed.
"""

from aic_control_interfaces.msg import TargetMode
from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from tf2_ros import TransformException


class CheatCodeSCGentle(CheatCode):
    """Delegates SFP (and non-SC) tasks to :class:`CheatCode`; softens SC insertion only."""

    _SC_APPROACH_STEPS = 140
    _SC_APPROACH_DT_SEC = 0.06
    _SC_APPROACH_Z_OFFSET = 0.2
    _SC_APPROACH_STIFFNESS = [72.0, 72.0, 72.0, 46.0, 46.0, 46.0]
    _SC_APPROACH_DAMPING = [56.0, 56.0, 56.0, 24.0, 24.0, 24.0]

    _SC_INSERT_Z_STEP = 0.00045
    _SC_INSERT_DT_SEC = 0.055
    _SC_INSERT_FINE_START_Z = -0.01
    _SC_INSERT_FINE_Z_STEP = 0.00022
    _SC_INSERT_FINE_DT_SEC = 0.08
    _SC_INSERT_Z_TERMINAL = -0.036
    _SC_INSERT_STIFFNESS = [44.0, 44.0, 36.0, 36.0, 36.0, 36.0]
    _SC_INSERT_DAMPING = [70.0, 70.0, 60.0, 32.0, 32.0, 32.0]
    _SC_INSERT_FINE_STIFFNESS = [38.0, 38.0, 30.0, 32.0, 32.0, 32.0]
    _SC_INSERT_FINE_DAMPING = [74.0, 74.0, 68.0, 34.0, 34.0, 34.0]

    _SC_PRESS_Z_TERMINAL = -0.044
    _SC_PRESS_Z_STEP = 0.00014
    _SC_PRESS_DT_SEC = 0.09
    _SC_PRESS_STIFFNESS = [34.0, 34.0, 26.0, 30.0, 30.0, 30.0]
    _SC_PRESS_DAMPING = [78.0, 78.0, 72.0, 36.0, 36.0, 36.0]

    _SC_SETTLE_SEC = 7.0

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        if task.plug_type != "sc":
            return super().insert_cable(
                task, get_observation, move_robot, send_feedback
            )
        return self._insert_cable_sc_gentle(
            task, get_observation, move_robot, send_feedback
        )

    def _insert_cable_sc_gentle(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"CheatCodeSCGentle.insert_cable() task: {task}")
        self._task = task
        self._reset_integrators()
        self._parent_node._target_mode = TargetMode.MODE_UNSPECIFIED

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        self.get_logger().info(
            f"Waiting {self._SETTLE_DELAY_SEC}s for robot/physics to settle..."
        )
        self.sleep_for(self._SETTLE_DELAY_SEC)

        self._parent_node._target_mode = TargetMode.MODE_UNSPECIFIED
        self._parent_node.set_target_mode(TargetMode.MODE_CARTESIAN)

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        z_offset = self._SC_APPROACH_Z_OFFSET
        n = self._SC_APPROACH_STEPS
        dt = self._SC_APPROACH_DT_SEC

        for t in range(0, n):
            self._reassert_cartesian_mode(t)
            interp_fraction = t / float(n)
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
                    stiffness=self._SC_APPROACH_STIFFNESS,
                    damping=self._SC_APPROACH_DAMPING,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(dt)

        insert_step = 0
        while z_offset > self._SC_INSERT_Z_TERMINAL:
            self._reassert_cartesian_mode(insert_step)
            if z_offset <= self._SC_INSERT_FINE_START_Z:
                step = self._SC_INSERT_FINE_Z_STEP
                dt = self._SC_INSERT_FINE_DT_SEC
                stiff = self._SC_INSERT_FINE_STIFFNESS
                damp = self._SC_INSERT_FINE_DAMPING
            else:
                step = self._SC_INSERT_Z_STEP
                dt = self._SC_INSERT_DT_SEC
                stiff = self._SC_INSERT_STIFFNESS
                damp = self._SC_INSERT_DAMPING

            z_offset -= step
            insert_step += 1
            if insert_step % 15 == 0:
                self.get_logger().info(f"z_offset: {z_offset:0.5}")
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(port_transform, z_offset=z_offset),
                    stiffness=stiff,
                    damping=damp,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
            self.sleep_for(dt)

        self.get_logger().info("SC press phase (final depth)...")
        press_iter = 0
        while z_offset > self._SC_PRESS_Z_TERMINAL:
            self._reassert_cartesian_mode(press_iter)
            press_iter += 1
            z_offset -= self._SC_PRESS_Z_STEP
            insert_step += 1
            if insert_step % 12 == 0:
                self.get_logger().info(f"z_offset (press): {z_offset:0.5}")
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(port_transform, z_offset=z_offset),
                    stiffness=self._SC_PRESS_STIFFNESS,
                    damping=self._SC_PRESS_DAMPING,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during press: {ex}")
            self.sleep_for(self._SC_PRESS_DT_SEC)

        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(self._SC_SETTLE_SEC)

        self.get_logger().info("CheatCodeSCGentle.insert_cable() exiting...")
        return True
