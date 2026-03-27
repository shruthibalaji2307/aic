from pyatk.tools import Admittance, AdmittanceParams

import pyatk
from pyatk import ARAASError, Transform, Vector

from .. import araas_init

"""
The pyaraas robot admittance controller built on the low-level pyatk admittance object.

The specified TCP is where admittance is located (i.e. attraction and compliance).

The registered callback is called on each control update tick. The callback determines the stop condition
and is also where controller parameters can be adjusted. The callback signtaure is
user_cb_func(controller, force, moment, world_T_tcp, elapsed).
"""


class AdmittanceController:
    def __init__(self, robot, force_sensor, world_T_tc, user_cb_func):
        self.robot = robot
        # control tcp is rigidly offset from the robot flange and sensor frame
        self.flange_T_tcp = robot.get_flange_transform().invert().multiply(world_T_tcp)
        self.tcp_T_flange = self.flange_T_tcp.invert()
        self.world_T_goal = Transform(world_T_tcp)

        # the robot will be running a cartesian velocity controller.
        self.robot.init_cartesian_velocity_control()

        # the specified user_cb_func parameter is where the stop condition and
        # any controller parameter adjustment happens.
        self.user_cb_func = user_cb_func

        self.params = AdmittanceParams(
            robot.get_joint(robot.num_joints - 1), world_T_tcp, force_sensor
        )
        self.adm_control = Admittance(self.params)

        if self.robot.workcell.hardware_enabled:
            # with real robot controller, a slower moving model keeps overshoot in check
            self.adm_control.params.M[:6] = 20
            self.adm_control.params.kp[:6] = 200
            self.adm_control.recompute_damping()
            self.adm_control.params.max_v = 100

    def adm_controller_cb(self, robot, user_data, dt, elapsed):
        # compute the admittance velocity located at the tcp frame
        self.robot.get_flange_transform()
        linear, angular = self.adm_control.update(self.world_T_goal, dt)

        # map the velocity to be located at the robot flange frame (for the robot velocity controller)
        linear, angular = self.map_velocities_to_robot(linear, angular)

        # call user specified cb for the stop condition (and allow for any controller parameter adjustments)
        stop = self.user_cb_func(
            self,
            self.adm_control.force,
            self.adm_control.moment,
            self.adm_control.p_world_T_tcp,
            elapsed,
        )
        return linear, angular, stop

    def map_velocities_to_robot(self, world_V_linear, world_V_angular):
        # move the velocity from the tcp location to the flange location
        world_T_tcp = self.adm_control.p_world_T_tcp
        world_T_flange = world_T_tcp.multiply(self.tcp_T_flange)

        offset = world_T_flange.position.subtract(world_T_tcp.position)
        extra = world_V_angular.cross(offset)
        world_V_linear = world_V_linear.add(extra)

        return world_V_linear, world_V_angular

    async def start(self):
        if self.user_cb_func is None:
            raise ARAASError(
                "control callback function must be registered before starting the controller."
            )
        # this creates an async velocity controller loop.
        # the underlying velocity controller calls self.adm_controller_cb at the action update rate (~200 Hz).
        # the loop runs until the user_cb_func returns True (i.e. stop)
        await self.robot.cartesian_velocity_control(self.adm_controller_cb, self)