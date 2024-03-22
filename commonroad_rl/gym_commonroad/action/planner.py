"""
polynomial planner for the vehicle
"""

import logging
from multiprocessing.context import Process
import numpy as np
from commonroad_rl.gym_commonroad.action.vehicle import ContinuousVehicle
import math
from typing import List
from commonroad.common.util import make_valid_orientation
from commonroad.scenario.trajectory import (
    State,
    STState,
    Trajectory,
    TrajectoryParams,
    CustomState,
)
from commonroad_rp.state import ReactivePlannerState
from commonroad_rp.utility.utils_coordinate_system import (
    CoordinateSystem,
    interpolate_angle,
)
from vehiclemodels.vehicle_parameters import VehicleParameters

import matplotlib.pyplot as plt
from commonroad_rl.gym_commonroad.utils.navigator import Navigator

# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.utility.logger import initialize_logger
from typing import List, Union, Optional, Tuple, Type, Dict
import time
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

KCONSTV = 1.5
MAX_KCONSTV = 15.0
LOW_S_DELTA_D = 0.1


def interpolate_angle(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """
    Interpolates an angle value between two angles according to the miminal value of the absolute difference
    :param x: value of other dimension to interpolate
    :param x1: lower bound of the other dimension
    :param x2: upper bound of the other dimension
    :param y1: lower bound of angle to interpolate
    :param y2: upper bound of angle to interpolate
    :return: interpolated angular value (in rad)
    """

    def absmin(x):
        return x[np.argmin(np.abs(x))]

    delta = y2 - y1
    # delta_2pi_minus = delta - 2 * np.pi
    # delta_2pi_plus = delta + 2 * np.pi
    # delta = absmin(np.array([delta, delta_2pi_minus, delta_2pi_plus]))

    return make_valid_orientation(delta * (x - x1) / (x2 - x1) + y1)


class PlanTrajectory:
    def __init__(
        self,
        frenet_traj: List[np.ndarray] = None,
        cart_traj: List[np.ndarray] = None,
        dt: float = 0.1,
    ) -> None:
        # [traj_s, traj_ds, traj_dds, traj_l, traj_dl, traj_ddl]
        self._frenet_traj = frenet_traj
        # [traj_x, traj_y, traj_theta, traj_kappa, traj_v, traj_a]
        self.cart_traj = cart_traj
        self.frenet_s = frenet_traj[0]
        self.frenet_l = frenet_traj[3]
        self.frenet_ds = frenet_traj[1]
        self.frenet_dl = frenet_traj[4]
        self.frenet_dds = frenet_traj[2]
        self.frenet_ddl = frenet_traj[5]
        self.cart_x = cart_traj[0]
        self.cart_y = cart_traj[1]
        self.cart_theta = cart_traj[2]  # velocity direction with the x axis
        self.cart_kappa = cart_traj[3]
        self.cart_v = cart_traj[4]
        self.cart_a = cart_traj[5]
        self.dt = dt

    @property
    def cartesian_path(self):
        return np.array([self.cart_x, self.cart_y]).transpose()

    def convert_to_viz_trajectory(self, **kwargs):
        """
        Converts the trajectory to a CommonRoad Trajectory object
        :return: CommonRoad Trajectory object
        """
        traj_color = kwargs.get("traj_color", (217 / 255, 79 / 255, 51 / 255))
        trajectory_viz_params = TrajectoryParams(
            time_begin=0,
            time_end=len(self.cart_x),
            draw_continuous=True,
            line_width=2.0,
            facecolor=traj_color,
        )
        state_list = []
        for i in range(len(self.cart_x)):
            state_list.append(
                STState(
                    time_step=i,
                    orientation=self.cart_theta[i],
                    position=np.array([self.cart_x[i], self.cart_y[i]]),
                )
            )
        return (
            Trajectory(initial_time_step=0, state_list=state_list),
            trajectory_viz_params,
        )


class PolynomialPlanner:
    def __init__(self, vehicle_params: VehicleParameters = None) -> None:
        self._vehicle_params = vehicle_params
        self._trajectory = None
        self._low_vel_mode = False
        self.initial_state_dict = None

    @property
    def trajectory(self) -> PlanTrajectory:
        return self._trajectory

    @property
    def state_nums(self):
        return self._max_horizon_step

    @property
    def time_step(self):
        return self._dt

    @property
    def max_plan_time(self):
        return self._plan_t

    @property
    def coordinate_system(self):
        return self._co

    @coordinate_system.setter
    def coordinate_system(self, coordinate_system: CoordinateSystem):
        self._co = coordinate_system
        EPS = 0.1
        curvilinear_border = self._co.ccosy.curvilinear_projection_domain()
        if not isinstance(curvilinear_border, np.ndarray):
            curvilinear_border = np.array(curvilinear_border)
        self.max_s = curvilinear_border[:, 0].max() - EPS
        self.min_s = curvilinear_border[:, 0].min() + EPS
        self.max_l = curvilinear_border[:, 1].max() - EPS
        self.min_l = curvilinear_border[:, 1].min() + EPS

    def reset(
        self,
        plan_t: float,
        dt: float = 0.1,
        local_ccosy: CurvilinearCoordinateSystem = None,
    ) -> None:
        """
        set the planning time and the time step
        plan_t: max planning time, default 5s
        dt: scenario time step, default 0.1s
        """
        self._plan_t = plan_t
        self._dt = dt
        self._max_horizon_step = int(plan_t / dt)
        self._trajectory = None
        self._low_vel_mode = False
        self.coordinate_system = CoordinateSystem(ccosy=local_ccosy)

    def ReferenceTraj(self, navigator: Navigator = None):
        """
        not use
        """
        reference_pts = navigator.route.reference_path
        # compute theta
        diff_x = reference_pts[1:, 0] - reference_pts[:-1, 0]
        diff_y = reference_pts[1:, 1] - reference_pts[:-1, 1]
        theta = np.arctan2(diff_y, diff_x)
        theta = np.append(theta, theta[-1])
        frenet_ds = 15 * np.ones(len(reference_pts[:, 0]))
        zeros = np.zeros(len(reference_pts[:, 0]))
        frenet_traj = [zeros, frenet_ds, zeros, zeros, zeros, zeros]
        cart_traj = [
            reference_pts[:, 0],
            reference_pts[:, 1],
            theta,
            zeros,
            zeros,
            zeros,
        ]
        # [traj_s, traj_ds, traj_dds, traj_l, traj_dl, traj_ddl]
        # [traj_x, traj_y, traj_theta, traj_kappa, traj_v, traj_a]
        ref_traj = PlanTrajectory(cart_traj=cart_traj, frenet_traj=frenet_traj)

        return ref_traj

    def PlanTraj(
        self,
        vehicle: ContinuousVehicle = None,
        rescaled_action: np.ndarray = None,
        logger: logging.Logger = None,
    ) -> PlanTrajectory:
        # action_target: [t_lon_target, t_lat_target, v_target, d_target]

        # lat_time = rescaled_action[2]
        lat_time = self._plan_t
        delta_v = rescaled_action[0]
        target_d = rescaled_action[1]
        max_plan_time = self._plan_t
        target_v = vehicle.state.velocity + delta_v
        target_v = np.clip(target_v, KCONSTV, MAX_KCONSTV)

        # find current s, l
        current_state: STState = vehicle.state
        logger.debug(f"current vehicle velocity is {current_state.velocity}, target velocity is {target_v}")
        self._low_vel_mode = False
        if current_state.velocity < 4.0:
            self._low_vel_mode = True
        s_0_list, l_0_list = self._compute_initial_states(
            current_state,
            vehicle.current_action[1],
            vehicle.current_action[0],
            logger=logger,
        )
        if s_0_list is None and l_0_list is None:
            return None
        cur_s_0, ds_0, dds_0 = s_0_list
        cur_l_0, dl_0, ddl_0 = l_0_list

        s_0, l_0 = self._stitch_trajectory(cur_s_0, cur_l_0)

        # clip target d
        target_d = np.clip(target_d, self.min_l, self.max_l)

        logger.debug(f"current s is {s_0}, ds is {ds_0}, dds is {dds_0}")
        logger.debug(f"current l is {l_0}, dl is {dl_0}, ddl is {ddl_0}")
        traj_s, traj_ds, traj_dds = self.quartic_polynomial(s_0, ds_0, dds_0, target_v, max_plan_time)
        traj_s, traj_ds, traj_dds = self._check_s_traj(
            traj_s=traj_s,
            traj_ds=traj_ds,
            traj_dds=traj_dds,
        )
        if not self._low_vel_mode:
            traj_l, traj_dl, traj_ddl = self.quintic_polynomial(l_0, dl_0, ddl_0, target_d, 0.0, 0.0, lat_time)
        else:
            traj_l, traj_dl, traj_ddl = self.quintic_polynomial_with_s(l_0, dl_0, ddl_0, target_d, 0.0, 0.0, traj_s)
        # save the initial state info
        self.initial_state_dict = {
            "s_0": s_0,
            "l_0": l_0,
            "ds_0": ds_0,
            "dds_0": dds_0,
            "dl_0": dl_0,
            "ddl_0": ddl_0,
            "target_v": target_v,
        }
        frenet_traj = [traj_s, traj_ds, traj_dds, traj_l, traj_dl, traj_ddl]

        valid_frenet_traj: List[np.ndarray] = self._check_valid_frenet_traj(frenet_traj=frenet_traj)

        cart_traj = self._get_cart_traj(valid_frenet_traj)

        if cart_traj is None or cart_traj[0].shape[0] != valid_frenet_traj[0].shape[0]:
            # plt.figure(2)
            # plt.plot(traj_s)
            # plt.title("s-t")
            # plt.figure(3)
            # plt.plot(traj_l)
            # plt.title("l-t")
            # plt.figure(4)
            # plt.plot(traj_s, traj_l)
            # plt.title("s-l")
            # plt.figure(5)
            # plt.plot(traj_ds)
            # plt.title("ds-t")
            # plt.figure(6)
            # plt.plot(traj_dl)
            # plt.title("dl-t")
            # plt.figure(7)
            # plt.plot(traj_dds)
            # plt.title("dds-t")
            # plt.figure(8)
            # plt.plot(traj_ddl)
            # plt.title("ddl-t")
            # plt.show()
            logger.warning("Cartesian trajectory is None or length is not equal to Frenet trajectory!")
            return None

        if not self._valid_cart_traj(cartesian_traj=cart_traj, logger=logger):
            return None

        trajectory = PlanTrajectory(frenet_traj=frenet_traj, cart_traj=cart_traj, dt=self._dt)

        self._trajectory = trajectory

        return trajectory

    # longitudinal planning
    def quartic_polynomial(self, s, ds, dds, v_target, t_target):
        # s = a0 t^4 + a1 * t^3 + a2 * t^2 + a3 * t + a4
        a4 = s
        a3 = ds
        a2 = dds / 2.0

        A = np.array([[4 * t_target**3, 3 * t_target**2], [12 * t_target**2, 6 * t_target]])
        b = np.array([v_target - a3 - 2 * t_target * a2, -2 * a2])

        x = np.linalg.solve(A, b)

        a0 = x[0]
        a1 = x[1]

        t_series_1 = np.arange(0.0, t_target, self._dt) + self._dt
        t_series_2 = np.arange(t_series_1[-1], self._plan_t, self._dt) + self._dt

        t_series = np.concatenate((t_series_1, t_series_2))
        traj_s = [s]
        traj_ds = [ds]
        traj_dds = [dds]
        for t in t_series:
            if t <= t_target:
                traj_s.append(a0 * t**4 + a1 * t**3 + a2 * t**2 + a3 * t + a4)
                traj_ds.append(4 * a0 * t**3 + 3 * a1 * t**2 + 2 * a2 * t + a3)
                traj_dds.append(12 * a0 * t**2 + 6 * a1 * t + 2 * a2)
            else:
                traj_s.append(traj_s[-1] + traj_ds[-1] * self._dt)
                traj_ds.append(traj_ds[-1])
                traj_dds.append(0)

        return (
            np.array(traj_s[0 : self._max_horizon_step]),
            np.array(traj_ds[0 : self._max_horizon_step]),
            np.array(traj_dds[0 : self._max_horizon_step]),
        )

    # lateral planning
    def quintic_polynomial(self, l, dl, ddl, l_target, dl_t, ddl_t, t_target):
        # d = b0 * t^5 + b1 * t^4 + b2 * t^3 + b3 * t^2 + b4 * t + b5
        b5 = l
        b4 = dl
        b3 = ddl / 2.0

        A = np.array(
            [
                [t_target**5, t_target**4, t_target**3],
                [5 * t_target**4, 4 * t_target**3, 3 * t_target**2],
                [20 * t_target**3, 12 * t_target**2, 6 * t_target],
            ]
        )

        b = np.array(
            [
                l_target - b3 * t_target**2 - b4 * t_target - b5,
                dl_t - b4 - 2 * b3 * t_target,
                ddl_t - 2 * b3,
            ]
        )

        x = np.linalg.solve(A, b)

        b0 = x[0]
        b1 = x[1]
        b2 = x[2]

        t_series_1 = np.arange(0.0, t_target, self._dt) + self._dt
        t_series_2 = np.arange(t_series_1[-1], self._plan_t, self._dt) + self._dt

        t_series = np.concatenate((t_series_1, t_series_2))
        traj_l = [l]
        traj_dl = [dl]
        traj_ddl = [ddl]
        for t in t_series:
            if t <= t_target:
                traj_l.append(b0 * t**5 + b1 * t**4 + b2 * t**3 + b3 * t**2 + b4 * t + b5)
                traj_dl.append(5 * b0 * t**4 + 4 * b1 * t**3 + 3 * b2 * t**2 + 2 * b3 * t + b4)
                traj_ddl.append(20 * b0 * t**3 + 12 * b1 * t**2 + 6 * b2 * t + 2 * b3)
            else:
                traj_l.append(traj_l[-1])
                traj_dl.append(0)
                traj_ddl.append(0)

        return (
            np.array(traj_l[0 : self._max_horizon_step]),
            np.array(traj_dl[0 : self._max_horizon_step]),
            np.array(traj_ddl[0 : self._max_horizon_step]),
        )

    def quintic_polynomial_with_s(self, l, dl, ddl, l_target, dl_s, ddl_s, s_list):
        target_s = s_list[-1] - s_list[0]
        b5 = l
        b4 = dl
        b3 = ddl / 2.0

        A = np.array(
            [
                [target_s**5, target_s**4, target_s**3],
                [5 * target_s**4, 4 * target_s**3, 3 * target_s**2],
                [20 * target_s**3, 12 * target_s**2, 6 * target_s],
            ]
        )

        b = np.array(
            [
                l_target - b3 * target_s**2 - b4 * target_s - b5,
                dl_s - b4 - 2 * b3 * target_s,
                ddl_s - 2 * b3,
            ]
        )

        x = np.linalg.solve(A, b)

        b0 = x[0]
        b1 = x[1]
        b2 = x[2]

        s_series = s_list - s_list[0]
        traj_l = [l]
        traj_dl = [dl]
        traj_ddl = [ddl]
        for s in s_series:
            traj_l.append(b0 * s**5 + b1 * s**4 + b2 * s**3 + b3 * s**2 + b4 * s + b5)
            traj_dl.append(5 * b0 * s**4 + 4 * b1 * s**3 + 3 * b2 * s**2 + 2 * b3 * s + b4)
            traj_ddl.append(20 * b0 * s**3 + 12 * b1 * s**2 + 6 * b2 * s + 2 * b3)

        return (
            np.array(traj_l[: self._max_horizon_step]),
            np.array(traj_dl[: self._max_horizon_step]),
            np.array(traj_ddl[: self._max_horizon_step]),
        )

    def _compute_initial_states(
        self,
        x_0: STState,
        acc: float = None,
        steering_angle: float = None,
        logger: logging.Logger = None,
    ):
        """
        Computes the curvilinear initial states for the polynomial planner based on the Cartesian initial state
        :param x_0: The Cartesion state object representing the initial state of the vehicle
        :return: A tuple containing the initial longitudinal and lateral states (lon,lat)
        """
        # compute curvilinear position
        try:
            s, d = self._co.convert_to_curvilinear_coords(x_0.position[0], x_0.position[1])
        except:
            logger.debug("Initial state could not be transformed.")
            return None, None

        # factor for interpolation
        s_idx = np.argmax(self._co.ref_pos > s) - 1
        s_lambda = (s - self._co.ref_pos[s_idx]) / (self._co.ref_pos[s_idx + 1] - self._co.ref_pos[s_idx])

        # compute orientation in curvilinear coordinate frame
        ref_theta = np.unwrap(self._co.ref_theta)
        theta_cl = x_0.orientation - interpolate_angle(
            s,
            self._co.ref_pos[s_idx],
            self._co.ref_pos[s_idx + 1],
            ref_theta[s_idx],
            ref_theta[s_idx + 1],
        )

        # compute reference curvature
        kr = (self._co.ref_curv[s_idx + 1] - self._co.ref_curv[s_idx]) * s_lambda + self._co.ref_curv[s_idx]
        # compute reference curvature change
        kr_d = (self._co.ref_curv_d[s_idx + 1] - self._co.ref_curv_d[s_idx]) * s_lambda + self._co.ref_curv_d[s_idx]

        # compute initial ego curvature from initial steering angle
        kappa_0 = np.tan(steering_angle) / (self._vehicle_params.a + self._vehicle_params.b)

        # compute d' and d'' -> derivation after arclength (s): see Eq. (A.3) and (A.5) in Diss. Werling
        d_p = (1 - kr * d) * np.tan(theta_cl)
        d_pp = -(kr_d * d + kr * d_p) * np.tan(theta_cl) + ((1 - kr * d) / (math.cos(theta_cl) ** 2)) * (
            kappa_0 * (1 - kr * d) / math.cos(theta_cl) - kr
        )

        # compute s dot (s_velocity) and s dot dot (s_acceleration) -> derivation after time
        s_velocity = x_0.velocity * math.cos(theta_cl) / (1 - kr * d)
        if s_velocity < 0:
            logger.debug(
                "Initial state or reference incorrect! Curvilinear velocity is negative which indicates"
                "that the ego vehicle is not driving in the same direction as specified by the reference"
            )
            return None, None

        s_acceleration = acc
        s_acceleration -= (s_velocity**2 / math.cos(theta_cl)) * (
            (1 - kr * d) * np.tan(theta_cl) * (kappa_0 * (1 - kr * d) / (math.cos(theta_cl)) - kr)
            - (kr_d * d + kr * d_p)
        )
        s_acceleration /= (1 - kr * d) / (math.cos(theta_cl))

        # compute d dot (d_velocity) and d dot dot (d_acceleration)
        if self._low_vel_mode:
            # in LOW_VEL_MODE: d_velocity and d_acceleration are derivatives w.r.t arclength (s)
            d_velocity = d_p
            d_acceleration = d_pp
        else:
            # in HIGH VEL MODE: d_velocity and d_acceleration are derivatives w.r.t time
            d_velocity = x_0.velocity * math.sin(theta_cl)
            d_acceleration = s_acceleration * d_p + s_velocity**2 * d_pp

        x_0_lon: List[float] = [s, s_velocity, s_acceleration]
        x_0_lat: List[float] = [d, d_velocity, d_acceleration]

        return x_0_lon, x_0_lat

    def _get_cart_traj(self, frenet_traj: List = None, x_0: STState = None):
        traj_s, traj_ds, traj_dds, traj_l, traj_dl, traj_ddl = frenet_traj
        traj_x = []
        traj_y = []
        traj_theta = []
        traj_kappa = []
        traj_v = []
        traj_a = []

        for i in range(len(traj_s)):
            cart_coords = self._co.convert_to_cartesian_coords(traj_s[i], traj_l[i])
            if cart_coords is not None:
                x, y = cart_coords
            else:
                return None
            traj_x.append(x)
            traj_y.append(y)

            if traj_ds[i] > 0.001:
                dp = traj_dl[i] / traj_ds[i]
            else:
                dp = 0.0

            ddot = traj_ddl[i] - dp * traj_dds[i]

            if traj_ds[i] > 0.001:
                dpp = ddot / (traj_ds[i] ** 2)
            else:
                dpp = 0.0

            s_idx = np.argmax(self._co.ref_pos > traj_s[i]) - 1
            s_lambda = (traj_s[i] - self._co.ref_pos[s_idx]) / (self._co.ref_pos[s_idx + 1] - self._co.ref_pos[s_idx])

            if traj_ds[i] > 0.001:
                curv_theta = np.arctan2(dp, 1.0)
                cart_theta = curv_theta + interpolate_angle(
                    traj_s[i],
                    self._co.ref_pos[s_idx],
                    self._co.ref_pos[s_idx + 1],
                    self._co.ref_theta[s_idx],
                    self._co.ref_theta[s_idx + 1],
                )
                cart_theta = make_valid_orientation(cart_theta)
                traj_theta.append(cart_theta)
            else:
                if self._low_vel_mode:
                    curv_theta = np.arctan2(dp, 1.0)
                    cart_theta = curv_theta + interpolate_angle(
                        traj_s[i],
                        self._co.ref_pos[s_idx],
                        self._co.ref_pos[s_idx + 1],
                        self._co.ref_theta[s_idx],
                        self._co.ref_theta[s_idx + 1],
                    )
                    cart_theta = make_valid_orientation(cart_theta)
                    traj_theta.append(cart_theta)
                else:
                    (traj_theta.append(x_0.orientation) if i == 0 else traj_theta.append(traj_theta[-1]))
                    curv_theta = traj_theta[-1] - interpolate_angle(
                        traj_s[i],
                        self._co.ref_pos[s_idx],
                        self._co.ref_pos[s_idx + 1],
                        self._co.ref_theta[s_idx],
                        self._co.ref_theta[s_idx + 1],
                    )

            # Interpolate curvature of reference path k_r at current position
            k_r = (self._co.ref_curv[s_idx + 1] - self._co.ref_curv[s_idx]) * s_lambda + self._co.ref_curv[s_idx]
            # Interpolate curvature rate of reference path k_r_d at current position
            k_r_d = (self._co.ref_curv_d[s_idx + 1] - self._co.ref_curv_d[s_idx]) * s_lambda + self._co.ref_curv_d[
                s_idx
            ]

            # compute global curvature (see appendix A of Moritz Werling's PhD thesis)
            oneKrD = 1 - k_r * traj_l[i]
            cosTheta = math.cos(curv_theta)
            tanTheta = np.tan(curv_theta)
            cart_kappa = (dpp + (k_r * dp + k_r_d * traj_l[i]) * tanTheta) * cosTheta * (cosTheta / oneKrD) ** 2 + (
                cosTheta / oneKrD
            ) * k_r
            curv_kappa = cart_kappa - k_r
            traj_kappa.append(cart_kappa)
            # compute (global) Cartesian velocity
            cart_v = traj_ds[i] * (oneKrD / (math.cos(curv_theta)))
            traj_v.append(cart_v)

            # compute (global) Cartesian acceleration
            cart_a = traj_dds[i] * oneKrD / cosTheta + ((traj_ds[i] ** 2) / cosTheta) * (
                oneKrD * tanTheta * (cart_kappa * oneKrD / cosTheta - k_r) - (k_r_d * traj_l[i] + k_r * dp)
            )
            traj_a.append(cart_a)

        traj_list = [traj_x, traj_y, traj_theta, traj_kappa, traj_v, traj_a]

        if len(traj_x) < 1:
            return None

        return [np.array(item) for item in traj_list]

    def _stitch_trajectory(self, s_0: np.ndarray = None, l_0: np.ndarray = None) -> np.ndarray:
        # find the match state as planning initial state
        if self.trajectory is not None:
            diff_s = self.trajectory.frenet_s - s_0
            nearest_idx = np.argmax(diff_s > 0)
            x_0 = self.trajectory.cart_x[nearest_idx]
            y_0 = self.trajectory.cart_y[nearest_idx]
            new_s_0, new_l_0 = self._co.convert_to_curvilinear_coords(x_0, y_0)
            return new_s_0, new_l_0
        else:
            return s_0, l_0

    def _check_s_traj(
        self,
        traj_s: np.ndarray = None,
        traj_ds: np.ndarray = None,
        traj_dds: np.ndarray = None,
    ):
        # check traj_s is continuously inceasing
        if not np.all(np.diff(traj_s) > 0):
            # find the first index that is not increasing
            s_idx = np.argmax(np.diff(traj_s) <= 0)
            # find the first index that traj_ds is less than zero
            v_idx = np.argmax(traj_ds < 0)
            target_idx = min(s_idx, v_idx)
            if target_idx < 1:
                traj_ds[target_idx:] = KCONSTV
                traj_dds[target_idx:] = 0.0
            else:
                # reset the traj_s
                traj_ds[target_idx:] = traj_ds[target_idx - 1]
                traj_dds[target_idx:] = (traj_ds[target_idx:] - traj_ds[target_idx - 1 : -1]) / self._dt

            for i in range(target_idx + 1, len(traj_s)):
                traj_s[i] = traj_s[i - 1] + traj_ds[target_idx] * self._dt

        return traj_s, traj_ds, traj_dds

    def _valid_cart_traj(self, cartesian_traj: List[np.ndarray] = None, logger: logging.Logger = None):
        # check kappa for cat traj
        MAX_KAPPA = 0.2
        traj_x, traj_y, traj_theta, traj_kappa, traj_v, traj_a = cartesian_traj
        max_kappa = np.abs(traj_kappa).max()
        if max_kappa > MAX_KAPPA:
            logger.warning(f"Cartesian trajectory is invalid! kappa {max_kappa} is exceeded {MAX_KAPPA}!")
            return False

        return True

    def _check_valid_frenet_traj(self, frenet_traj: List = None):
        traj_s, traj_ds, traj_dds, traj_l, traj_dl, traj_ddl = frenet_traj

        # check traj_s is less than the max_s
        large_idx = np.argmax((traj_s - self.max_s) > 0)
        if large_idx > 1:
            traj_s = traj_s[:large_idx]
            traj_ds = traj_ds[:large_idx]
            traj_dds = traj_dds[:large_idx]
            traj_l = traj_l[:large_idx]
            traj_dl = traj_dl[:large_idx]
            traj_ddl = traj_ddl[:large_idx]

        # check traj_l is within the range
        # valid_min_idx = np.argmax((traj_l - self.min_l) < 0)
        # valid_max_idx = np.argmax((traj_l - self.max_l) > 0)
        # if traj_l[valid_min_idx] > self.min_l:
        #     valid_min_idx = None
        # if traj_l[valid_max_idx] < self.max_l:
        #     valid_max_idx = None
        # if valid_max_idx is not None or valid_min_idx is not None:
        #     re_compute_idx = (
        #         valid_max_idx if valid_max_idx is not None else valid_min_idx
        #     )
        #     target_d = np.clip(traj_l[re_compute_idx], self.min_l, self.max_l)
        #     target_max_l = min(target_d, self.max_l) if target_d > 0 else self.max_l
        #     target_min_l = max(target_d, self.min_l) if target_d < 0 else self.min_l
        #     for i in range(re_compute_idx, len(traj_l)):
        #         traj_l[i] = np.clip(traj_l[i], target_min_l, target_max_l)
        #         traj_dl[i - 1] = (traj_l[i] - traj_l[i - 1]) / self._dt
        #         traj_ddl[i - 1] = (traj_dl[i] - traj_dl[i - 1]) / self._dt
        #     traj_dl[-1] = traj_dl[-2]
        #     traj_ddl[-1] = traj_ddl[-2]

        frenet_traj = [traj_s, traj_ds, traj_dds, traj_l, traj_dl, traj_ddl]

        return frenet_traj


class RLReactivePlanner(ReactivePlanner):
    def __init__(self, config: ReactivePlannerConfiguration = None) -> None:
        super(RLReactivePlanner, self).__init__(config)
        # initialize and get logger
        initialize_logger(config)
        self.logger = logging.getLogger("RP_LOGGER")
        self._trajectory = None

    @property
    def trajectory(self):
        return self._trajectory

    def PlanTraj(
        self,
        vehicle: ContinuousVehicle = None,
        rescaled_action=None,
        collisionchecker=None,
        coordinate_system=None,
    ) -> PlanTrajectory:
        current_state = vehicle.state
        current_action = vehicle.current_action
        rp_state_cart = ReactivePlannerState(
            time_step=current_state.time_step,
            position=current_state.position,
            orientation=current_state.orientation,
            yaw_rate=current_state.yaw_rate,
            velocity=current_state.velocity,
            acceleration=current_action[1],
            steering_angle=current_action[0],
        )
        # reset planner state for re-planning
        self.reset(
            initial_state_cart=rp_state_cart,
            collision_checker=self.collision_checker,
            coordinate_system=self.coordinate_system,
        )

        optimal_traj = self.plan()
        # record state and input
        self.record_state_and_input(optimal_traj[0].state_list[1])

        cart_opt_traj = optimal_traj[0]
        frenet_opt_traj = optimal_traj[1]
        traj_x = []
        traj_y = []
        traj_theta = []
        traj_kappa = []
        traj_v = []
        traj_a = []
        traj_s = []
        traj_ds = []
        traj_dds = []
        traj_l = []
        traj_dl = []
        traj_ddl = []

        for cart_item, frenet_item in zip(cart_opt_traj.state_list, frenet_opt_traj.state_list):
            traj_x.append(cart_item.position[0])
            traj_y.append(cart_item.position[1])
            traj_theta.append(cart_item.orientation)
            traj_kappa.append(frenet_item.kappa)
            traj_v.append(cart_item.velocity)
            traj_a.append(cart_item.acceleration)
            traj_s.append(frenet_item.frenet_s)
            traj_ds.append(frenet_item.frenet_s_dot)
            traj_dds.append(frenet_item.frenet_s_ddot)
            traj_l.append(frenet_item.frenet_d)
            traj_dl.append(frenet_item.frenet_d_dot)
            traj_ddl.append(frenet_item.frenet_d_ddot)

        # plt.figure(2)
        # plt.plot(traj_x, traj_y)
        # plt.title('x-y')
        # plt.show()

        frenet_traj = [traj_s, traj_ds, traj_dds, traj_l, traj_dl, traj_ddl]
        cart_traj = [traj_x, traj_y, traj_theta, traj_kappa, traj_v, traj_a]
        plan_traj = PlanTrajectory(
            cart_traj=[np.array(item) for item in cart_traj],
            frenet_traj=[np.array(item) for item in frenet_traj],
        )

        self._trajectory = plan_traj

        return plan_traj

    def ReferenceTraj(self, navigator: Navigator = None):
        reference_pts = navigator.route.reference_path
        # compute theta
        diff_x = reference_pts[1:, 0] - reference_pts[:-1, 0]
        diff_y = reference_pts[1:, 1] - reference_pts[:-1, 1]
        theta = np.arctan2(diff_y, diff_x)
        theta = np.append(theta, theta[-1])
        frenet_ds = 15 * np.ones(len(reference_pts[:, 0]))
        zeros = np.zeros(len(reference_pts[:, 0]))
        frenet_traj = [zeros, frenet_ds, zeros, zeros, zeros, zeros]
        cart_traj = [
            reference_pts[:, 0],
            reference_pts[:, 1],
            theta,
            zeros,
            zeros,
            zeros,
        ]
        # [traj_s, traj_ds, traj_dds, traj_l, traj_dl, traj_ddl]
        # [traj_x, traj_y, traj_theta, traj_kappa, traj_v, traj_a]
        ref_traj = PlanTrajectory(cart_traj=cart_traj, frenet_traj=frenet_traj)

        return ref_traj
