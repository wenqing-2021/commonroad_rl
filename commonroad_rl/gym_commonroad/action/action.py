"""
Module containing the action base class
"""

from commonroad_route_planner.route import Route
from commonroad.scenario.trajectory import STState
import gymnasium as gym
from typing import Union
import copy
import logging
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from numpy import ndarray
from commonroad_rl.gym_commonroad.action.vehicle import *
from commonroad_rl.gym_commonroad.action.planner import (
    PolynomialPlanner,
    RLReactivePlanner,
    PlanTrajectory,
)
from commonroad_reach.data_structure.reach.reach_interface import ReachableSetInterface
from commonroad_reach.data_structure.reach.reach_node import ReachNodeMultiGeneration
from commonroad_rl.gym_commonroad.action.controller import Controller
from commonroad_rl.gym_commonroad.utils.navigator import Navigator
from commonroad_rp.utility.utils_coordinate_system import CoordinateSystem
from commonroad_rp.utility.config import ReactivePlannerConfiguration


def _rotate_to_curvi(vector: np.ndarray, local_ccosy: CurvilinearCoordinateSystem, pos: np.ndarray) -> np.ndarray:
    """
    Function to rotate a vector in the curvilinear system to its counterpart in the normal coordinate system

    :param vector: The vector in question
    :returns: The rotated vector
    """
    try:
        long, _ = local_ccosy.convert_to_curvilinear_coords(pos[0], pos[1])
    except ValueError:
        long = 0.0

    tangent = local_ccosy.tangent(long)
    theta = np.math.atan2(tangent[1], tangent[0])
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return np.matmul(rot_mat, vector)


class Action(ABC):
    """
    Description:
        Abstract base class of all action spaces
    """

    def __init__(self):
        """Initialize empty object"""
        super().__init__()
        self.vehicle = None

    @abstractmethod
    def step(
        self,
        action: Union[np.ndarray, int],
        local_ccosy: CurvilinearCoordinateSystem = None,
    ) -> None:
        """
        Function which acts on the current state and generates the new state
        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        """
        pass


class DiscreteAction(Action):
    """
    Description:
        Abstract base class of all discrete action spaces. Each high-level discrete
        action is converted to a low-level trajectory by a specified planner.
    """

    def __init__(self, vehicle_params_dict: dict, long_steps: int, lat_steps: int):
        """Initialize empty object"""
        super().__init__()

        assert (
            VehicleModel(vehicle_params_dict["vehicle_model"]) == VehicleModel.PM
        ), "ERROR in ACTION INITIALIZATION: DiscreteAction only supports the PM vehicle_type no"

        assert long_steps % 2 != 0 and lat_steps % 2 != 0, (
            "ERROR in ACTION INITIALIZATION: The discrete steps for longitudinal and lateral action "
            "have to be odd numbers, so constant velocity without turning is an possible action"
        )

        self.vehicle = ContinuousVehicle(vehicle_params_dict)
        self.local_ccosy = None

    def reset(self, initial_state: State, dt: float) -> None:
        """
        resets the vehicle
        :param initial_state: initial state
        :param dt: time step size of scenario
        """
        self.vehicle.reset(initial_state, dt)

    def step(
        self,
        action: Union[np.ndarray, int],
        local_ccosy: CurvilinearCoordinateSystem = None,
    ) -> None:
        """
        Function which acts on the current state and generates the new state

        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        """
        self.local_ccosy = local_ccosy
        state = self.get_new_state(action)
        self.vehicle.set_current_state(state)

    @abstractmethod
    def get_new_state(self, action: Union[np.ndarray, int]) -> State:
        """function which return new states given the action and current state"""
        pass

    def _propagate(self, control_input: np.array):
        # Rotate the action according to the curvilinear coordinate system
        if self.local_ccosy is not None:
            control_input = _rotate_to_curvi(control_input, self.local_ccosy, self.vehicle.state.position)

        # get the next state from the PM model
        return self.vehicle.get_new_state(control_input, "acceleration")


class DiscretePMJerkAction(DiscreteAction):
    """
    Description:
        Discrete / High-level action class with point mass model and jerk control
    """

    def __init__(self, vehicle_params_dict: dict, long_steps: int, lat_steps: int):
        """
        Initialize object
        :param vehicle_params_dict: vehicle parameter dictionary
        :param long_steps: number of discrete longitudinal jerk steps
        :param lat_steps: number of discrete lateral jerk steps
        """
        super().__init__(vehicle_params_dict, long_steps, lat_steps)

        self.j_max = 10  # set the maximum jerk
        self.long_step_size = (self.j_max * 2) / (long_steps - 1)
        self.lat_step_size = (self.j_max * 2) / (lat_steps - 1)
        self.action_mapping_long = {}
        self.action_mapping_lat = {}

        for idx in range(long_steps):
            self.action_mapping_long[idx] = self.j_max - (idx * self.long_step_size)

        for idx in range(lat_steps):
            self.action_mapping_lat[idx] = self.j_max - (idx * self.lat_step_size)

    def get_new_state(self, action: Union[np.ndarray, int]) -> State:
        """
        calculation of next state depending on the discrete action
        :param action: discrete action
        :return: next state
        """
        # map discrete action to jerk and calculate a
        # correct rescale in order to make 0 acceleration achievable again when sign of acc switches
        a_long = self.action_mapping_long[action[0]] * self.vehicle.dt + self.vehicle.state.acceleration
        if (
            self.vehicle.state.acceleration != 0
            and np.sign(a_long) != np.sign(self.vehicle.state.acceleration)
            and (np.abs(a_long) % (self.long_step_size * self.vehicle.dt)) != 0
        ):
            a_long = (
                self.action_mapping_long[action[0]] * self.vehicle.dt
                + self.vehicle.state.acceleration
                - np.sign(a_long) * (np.abs(a_long) % (self.long_step_size * self.vehicle.dt))
            )

        a_lat = self.action_mapping_lat[action[1]] * self.vehicle.dt + self.vehicle.state.acceleration_y
        if (
            self.vehicle.state.acceleration_y != 0
            and np.sign(a_lat) != np.sign(self.vehicle.state.acceleration_y)
            and (np.abs(a_lat) % (self.lat_step_size * self.vehicle.dt)) != 0
        ):
            a_lat = (
                self.action_mapping_long[action[1]] * self.vehicle.dt
                + self.vehicle.state.acceleration_y
                - np.sign(a_lat) * (np.abs(a_lat) % (self.lat_step_size * self.vehicle.dt))
            )

        control_input = np.array([a_long, a_lat])

        return self._propagate(control_input)


class DiscretePMAction(DiscreteAction):
    """
    Description:
        Discrete / High-level action class with point mass model
    """

    def __init__(self, vehicle_params_dict: dict, long_steps: int, lat_steps: int):
        """
        Initialize object
        :param vehicle_params_dict: vehicle parameter dictionary
        :param long_steps: number of discrete acceleration steps
        :param lat_steps: number of discrete turning steps
        """
        super().__init__(vehicle_params_dict, long_steps, lat_steps)

        a_max = self.vehicle.parameters.longitudinal.a_max
        a_long_steps = (a_max * 2) / (long_steps - 1)
        a_lat_steps = (a_max * 2) / (lat_steps - 1)

        self.action_mapping_long = {}
        self.action_mapping_lat = {}

        for idx in range(long_steps):
            self.action_mapping_long[idx] = a_max - (idx * a_long_steps)

        for idx in range(lat_steps):
            self.action_mapping_lat[idx] = a_max - (idx * a_lat_steps)

    def propogate_one_state(self, state: State, action: Union[np.ndarray, int]):
        """
        Used to generate a trajectory from a given action
        :param state:
        :param action:
        :return:
        """
        control_input = np.array([self.action_mapping_long[action[0]], self.action_mapping_lat[action[1]]])
        # Rotate the action according to the curvilinear coordinate system
        if self.local_ccosy is not None:
            control_input = _rotate_to_curvi(control_input, self.local_ccosy, state.position)

        return self.vehicle.propagate_one_time_step(state, control_input, "acceleration")

    def get_new_state(self, action: Union[np.ndarray, int]) -> State:
        """
        calculation of next state depending on the discrete action
        :param action: discrete action
        :return: next state
        """
        return self.propogate_one_state(state=self.vehicle.state, action=action)


class ContinuousAction(Action):
    """
    Description:
        Module for continuous action space; actions correspond to vehicle control inputs
    """

    def __init__(self, params_dict: dict, action_dict: dict):
        """Initialize object"""
        super().__init__()
        # create vehicle object
        self.action_base = action_dict["action_base"]
        self._continous_collision_check = action_dict.get("continuous_collision_checking", True)
        self.vehicle = ContinuousVehicle(params_dict, continuous_collision_checking=self._continous_collision_check)

    def _set_rescale_factors(self):
        a_max = self.vehicle.parameters.longitudinal.a_max
        # rescale factors for PM model
        if self.vehicle.vehicle_model == VehicleModel.PM:
            self._rescale_factor = np.array([a_max, a_max])
            self._rescale_bias = 0.0
        # rescale factors for KS model
        elif self.vehicle.vehicle_model == VehicleModel.KS:
            steering_v_max = self.vehicle.parameters.steering.v_max
            steering_v_min = self.vehicle.parameters.steering.v_min
            self._rescale_factor = np.array([(steering_v_max - steering_v_min) / 2.0, a_max])
            self._rescale_bias = np.array([(steering_v_max + steering_v_min) / 2.0, 0.0])
        # rescale factors for YawRate model
        elif self.vehicle.vehicle_model == VehicleModel.YawRate:
            yaw_rate_max = self.vehicle.parameters.yaw.v_max = np.abs(
                self.vehicle.parameters.longitudinal.a_max / (self.vehicle.state.velocity + 1e-6)
            )
            yaw_rate_min = self.vehicle.parameters.yaw.v_min = -self.vehicle.parameters.yaw.v_max

            self._rescale_factor = np.array([(yaw_rate_max - yaw_rate_min) / 2.0, a_max])
            self._rescale_bias = np.array([0.0, 0.0])
        elif self.vehicle.vehicle_model == VehicleModel.QP:
            ub, lb = (
                self.vehicle.vehicle_dynamic.input_bounds.ub,
                self.vehicle.vehicle_dynamic.input_bounds.lb,
            )
            self._rescale_factor = (ub - lb) / 2
            self._rescale_bias = (ub + lb) / 2
        elif self.vehicle.vehicle_model == VehicleModel.PMNonlinear:
            # TODO: check with Niklas which bounds are used in the reachable sets
            self._rescale_factor = np.array([a_max, 2.0])
            self._rescale_bias = np.array([0.0, 0.0])
        else:
            raise ValueError(
                f"action.py/_set_rescale_factors: rescale factors not defined for model {self.vehicle.vehicle_model}"
            )

    def reset(self, initial_state: State, dt: float) -> None:
        self.vehicle.reset(initial_state, dt)
        self._set_rescale_factors()

    def step(
        self,
        action: Union[np.ndarray, int],
        local_ccosy: CurvilinearCoordinateSystem = None,
    ) -> None:
        """
        Function which acts on the current state and generates the new state

        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        :return: New state of ego vehicle
        """
        rescaled_action = self.rescale_action(action)
        new_state = self.vehicle.get_new_state(rescaled_action, self.action_base)
        self.vehicle.set_current_state(new_state)
        if self.vehicle.vehicle_model == VehicleModel.YawRate:
            self._set_rescale_factors()

    def rescale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range

        :param action: action from the CommonroadEnv.
        :return: rescaled action
        """
        assert hasattr(self, "_rescale_bias") and hasattr(
            self, "_rescale_factor"
        ), "<ContinuousAction/rescale_action>: rescale factors not set, please run action.reset() first"

        return self._rescale_factor * action + self._rescale_bias


class ParameterAction(ContinuousAction):
    """
    Description:
        Module for continuous action space; actions correspond to polynomial function parameters
    """

    @property
    def matched_state(self):
        return self._last_matched_state

    @property
    def current_trajectory(self) -> PlanTrajectory:
        if self.trajectory_history.__len__() == 0:
            return None
        return self.trajectory_history[-1]

    @property
    def current_plan_trajectory(self) -> PlanTrajectory:
        if self.plan_trajecotry_history.__len__() == 0:
            return None
        return self.plan_trajecotry_history[-1]

    @property
    def current_refine_trajectory(self) -> PlanTrajectory:
        if self.refine_traj_history.__len__() == 0:
            return None
        return self.refine_traj_history[-1]

    def __init__(self, params_dict: dict, action_dict: dict):
        """Initialize object"""
        # create vehicle object
        self.params_range = action_dict["parameters_range"]
        self.action_base = action_dict["action_base"]
        self._continous_collision_check = action_dict.get("continuous_collision_checking", True)
        self.vehicle = ContinuousVehicle(params_dict, continuous_collision_checking=self._continous_collision_check)
        self.trajectory_history = []
        self.refine_traj_history = []
        self.plan_trajecotry_history = []
        self.planner = PolynomialPlanner(self.vehicle.parameters)
        self.controller = Controller(
            config=action_dict["control_config"],
            vehicle_parameters=self.vehicle.parameters,
            control_frequency=action_dict["control_frequency"],
        )
        self.goal_region = None
        self._scenario_dt = 0.1  # s
        self._last_matched_state = None
        self._act_lon_size = action_dict["long_steps"] - 1  # 3 - 1
        self._act_lat_size = action_dict["lat_steps"] - 1  # 6 -1

    def reset(
        self,
        dt: float,
        local_ccosy: CurvilinearCoordinateSystem = None,
        goal_region=None,
    ) -> None:
        # ego vehicle reset has been done in the environment
        self._scenario_dt = dt
        self._set_rescale_factors()
        self.planner.reset(self.params_range["max_plan_time"], self._scenario_dt, local_ccosy)
        self._last_matched_state = None
        self.edit_cost = None
        goal_s, goal_l = local_ccosy.convert_to_curvilinear_coords(goal_region[0], goal_region[1])
        self.goal_region = (goal_s, goal_l)
        self.could_to_goal = False
        self.not_plan = False
        self.trajectory_history.clear()
        self.refine_traj_history.clear()

    def _set_rescale_factors(self):
        max_plan_time = self.params_range["max_plan_time"]
        min_lat_time = 1.0
        max_long_v = self.params_range["max_long_v"]
        min_long_v = self.params_range["min_long_v"]
        max_lat_dis = self.params_range["max_lat_dis"]
        min_lat_dis = -max_lat_dis

        # ---------------- discrete action
        self.delta_v = (max_long_v - min_long_v) / self._act_lon_size  # 3.0
        self.delta_l = (max_lat_dis - min_lat_dis) / self._act_lat_size  # 1.2
        # ---------------- discrete action

        # ---------------- continuous action
        self._rescale_factor = np.array(
            [
                (max_long_v - min_long_v) / 2.0,
                (max_lat_dis - min_lat_dis) / 2.0,
                # (max_plan_time - min_lat_time) / 2.0,
            ]
        )

        self._rescale_bias = np.array(
            [
                (max_long_v + min_long_v) / 2.0,
                (max_lat_dis + min_lat_dis) / 2.0,
                # (max_plan_time + min_lat_time) / 2.0,
            ]
        )
        # ---------------- continuous action

    # ---------------- discrete action
    def rescale_action(self, action: ndarray) -> ndarray:
        # only for the discrete action
        # if action < 6:
        #     act_target_l = -self.params_range["max_lat_dis"] + self.delta_l * action
        #     delta_v = 3
        # else:
        #     act_target_l = 0
        #     delta_v = self.params_range["min_long_v"] + self.delta_v * (action - 6)
        act_target_l = -self.params_range["max_lat_dis"] + self.delta_l * action

        return np.array([0, act_target_l])

    # ---------------- discrete action

    def step(
        self,
        action: Union[np.ndarray, int],
        logger: logging.Logger = None,
        reach_interface: ReachableSetInterface = None,
        only_plan: bool = False,
        ilqr_traj: np.ndarray = None,
    ) -> None:
        """
        Function which acts on the current state and generates the new state

        :param action: current action
        :param local_ccosy: Current curvilinear coordinate system
        :return: New state of ego vehicle
        """
        v_max = None
        if (
            reach_interface is not None
            and reach_interface.reachable_set is not None
            and len(reach_interface.reachable_set) > 0
        ):
            end_step = reach_interface.step_end
            last_reach_set = reach_interface.reachable_set_at_step(end_step - 1)
            if last_reach_set is not None and len(last_reach_set) > 0:
                v_max = last_reach_set[-1].v_lon_max
        if v_max is not None:
            delta_v = v_max - self.vehicle.state.velocity
        else:
            delta_v = 0
        rescaled_action = self.rescale_action(action)
        rescaled_action[0] = delta_v
        refine_plan_trajectory: PlanTrajectory = None

        # # check if could reach the goal region
        # if self.trajectory_history.__len__() > 0:
        #     last_trajectory = self.trajectory_history[-1]
        #     if last_trajectory is not None:
        #         if last_trajectory.frenet_s[-1] >= self.goal_region[0]:
        #             self.could_to_goal = True
        # if self.could_to_goal:
        #     rescaled_action[1] = self.goal_region[1]

        if ilqr_traj is None:
            plan_trajectory: PlanTrajectory = self.planner.PlanTraj(
                vehicle=self.vehicle, rescaled_action=rescaled_action, logger=logger
            )
            if plan_trajectory is not None:
                self.plan_trajecotry_history.append(plan_trajectory)
            else:
                if self.plan_trajecotry_history.__len__() > 0:
                    plan_trajectory = self.plan_trajecotry_history[-1]

            nodes_vertices = None
            if reach_interface is not None and plan_trajectory is not None:
                refine_plan_trajectory, nodes_vertices = self.refine_trajectory(reach_interface, plan_trajectory)

                if refine_plan_trajectory is not None:
                    self.refine_traj_history.append(refine_plan_trajectory)

                if refine_plan_trajectory is None and self.refine_traj_history.__len__() > 0:
                    refine_plan_trajectory = self.refine_traj_history[-1]

        else:
            # construct PlanTrajectory from ilqr_traj
            if isinstance(ilqr_traj, PlanTrajectory):
                plan_trajectory = ilqr_traj
            else:
                plan_trajectory = self._convert_ilqr2plantraj(ilqr_trajectory=ilqr_traj, logger=logger)
        # planning trajectory
        track_trajectory = refine_plan_trajectory if refine_plan_trajectory is not None else plan_trajectory
        if refine_plan_trajectory is not None:
            # cacluate the edit cost
            previous_l = rescaled_action[1]
            current_l = track_trajectory.frenet_l[-1]
            self.edit_cost = np.abs(previous_l - current_l)

        # if self.could_to_goal and plan_trajectory is not None:
        #     track_trajectory = plan_trajectory
        #     self.not_plan = True

        # if self.not_plan:
        #     track_trajectory = self.trajectory_history[-1]

        if only_plan:
            return track_trajectory, nodes_vertices

        if track_trajectory is not None:
            self.trajectory_history.append(track_trajectory)
        elif self.trajectory_history.__len__() > 0:
            track_trajectory = self.trajectory_history[-1]
        else:
            return True
        # update vehicle state
        t_step = 0
        while t_step < self._scenario_dt:
            # match trajectory
            matched_state = self.controller.match_trajectory(self.vehicle.state, track_trajectory, t_step)
            # logger.debug(
            #     f'matched_state: vecl is {matched_state.velocity}'
            # )
            # compute control input
            control_input = self.controller.compute_control_input(self.vehicle.state, matched_state)
            # logger.debug(
            #     f'control_input: acc is {control_input[1]}, steer_angle is {control_input[0]}')
            # update vehicle state
            new_state = self.vehicle.get_new_state_fix_step(control_input, self.action_base)
            self.vehicle.set_current_state(new_state)
            t_step += self.controller.control_dt

        # NOTE: we use fix step, therefore, we need to plus one
        self.vehicle.state.time_step += 1

        self._last_matched_state = matched_state
        if self.vehicle.state.has_value("slip_angle"):
            logger.debug(
                f"slip_angle is {self.vehicle.state.slip_angle}, yaw rate is {self.vehicle.state.yaw_rate}, orientation is {self.vehicle.state.orientation}"
            )
        else:
            logger.debug(f"orientation is {self.vehicle.state.orientation}, yaw rate is {self.vehicle.state.yaw_rate}")

        return False

    def refine_trajectory(
        self, reach_interface: ReachableSetInterface, plan_trajectory: PlanTrajectory
    ) -> Tuple[PlanTrajectory, List]:
        def _proj_out_of_domain(position, ccosy):
            eps = 0.0001
            curvi_coords_of_projection_domain = np.array(ccosy.curvilinear_projection_domain())

            longitudinal_min, normal_min = np.min(curvi_coords_of_projection_domain, axis=0) + eps
            longitudinal_max, normal_max = np.max(curvi_coords_of_projection_domain, axis=0) - eps
            normal_center = (normal_min + normal_max) / 2
            bounding_points = np.array(
                [
                    ccosy.convert_to_cartesian_coords(longitudinal_min, normal_center),
                    ccosy.convert_to_cartesian_coords(longitudinal_max, normal_center),
                ]
            )
            rel_positions = position - np.array([bounding_point for bounding_point in bounding_points])
            distances = np.linalg.norm(rel_positions, axis=1)

            if distances[0] < distances[1]:
                # Nearer to the first bounding point
                rel_pos_to_domain = -1
                long_dist = longitudinal_min + np.dot(ccosy.tangent(longitudinal_min), rel_positions[0])
                lat_dist = normal_center + np.dot(ccosy.normal(longitudinal_min), rel_positions[0])
            else:
                # Nearer to the last bounding point
                rel_pos_to_domain = 1
                long_dist = longitudinal_max + np.dot(ccosy.tangent(longitudinal_max), rel_positions[1])
                lat_dist = normal_center + np.dot(ccosy.normal(longitudinal_max), rel_positions[1])

            return np.array([long_dist, lat_dist]), rel_pos_to_domain

        def find_nodes_v_range(nodes: List[ReachNodeMultiGeneration]):
            min_v_lon = np.inf
            max_v_lon = -np.inf
            for node in nodes:
                min_v_lon = min(min_v_lon, node.v_lon_min)
                max_v_lon = max(max_v_lon, node.v_lon_max)
            return min_v_lon, max_v_lon

        current_step = self.vehicle.state.time_step
        # loop the time horizon
        end_step = current_step + int(self.params_range["max_plan_time"] / self._scenario_dt)

        def find_nodes_l_range(nodes: List[ReachNodeMultiGeneration], target_s):
            for node in nodes:
                if node.p_lon_min <= target_s <= node.p_lon_max:
                    return node.p_lat_min, node.p_lat_max, node

            return nodes[0].p_lat_min, nodes[0].p_lat_max, nodes[0]

        def convert_cart_pos(position_rectangle, clcs):
            p_lon_min, p_lat_min, p_lon_max, p_lat_max = position_rectangle.bounds
            sl_bound = [
                (p_lon_min, p_lat_min),
                (p_lon_max, p_lat_min),
                (p_lon_max, p_lat_max),
                (p_lon_min, p_lat_max),
            ]
            # safa project
            all_vertex = []
            for sl_vertex in sl_bound:
                vertex = clcs.convert_to_cartesian_coords(sl_vertex[0], sl_vertex[1])
                if vertex is None:
                    vertex, _ = _proj_out_of_domain(sl_vertex, clcs.ccosy)

                all_vertex.append(vertex)

            try:
                left_midd_pts = (all_vertex[2] + all_vertex[3]) / 2
                right_midd_pts = (all_vertex[0] + all_vertex[1]) / 2
            except:
                print(f"all vertex is {all_vertex}")

            vertex_arry = np.array(all_vertex)

            return np.vstack((vertex_arry, left_midd_pts, right_midd_pts))

        plan_trajectory_iter = copy.deepcopy(plan_trajectory)
        end_step = min(end_step, reach_interface.step_end)
        s_0 = self.planner.initial_state_dict["s_0"]
        ds_0 = self.planner.initial_state_dict["ds_0"]
        dds_0 = self.planner.initial_state_dict["dds_0"]
        l_0 = self.planner.initial_state_dict["l_0"]
        dl_0 = self.planner.initial_state_dict["dl_0"]
        ddl_0 = self.planner.initial_state_dict["ddl_0"]
        plan_T = self.params_range["max_plan_time"]
        original_target_v = self.planner.initial_state_dict["target_v"]
        last_list_nodes = reach_interface.reachable_set_at_step(end_step - 1)
        min_v_T, max_v_T = find_nodes_v_range(last_list_nodes)
        valid_target_v = max(min(max_v_T, original_target_v), min_v_T)
        t_series = np.linspace(0, plan_T - self._scenario_dt, int(plan_T / self._scenario_dt))
        # loop the time horizon
        nodes_vertices = []
        first_list_nodes = reach_interface.reachable_set_at_step(current_step)
        if len(first_list_nodes) < 1:
            # no reachable set at the step
            return None, None
        first_vertices = convert_cart_pos(first_list_nodes[0].position_rectangle, self.planner.coordinate_system)
        nodes_vertices.append(first_vertices)
        for step in range(current_step + 6, end_step):
            # get the reachable set at the step
            list_nodes = reach_interface.reachable_set_at_step(step)
            # longidutinal trajectory refinement
            ## find the min_v_lon in each node
            min_v_lon, max_v_lon = find_nodes_v_range(list_nodes)
            ## check the plan_trajectory
            dot_s = plan_trajectory_iter.frenet_ds[step - current_step]
            plan_t = (step - current_step) * self._scenario_dt
            if not (dot_s >= min_v_lon and dot_s <= max_v_lon):
                valid_dot_s = max(min(max_v_lon, dot_s), min_v_lon)
                A = np.array(
                    [
                        [4 * plan_T * plan_T * plan_T, 3 * plan_T * plan_T],
                        [4 * plan_t * plan_t * plan_t, 3 * plan_t * plan_t],
                    ]
                )
                B = np.array(
                    [
                        [valid_target_v - ds_0 - dds_0 * plan_T],
                        [valid_dot_s - ds_0 - dds_0 * plan_t],
                    ]
                )
                X = np.linalg.solve(A, B)
                a0 = X[0]
                a1 = X[1]
                s_traj = a0 * t_series**4 + a1 * t_series**3 + dds_0 / 2 * t_series**2 + ds_0 * t_series + s_0
                dot_s_traj = 4 * a0 * t_series**3 + 3 * a1 * t_series**2 + dds_0 * t_series + ds_0
                ddot_s_traj = 12 * a0 * t_series**2 + 6 * a1 * t_series + dds_0
                plan_trajectory_iter.frenet_s = s_traj
                plan_trajectory_iter.frenet_ds = dot_s_traj
                plan_trajectory_iter.frenet_dds = ddot_s_traj

            # lateral trajectory refinement
            ## find the target node
            step_target_s = plan_trajectory_iter.frenet_s[step - current_step]
            min_l, max_l, node = find_nodes_l_range(list_nodes, step_target_s)
            # store the node vertices for cilqr planner
            vertices_i = convert_cart_pos(node.position_rectangle, self.planner.coordinate_system)
            if vertices_i is None:
                return None, None

            nodes_vertices.append(vertices_i)
            ## check the plan_trajectory
            s_traj = plan_trajectory_iter.frenet_s
            l = plan_trajectory_iter.frenet_l[step - current_step]
            if not (l >= min_l and l <= max_l):
                valid_l = max(min(max_l, l), min_l)
                b5 = l_0
                b4 = dl_0
                b3 = ddl_0 / 2.0
                if not self.planner._low_vel_mode:
                    lateral_series = t_series
                    A = np.array(
                        [
                            [plan_t**5, plan_t**4, plan_t**3],
                            [5 * plan_T**4, 4 * plan_T**3, 3 * plan_T**2],
                            [20 * plan_T**3, 12 * plan_T**2, 6 * plan_T],
                        ]
                    )

                    B = np.array(
                        [
                            valid_l - b3 * plan_t**2 - b4 * plan_t - b5,
                            -b4 - 2 * b3 * plan_T,
                            -2 * b3,
                        ]
                    )
                    X = np.linalg.solve(A, B)
                    b0 = X[0]
                    b1 = X[1]
                    b2 = X[2]
                    l = (
                        b0 * lateral_series**5
                        + b1 * lateral_series**4
                        + b2 * lateral_series**3
                        + b3 * lateral_series**2
                        + b4 * lateral_series
                        + b5
                    )
                    dot_l = (
                        5 * b0 * lateral_series**4
                        + 4 * b1 * lateral_series**3
                        + 3 * b2 * lateral_series**2
                        + 2 * b3 * lateral_series
                        + b4
                    )
                    ddot_l = (
                        20 * b0 * lateral_series**3 + 12 * b1 * lateral_series**2 + 6 * b2 * lateral_series + 2 * b3
                    )
                    plan_trajectory_iter.frenet_l = l
                    plan_trajectory_iter.frenet_dl = dot_l
                    plan_trajectory_iter.frenet_ddl = ddot_l

                else:
                    if s_traj[0] < 1e-3:
                        lateral_series = s_traj
                        plan_S = s_traj[-1]
                    else:
                        lateral_series = s_traj - s_traj[0]
                        plan_S = s_traj[-1] - s_traj[0]

                    plan_s = step_target_s
                    A = np.array(
                        [
                            [plan_s**5, plan_s**4, plan_s**3],
                            [5 * plan_S**4, 4 * plan_S**3, 3 * plan_S**2],
                            [20 * plan_S**3, 12 * plan_S**2, 6 * plan_S],
                        ]
                    )

                    B = np.array(
                        [
                            valid_l - b3 * plan_s**2 - b4 * plan_s - b5,
                            -b4 - 2 * b3 * plan_S,
                            -2 * b3,
                        ]
                    )
                    X = np.linalg.solve(A, B)
                    b0 = X[0]
                    b1 = X[1]
                    b2 = X[2]
                    l = (
                        b0 * lateral_series**5
                        + b1 * lateral_series**4
                        + b2 * lateral_series**3
                        + b3 * lateral_series**2
                        + b4 * lateral_series
                        + b5
                    )
                    dot_l = (
                        5 * b0 * lateral_series**4
                        + 4 * b1 * lateral_series**3
                        + 3 * b2 * lateral_series**2
                        + 2 * b3 * lateral_series
                        + b4
                    )
                    ddot_l = (
                        20 * b0 * lateral_series**3 + 12 * b1 * lateral_series**2 + 6 * b2 * lateral_series + 2 * b3
                    )
                    plan_trajectory_iter.frenet_l = l
                    plan_trajectory_iter.frenet_dl = dot_l
                    plan_trajectory_iter.frenet_ddl = ddot_l

        frenet_traj = [
            plan_trajectory_iter.frenet_s,
            plan_trajectory_iter.frenet_ds,
            plan_trajectory_iter.frenet_dds,
            plan_trajectory_iter.frenet_l,
            plan_trajectory_iter.frenet_dl,
            plan_trajectory_iter.frenet_ddl,
        ]

        cart_traj = self.planner._get_cart_traj(frenet_traj)
        if cart_traj is None:
            return None, None
        final_traj = PlanTrajectory(frenet_traj=frenet_traj, cart_traj=cart_traj, dt=self._scenario_dt)

        # update the plan_trajectory
        return final_traj, nodes_vertices[1:]

    def _convert_ilqr2plantraj(self, ilqr_trajectory: np.ndarray, logger: logging.Logger = None) -> PlanTrajectory:
        """
        Convert the ilqr trajectory to PlanTrajectory
        :param ilqr_trajectory: ilqr trajectory
        :return: PlanTrajectory
        """
        cart_x = ilqr_trajectory[:, 0]
        cart_y = ilqr_trajectory[:, 1]
        cart_theta = ilqr_trajectory[:, 2]
        cart_v = ilqr_trajectory[:, 3]
        cart_a = ilqr_trajectory[:, 4]
        cart_steer_angle = ilqr_trajectory[:, 5]
        cart_kappa = np.tan(cart_steer_angle) / (self.vehicle.parameters.a + self.vehicle.parameters.b)

        wheel_base = self.vehicle.parameters.a + self.vehicle.parameters.b
        frenet_s = []
        frenet_ds = []
        frenet_dds = []
        frenet_l = []
        frenet_dl = []
        frenet_ddl = []
        for x, y, theta, v, a, steer_angle in zip(cart_x, cart_y, cart_theta, cart_v, cart_a, cart_steer_angle):
            st_state = STState(position=(x, y), orientation=theta, velocity=v)
            low_vel_mode = True if v < 4 else False
            state_lon, state_lat = PolynomialPlanner.compute_frenet_states(
                x_0=st_state,
                acc=a,
                steering_angle=steer_angle,
                logger=logger,
                CLCS=self.planner._co,
                whee_base=wheel_base,
                low_vel_mode=low_vel_mode,
            )
            s, ds, dds = state_lon
            l, dl, ddl = state_lat
            frenet_s.append(s)
            frenet_ds.append(ds)
            frenet_dds.append(dds)
            frenet_l.append(l)
            frenet_dl.append(dl)
            frenet_ddl.append(ddl)

        # construct frenet trajectory
        cart_traj = [cart_x, cart_y, cart_theta, cart_kappa, cart_v, cart_a]
        frenet_traj = [frenet_s, frenet_ds, frenet_dds, frenet_l, frenet_dl, frenet_ddl]
        final_traj = PlanTrajectory(dt=0.1, cart_traj=cart_traj, frenet_traj=frenet_traj)

        return final_traj


def action_constructor(
    action_configs: dict, vehicle_params: dict
) -> Tuple[Action, Union[gym.spaces.Box, gym.spaces.MultiDiscrete, gym.spaces.Discrete]]:
    if action_configs["action_type"] == "continuous":
        action = ContinuousAction(vehicle_params, action_configs)
    elif action_configs["action_type"] == "discrete":
        if action_configs["action_base"] == "acceleration":
            action = DiscretePMAction
        elif action_configs["action_base"] == "jerk":
            action = DiscretePMJerkAction
        else:
            raise NotImplementedError(
                f"action_base {action_configs['action_base']} not supported. " f"Please choose acceleration or jerk"
            )
        action = action(vehicle_params, action_configs["long_steps"], action_configs["lat_steps"])
    elif action_configs["action_type"] == "parameters":
        action = ParameterAction(vehicle_params, action_configs)
    else:
        raise NotImplementedError(
            f"action_type {action_configs['action_type']} not supported. " f"Please choose continuous or discrete"
        )

        # Action space remove
        # TODO initialize action space with class
    if action_configs["action_type"] == "continuous":
        action_high = np.array([1.0, 1.0])
        action_space = gym.spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
    elif action_configs["action_type"] == "parameters":
        ## ----------------- continue parameters
        # action_high = np.array([1.0])
        # action_space = gym.spaces.Box(
        #     low=-action_high, high=action_high, dtype=np.float32
        # )

        ## ----------------- discrete parameters
        action_space = gym.spaces.Discrete(action_configs["lat_steps"])

    else:
        action_space = gym.spaces.MultiDiscrete([action_configs["long_steps"], action_configs["lat_steps"]])

    return action, action_space
