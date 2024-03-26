"""Class for param action reward"""

import logging
from math import exp

import numpy as np
from commonroad.common.solution import VehicleModel

from commonroad_rl.gym_commonroad.action import Action, ParameterAction
from commonroad_rl.gym_commonroad.reward.reward import Reward
from commonroad_rl.gym_commonroad.action.planner import PlanTrajectory

LOGGER = logging.getLogger(__name__)

MAX_KAPPA = 0.1
MAX_LAT_ACC = 5.0


class ParamReward(Reward):
    def __init__(self, config: dict) -> None:
        param_reward_config = config["reward_configs"]["param_reward"]
        self.comfort_config = param_reward_config["comfortable"]
        self.efficiency_config = param_reward_config["efficiency"]
        self.goal_reach_config = param_reward_config["goal_reach"]
        self.infeasible_reward = param_reward_config["infeasible_reward"]

    def reset(self, observation_dict: dict, ego_action: Action):
        pass

    @staticmethod
    def calc_jerk(trajectory: PlanTrajectory):
        a = trajectory.cart_a
        jerk = np.mean(np.abs(np.diff(a) / trajectory.dt))

        return jerk

    @staticmethod
    def calc_kappa(traj: PlanTrajectory):
        max_kappa = np.max(np.abs(traj.cart_kappa))

        return max_kappa

    @staticmethod
    def calc_lat_acc(trajectory: PlanTrajectory) -> float:
        max_lat_acc = np.max(np.abs(trajectory.frenet_ddl))

        return max_lat_acc

    @staticmethod
    def calc_reference_offset(
        max_mean_offset: float = None, trajectory: PlanTrajectory = None
    ):
        """
        Calculate the offset of the trajectory to the reference trajectory
        """
        mean_offset = np.mean(np.abs(trajectory.frenet_l))

        delta_offset = mean_offset - max_mean_offset
        if delta_offset <= 0:
            return 0.0
        else:
            return delta_offset

    @staticmethod
    def calc_yaw_rate(trajectory: PlanTrajectory):
        yaw_rate = np.mean(np.abs(np.diff(trajectory.cart_theta) / trajectory.dt))

        return yaw_rate

    def calc_reward(
        self,
        observation_dict: dict,
        ego_action: ParameterAction,
        action_false: bool = None,
    ) -> float:
        """
        Calculate the reward according to the observations

        :param observation_dict: current observations
        :param ego_action: Current ego_action of the environment
        :return: Reward of this step
        """
        final_reward = 0.0

        trajectory = ego_action.current_trajectory
        if trajectory is None or action_false:
            return self.infeasible_reward

        ## comfort reward
        # jerk
        jerk_reward = self.comfort_config["jerk_reward_scale"] * ParamReward.calc_jerk(
            trajectory
        )
        # lat_acc
        max_lat_acc = ParamReward.calc_lat_acc(trajectory)
        lat_acc_reward = (
            0.0
            if max_lat_acc < MAX_LAT_ACC
            else self.comfort_config["lat_acc_reward_scale"]
            * (max_lat_acc - MAX_LAT_ACC)
        )
        # reference offset
        max_offset = self.comfort_config["max_allowable_reference_offset"]
        reference_offset_reward = self.comfort_config[
            "reference_offset_reward_scale"
        ] * ParamReward.calc_reference_offset(max_offset, trajectory)
        # yaw rate
        yaw_rate_reward = self.comfort_config[
            "omega_reward_scale"
        ] * ParamReward.calc_yaw_rate(trajectory)
        # kappa
        max_kappa = ParamReward.calc_kappa(trajectory)
        kappa_reward = (
            0.0
            if max_kappa < MAX_KAPPA
            else self.comfort_config["kappa_reward_scale"] * (max_kappa - MAX_KAPPA)
        )
        comfort_reward = (
            jerk_reward
            + lat_acc_reward
            + reference_offset_reward
            + yaw_rate_reward
            + kappa_reward
        )

        final_reward -= comfort_reward

        # efficiency reward
        max_rel_lead_v = max(observation_dict["lane_based_v_rel"][-3:])
        max_lead_v = max_rel_lead_v + observation_dict["v_ego"]
        v_reward = (
            self.efficiency_config["velocity_reward_scale"]
            * np.clip(observation_dict["v_ego"] / (max_lead_v + 1e-3), 0, 1)[0]
        )

        time_reward = self.efficiency_config["reward_time_step"]

        efficiency_reward = v_reward + time_reward

        final_reward += efficiency_reward

        # goal reach reward
        goal_reward = 0
        if observation_dict["is_goal_reached"]:
            goal_reward += self.goal_reach_config["reward_goal_reached"]

        goal_lon_dis = observation_dict["distance_goal_long_advance"]
        goal_lat_dis = observation_dict["distance_goal_lat_advance"]

        if ego_action.vehicle.current_time_step % 5 == 0:
            goal_reward += self.goal_reach_config["forward_reward_scale"]

        goal_reward += (
            self.goal_reach_config["goal_long_advance_reward_scale"] * goal_lon_dis
            + self.goal_reach_config["goal_lat_advance_reward_scale"] * goal_lat_dis
        )[0]

        final_reward += goal_reward

        return final_reward
