from abc import ABC, abstractmethod

from commonroad_rl.gym_commonroad.action import Action


class Cost(ABC):
    """Abstract class for rewards"""

    def reset(self, observation_dict: dict, ego_action: Action):
        pass

    @abstractmethod
    def calc_cost(self, observation_dict: dict, ego_action: Action) -> float:
        """
        Calculate the cost according to the observations

        :param observation_dict: current observations
        :param ego_action: Current ego_action of the environment
        :return: Cost of this step
        """


class SparseCost(Cost):
    def __init__(self, config: dict = None) -> None:
        self.cost_config = config["cost_configs"]["sparse_cost"]
        self._use_cost = config["cost_configs"]["use_cost"]

    def reset(self, observation_dict: dict, ego_action: Action):
        pass

    def calc_cost(self, observation_dict: dict, ego_action: Action) -> float:
        """
        Calculate the cost according to the observations

        :param observation_dict: current observations
        :param ego_action: Current ego_action of the environment
        :return: Cost of this step
        """
        cost = 0.0
        if not self._use_cost:
            return cost

        if observation_dict["is_collision"][0]:
            cost += self.cost_config["cost_collision"]
        if observation_dict["is_off_road"][0]:
            cost += self.cost_config["cost_off_road"]
        if observation_dict["is_time_out"][0]:
            cost += self.cost_config["cost_time_out"]

        # compute same lane lead obstacle ttc
        rel_p = observation_dict["lane_based_p_rel"][-2]  # absolute value
        rel_v = observation_dict["lane_based_v_rel"][-2]  # obstacle_v - ego_v
        ttc = rel_p / (-rel_v + 1e-5)
        if ttc < self.cost_config["tts_threshold"]:
            cost += self.cost_config["cost_ttc"]

        return cost
