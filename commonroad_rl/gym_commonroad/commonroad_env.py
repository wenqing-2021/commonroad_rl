"""
Module for the CommonRoad Gym environment
"""

import os
import pathlib

import gymnasium as gym
import yaml
import pickle
import random
import logging
import numpy as np
from queue import Queue
import seaborn as sns
from typing import Tuple, Union, Dict

# import from commonroad-drivability-checker
from commonroad.geometry.shape import Rectangle

# import from commonroad-reach
from commonroad_reach.utility.visualization import (
    generate_default_drawing_parameters,
    compute_plot_limits_from_reachable_sets,
    draw_reachable_sets,
    draw_drivable_area,
)
from commonroad_reach.data_structure.reach.reach_interface import ReachableSetInterface

# import from commonroad-io
from commonroad.scenario.scenario import ScenarioID, Scenario
from commonroad.scenario.trajectory import (
    State,
    InitialState,
    Trajectory,
    STState,
)
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import (
    MPDrawParams,
    DynamicObstacleParams,
    TrajectoryParams,
    ShapeParams,
)
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType

# import from commonroad-rl
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
from commonroad_rl.gym_commonroad.observation import ObservationCollector
from commonroad_rl.gym_commonroad.utils.scenario_io import restore_scenario
from commonroad_rl.gym_commonroad.utils.scenario import parse_map_name
from commonroad_rl.gym_commonroad.action import action_constructor, ParameterAction
from commonroad_rl.gym_commonroad.reward import reward_constructor
from commonroad_rl.gym_commonroad.cost import cost_constructor
from commonroad_rl.gym_commonroad.reward.reward import Reward
from commonroad_rl.gym_commonroad.cost.cost import Cost
from commonroad_rl.gym_commonroad.reward.termination import Termination
from commonroad_rl.gym_commonroad.utils.collision_type_checker import (
    check_collision_type,
)

LOGGER = logging.getLogger(__name__)


class CommonroadEnv(gym.Env):
    """
    Description:
        This environment simulates the ego vehicle in a traffic scenario using commonroad environment. The task of
        the ego vehicle is to reach the predefined goal without going off-road, collision with other vehicles, and
        finish the task in specific time frame. Please consult `commonroad_rl/gym_commonroad/README.md` for details.
    """

    metadata = {"render_modes": ["human"]}

    # For the current configuration check the ./configs.yaml file
    def __init__(
        self,
        meta_scenario_path=PATH_PARAMS["meta_scenario"],
        train_reset_config_path=PATH_PARAMS["train_reset_config"],
        test_reset_config_path=PATH_PARAMS["test_reset_config"],
        visualization_path=PATH_PARAMS["visualization"],
        logging_path=None,
        test_env=False,
        play=False,
        config_file=PATH_PARAMS["configs"]["commonroad-v1"],
        logging_mode=1,
        **kwargs,
    ) -> None:
        """
        Initialize environment, set scenario and planning problem.
        """
        # Set logger if not yet exists
        LOGGER.setLevel(logging_mode)

        if not len(LOGGER.handlers):
            formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging_mode)
            stream_handler.setFormatter(formatter)
            LOGGER.addHandler(stream_handler)

            if logging_path is not None:
                file_handler = logging.FileHandler(filename=os.path.join(logging_path, "console_copy.txt"))
                file_handler.setLevel(logging_mode)
                file_handler.setFormatter(formatter)
                LOGGER.addHandler(file_handler)

        LOGGER.debug("Initialization started")

        # Default configuration
        if isinstance(config_file, (str, pathlib.Path)):
            with pathlib.Path(config_file).open("r") as config_file:
                config = yaml.safe_load(config_file)

        # Assume default environment configurations
        self.configs = config.get("env_configs", config)

        # Overwrite environment configurations if specified
        if kwargs is not None:
            for k, v in kwargs.items():
                assert k in self.configs, f"Configuration item not supported: {k}"
                # TODO: update only one term in configs
                if isinstance(v, dict):
                    self.configs[k].update(v)
                else:
                    self.configs.update({k: v})

        # Make environment configurations as attributes
        self.vehicle_params: dict = self.configs["vehicle_params"]
        self.action_configs: dict = self.configs["action_configs"]
        self.render_configs: dict = self.configs["render_configs"]
        self.reward_type: str = self.configs["reward_type"]

        # change configurations when using point mass vehicle model
        if self.vehicle_params["vehicle_model"] == 0:
            self.observe_heading = False
            self.observe_steering_angle = False
            self.observe_global_turn_rate = False
            self.observe_distance_goal_long_lane = False

        # Flag for popping out scenarios
        self.play = play

        # Flags for collision type checker evaluation
        self.check_collision_type = self.configs["check_collision_type"]
        self.lane_change_time_threshold = self.configs["lane_change_time_threshold"]

        # Load scenarios and problems
        self.meta_scenario_path = meta_scenario_path
        self.all_problem_dict = dict()
        self.planning_problem_set_dict = dict()
        self._planning_problems_queue = Queue()
        self.benchmark_id = None
        # Accelerator structures
        # self.cache_goal_obs = dict()

        if isinstance(meta_scenario_path, (str, pathlib.Path)):
            meta_scenario_reset_dict_path = pathlib.Path(self.meta_scenario_path) / "meta_scenario_reset_dict.pickle"
            with meta_scenario_reset_dict_path.open("rb") as f:
                self.meta_scenario_reset_dict = pickle.load(f)
        else:
            self.meta_scenario_reset_dict = meta_scenario_path

        self.train_reset_config_path = train_reset_config_path

        def load_reset_config(path):
            path = pathlib.Path(path)
            problem_dict = {}
            for p in sorted(path.glob("*.pickle")):
                with p.open("rb") as f:
                    problem_dict[p.stem] = pickle.load(f)
            return problem_dict

        if not test_env and not play:
            if isinstance(train_reset_config_path, (str, pathlib.Path)):
                self.all_problem_dict = load_reset_config(train_reset_config_path)
            else:
                self.all_problem_dict = train_reset_config_path
            self.is_test_env = False
            LOGGER.info(f"Training on {train_reset_config_path} with {len(self.all_problem_dict.keys())} scenarios")
        else:
            if isinstance(test_reset_config_path, (str, pathlib.Path)):
                self.all_problem_dict = load_reset_config(test_reset_config_path)
            else:
                self.all_problem_dict = test_reset_config_path
            LOGGER.info(f"Testing on {test_reset_config_path} with {len(self.all_problem_dict.keys())} scenarios")

        self.visualization_path = visualization_path

        self.termination = Termination(self.configs)
        self.terminated = False
        self.termination_reason = None

        action_constrcut_results = action_constructor(self.action_configs, self.vehicle_params)

        self.ego_action: ParameterAction = action_constrcut_results[0]
        self.action_space = action_constrcut_results[1]
        self.ilqr_trajectory = None

        # Observation space
        self.observation_collector = ObservationCollector(self.configs)
        self._enlarge_goal = self.configs["goal_configs"]["enlarge_goal"]
        # Reward function
        self.reward_function: Reward = reward_constructor.make_reward(self.configs)
        # Cost function
        self.cost_function: Cost = cost_constructor.make_cost(self.configs)

        # TODO initialize reward class

        LOGGER.debug(f"Meta scenario path: {meta_scenario_path}")
        LOGGER.debug(f"Training data path: {train_reset_config_path}")
        LOGGER.debug(f"Testing data path: {test_reset_config_path}")
        LOGGER.debug("Initialization done")

        # ----------- Visualization -----------
        self.cr_render = None
        self.draw_params = None

    @property
    def observation_space(self):
        return self.observation_collector.observation_space

    @property
    def observation_dict(self):
        return self.observation_collector.observation_dict

    def seed(self, seed=Union[None, int]):
        self.action_space.seed(seed)

    def reset_planning_problem(
        self,
        benchmark_id=None,
        planning_problem_id=None,
        scenario=None,
        planning_problem=None,
    ):
        # initial
        if self.benchmark_id is None:
            self._set_scenario_problem(
                benchmark_id=benchmark_id,
                planning_problem_id=planning_problem_id,
                scenario=scenario,
                planning_problem=planning_problem,
            )
        else:
            if benchmark_id is not None and benchmark_id != self.benchmark_id:
                self._set_scenario_problem(
                    benchmark_id=benchmark_id,
                    planning_problem_id=planning_problem_id,
                    scenario=scenario,
                    planning_problem=planning_problem,
                )
            else:
                if not self._set_planning_problem(planning_problem_id=planning_problem_id):
                    # this case mean: the problem in this scenario has been explored, and we need to change the scenario
                    self._set_scenario_problem(
                        benchmark_id,
                        planning_problem_id=planning_problem_id,
                        scenario=scenario,
                        planning_problem=planning_problem,
                    )

    def reset(
        self,
        seed=1,
        options: dict = None,
    ) -> np.ndarray:
        """
        Reset the environment.
        :param benchmark_id: benchmark id used for reset to specific scenario
        :param reset_renderer: parameter used for reset the renderer to default

        :return: observation
        """
        super().reset(seed=seed)
        benchmark_id = None
        planning_problem_id = None
        scenario: Scenario = None
        planning_problem: PlanningProblem = None
        if options is not None:
            if "benchmark_id" in options:
                benchmark_id = options["benchmark_id"]
            if "planning_problem_id" in options:
                planning_problem_id = options["planning_problem_id"]
            if "scenario" in options:
                scenario = options["scenario"]
            if "planning_problem" in options:
                planning_problem = options["planning_problem"]
        while True:
            self.reset_planning_problem(
                benchmark_id=benchmark_id,
                planning_problem_id=planning_problem_id,
                scenario=scenario,
                planning_problem=planning_problem,
            )
            try:
                LOGGER.debug(f"benchmark id is {self.benchmark_id}")
                LOGGER.debug(f"planning problem id is {self.planning_problem.planning_problem_id}")
                self.reset_config.update({"enlarge_goal": self._enlarge_goal})
                self.observation_collector.reset(
                    self.scenario,
                    self.planning_problem,
                    self.reset_config,
                    self.benchmark_id,
                    clone_collision_checker=scenario is None or planning_problem is None,
                )
                self.reset_renderer()
                # TODO: remove self._set_goal()
                self._set_initial_goal_reward()

                self.terminated = False
                # initial vehicle
                self.ego_action.vehicle.reset(
                    self.planning_problem.initial_state,
                    self.ego_action.controller.control_dt,
                )
                initial_observation = self.observation_collector.observe(self.ego_action.vehicle)
                break
            except:
                continue
        self.reward_function.reset(self.observation_dict, self.ego_action)
        self.cost_function.reset(self.observation_dict, self.ego_action)
        self.termination.reset(self.observation_dict, self.ego_action)

        self.v_ego_mean = self.ego_action.vehicle.state.velocity
        # TODO: tmp store all observations in info for paper draft, remove afterwards
        self.observation_list = [self.observation_dict]
        self.ego_action.reset(
            self.scenario.dt,
            local_ccosy=self.observation_collector.local_ccosy,
        )
        self.ilqr_trajectory = None
        info = dict()

        return initial_observation, info

    @property
    def current_step(self):
        return self.observation_collector.time_step

    @current_step.setter
    def current_step(self, time_step):
        raise ValueError(f"<CommonroadEnv> Set current_step is prohibited!")

    def step(self, action: Union[np.ndarray, State, Dict]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Propagate to next time step, compute next observations, reward and status.

        :param action: vehicle acceleration, vehicle steering velocity
        :return: observation, reward, status and other information

        if you use reach_interface, makesure the input is dict{"action":action, "reach_interface":reach_interface}.
        """
        # check the action has risk result
        reach_interface = None
        only_plan = False
        if isinstance(action, Dict):
            if "risk_info" in action.keys():
                risk_info = action["risk_info"]
            if "reach_interface" in action.keys():
                reach_interface = action["reach_interface"]
            if "only_plan" in action.keys():
                only_plan = action["only_plan"]

            # get the action
            action = action["action"]

        if isinstance(action, State):
            # set ego_state directly
            ego_state = action
            self.ego_action.vehicle.set_current_state(ego_state)
        else:
            if self.action_configs["action_type"] == "continuous":
                action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
            elif self.action_configs["action_type"] == "parameters":
                action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
                if not self.action_configs["refine_trajectory"] or reach_interface is None:
                    action_false = self.ego_action.step(
                        action,
                        logger=LOGGER,
                    )
                else:
                    if only_plan:
                        refine_trajectory, nodes_vertices = self.ego_action.step(
                            action,
                            logger=LOGGER,
                            reach_interface=reach_interface,
                            only_plan=only_plan,
                        )
                        return (
                            np.array([0.0]),
                            0.0,
                            False,
                            False,
                            {
                                "cost": 0.0,
                                "refine_trajectory": refine_trajectory,
                                "nodes_vertices": nodes_vertices,
                            },
                        )
                    action_false = self.ego_action.step(action, logger=LOGGER, reach_interface=reach_interface)
            else:
                self.ego_action.step(action, local_ccosy=self.observation_collector.local_ccosy)
        LOGGER.debug(
            f"current vehicle position is {self.ego_action.vehicle.state.position}, vel is {self.ego_action.vehicle.state.velocity}"
        )
        try:
            observation = self.observation_collector.observe(self.ego_action.vehicle)
        except:
            return np.array([0.0]), 0.0, True, False, {"cost": 0.0}

        # Check for termination
        done, reason, termination_info = self.termination.is_terminated(self.observation_dict, self.ego_action)
        if reason is not None:
            self.termination_reason = reason

        if action_false:
            self.termination_reason = "invalid_action"
            done = True

        if done:
            self.terminated = True

        # Calculate reward
        reward = self.reward_function.calc_reward(self.observation_dict, self.ego_action, action_false)

        # Calculate cost
        cost = self.cost_function.calc_cost(self.observation_dict, self.ego_action)

        self.v_ego_mean += self.ego_action.vehicle.state.velocity
        self.observation_list.append(self.observation_list)
        # assert str(self.scenario.scenario_id) == self.benchmark_id
        info = {
            "scenario_name": self.benchmark_id,
            "chosen_action": action,
            "current_episode_time_step": self.current_step,
            "max_episode_time_steps": self.observation_collector.episode_length,
            "termination_reason": self.termination_reason,
            "v_ego_mean": self.v_ego_mean / (self.current_step + 1),
            "observation_list": self.observation_list,
            "cost": cost,
        }
        info.update(termination_info)

        if (
            self.configs["surrounding_configs"]["observe_lane_circ_surrounding"]
            or self.configs["surrounding_configs"]["observe_lane_rect_surrounding"]
        ):
            info["ttc_follow"], info["ttc_lead"] = CommonroadEnv.get_ttc_lead_follow(self.observation_dict)

        if info["is_collision"] and self.check_collision_type:
            # TODO: what is updated here?
            check_collision_type(
                info,
                LOGGER,
                self.ego_action.vehicle,
                self.observation_collector,
                self.scenario,
                self.benchmark_id,
                self.lane_change_time_threshold,
                self.observation_collector.local_ccosy,
            )
            info["termination_reason"] = "invalid_collision"

        return observation, reward, done, False, info

    def reset_renderer(
        self,
        renderer: Union[MPRenderer, None] = None,
        draw_params: Union[MPDrawParams, dict, None] = None,
    ) -> None:
        if renderer:
            self.cr_render = renderer
        else:
            self.cr_render = MPRenderer()

        if draw_params:
            self.draw_params = draw_params
        else:
            self.draw_params = MPDrawParams()
            self.draw_params.time_begin = self.current_step
            # TODO: test if needed for time step > 200
            self.draw_params.time_end = self.current_step
            self.draw_params.lanelet_network.lanelet.show_label = False
            self.draw_params.lanelet_network.lanelet.fill_lanelet = True
            self.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
            self.draw_params.lanelet_network.traffic_sign.show_traffic_signs = "all"
            self.draw_params.lanelet_network.traffic_sign.show_label = False
            self.draw_params.lanelet_network.traffic_sign.scale_factor = 0.1
            self.draw_params.lanelet_network.intersection.draw_intersections = False
            self.draw_params.dynamic_obstacle.show_label = False

    def render(self, mode: str = "human", **kwargs) -> None:
        """
        Generate images for visualization.

        :param mode: default as human for visualization
        :return: None
        """
        # Render only every xth timestep, the first and the last
        if not (self.current_step % self.render_configs["render_skip_timesteps"] == 0 or self.terminated):
            return

        # update timestep in draw_params
        if self.draw_params is None:
            print("draw params is None, cannot render")
            return
        if "vec_env_show" in kwargs.keys():
            if kwargs["vec_env_show"]:
                self.cr_render.clear(True)
        if isinstance(self.draw_params, dict):
            self.draw_params.update(
                {
                    "scenario": {
                        "time_begin": self.current_step,
                        "time_end": self.current_step,
                    }
                }
            )
        else:
            self.draw_params.time_begin = self.current_step
            self.draw_params.time_end = self.current_step
        # Draw scenario, goal, sensing range and detected obstacles
        self.scenario.draw(self.cr_render, self.draw_params)

        # Draw certain objects only once
        if (not self.render_configs["render_combine_frames"] or self.current_step == 0) and not isinstance(mode, int):
            self.planning_problem.draw(self.cr_render)

        self.observation_collector.render(self.render_configs, self.cr_render)

        # Draw ego vehicle # draw icon
        draw_ego_initial_state = InitialState(
            position=self.ego_action.vehicle.state.position,
            velocity=self.ego_action.vehicle.state.velocity,
            orientation=self.ego_action.vehicle.state.orientation,
            acceleration=self.ego_action.vehicle.current_action[1],
            yaw_rate=self.ego_action.vehicle.state.yaw_rate,
            time_step=self.ego_action.vehicle.state.time_step,
        )
        ego_obstacle = DynamicObstacle(
            obstacle_id=self.scenario.generate_object_id(),
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=Rectangle(
                length=self.ego_action.vehicle.parameters.l,
                width=self.ego_action.vehicle.parameters.w,
            ),
            initial_state=draw_ego_initial_state,
        )

        ego_draw_params = DynamicObstacleParams()
        ego_draw_params.time_begin = self.current_step
        # TODO: check if needed for time step > 200
        ego_draw_params.time_end = self.current_step
        ego_draw_params.draw_icon = True
        ego_draw_params.vehicle_shape.occupancy.shape.facecolor = "red"
        ego_draw_params.vehicle_shape.occupancy.shape.edgecolor = "red"
        ego_draw_params.vehicle_shape.occupancy.shape.zorder = 20
        ego_obstacle.draw(self.cr_render, draw_params=ego_draw_params)

        # show trajectory
        if self.ego_action.planner.trajectory is not None:
            (
                viz_trajecotry,
                viz_traj_params,
            ) = self.ego_action.planner.trajectory.convert_to_viz_trajectory()
            self.cr_render.draw_trajectory(viz_trajecotry, viz_traj_params)

        # show refine trajectory if exists
        if self.ego_action.current_refine_trajectory is not None:
            traj_color = (238 / 255, 191 / 255, 109 / 255)
            (
                viz_refine_trajecotry,
                viz_traj_params,
            ) = self.ego_action.current_refine_trajectory.convert_to_viz_trajectory(traj_color=traj_color)
            self.cr_render.draw_trajectory(viz_refine_trajecotry, viz_traj_params)

        # show reference
        # print(self.ego_action.polynomial_planner._co.reference)
        reference_pts_list = self.ego_action.planner._co.reference
        reference_traj_list = []
        for index, pts in enumerate(reference_pts_list):
            reference_traj_list.append(STState(time_step=index, position=np.array(pts)))
        reference_trajectory = Trajectory(initial_time_step=0, state_list=reference_traj_list)
        reference_viz_params = TrajectoryParams(
            time_begin=0,
            time_end=len(reference_pts_list),
            draw_continuous=True,
            facecolor="g",
        )
        self.cr_render.draw_trajectory(reference_trajectory, reference_viz_params)

        # show tracked state
        # if self.ego_action.matched_state is not None:
        #     matched_state = self.ego_action.matched_state
        #     matched_position = matched_state.position
        #     self.cr_render.draw_ellipse(
        #         center=[matched_position[0], matched_position[1]],
        #         radius_x=0.5,
        #         radius_y=0.5,
        #         draw_params=self.draw_params.shape,
        #     )

        # self.ego_action.vehicle.collision_object.draw(self.cr_render, draw_params={"facecolor": "green", "zorder": 30})

        # Save figure, only if frames should not be combined or simulation is over
        os.makedirs(
            os.path.join(self.visualization_path, str(self.scenario.scenario_id)),
            exist_ok=True,
        )
        if not self.render_configs["render_combine_frames"] or self.terminated:
            if isinstance(mode, int):
                filename = (
                    os.path.join(
                        self.visualization_path,
                        str(self.scenario.scenario_id),
                        self.file_name_format % mode,
                    )
                    + ".png"
                )
            else:
                filename = (
                    os.path.join(
                        self.visualization_path,
                        str(self.scenario.scenario_id),
                        self.file_name_format % self.current_step,
                    )
                    + ".png"
                )
            if self.render_configs["render_follow_ego"]:
                # TODO: works only for highD, implement range relative to ego orientation
                # follow ego
                x, y = self.ego_action.vehicle.state.position
                range = self.render_configs["render_range"]
                self.cr_render.plot_limits = [
                    x - range[0],
                    x + range[0],
                    y - range[1],
                    y + range[1],
                ]

            if "vec_env_show" in kwargs.keys():
                if kwargs["vec_env_show"]:
                    result = {
                        "render": self.cr_render,
                        "filename": filename,
                        "keep_static_artists": True,
                        "time_step": self.current_step,
                    }
                    return result
                else:
                    return None
            else:
                self.cr_render.render(show=True, filename=filename, keep_static_artists=True)

        # =================================================================================================================
        #
        #                                    reset functions
        #
        # =================================================================================================================

    def _reset_used_planning_problem(self, problem_dict: dict = None):
        while not self._planning_problems_queue.empty():
            self._planning_problems_queue.get()
        for pb_item in problem_dict["planning_problem_set"].planning_problem_dict.items():
            key_i = pb_item[0]
            self._planning_problems_queue.put(key_i)
        self.planning_problem_set_dict = problem_dict["planning_problem_set"].planning_problem_dict

    def _set_planning_problem(self, planning_problem_id=None) -> bool:
        """
        Select a planning problem from the defined planning problem set.
        return: if all planning problem has used, return False, else True.
        """
        if not self._planning_problems_queue.empty():
            # find the target id
            if planning_problem_id is not None:
                find_id = False
                temp_queue = Queue()
                while not self._planning_problems_queue.empty():
                    queue_id = self._planning_problems_queue.get()
                    temp_queue.put(queue_id)
                    if queue_id == planning_problem_id:
                        find_id = True
                while not temp_queue.empty():
                    self._planning_problems_queue.put(temp_queue.get())

                if find_id:
                    self.planning_problem: PlanningProblem = self.planning_problem_set_dict[planning_problem_id]
                    return True
                else:
                    return False
            else:
                pb_key = self._planning_problems_queue.get()
                self.planning_problem: PlanningProblem = self.planning_problem_set_dict[pb_key]
                return True
        else:
            return False

    def _set_scenario_problem(
        self,
        benchmark_id=None,
        planning_problem_id=None,
        scenario: Scenario = None,
        planning_problem: PlanningProblem = None,
    ) -> None:
        """
        Select scenario and new a planning problem set. Only USED after the planning problems all used.

        :return: None
        """
        if scenario is None:
            if benchmark_id is not None:
                self.benchmark_id = benchmark_id
                problem_dict = self.all_problem_dict[benchmark_id]
            else:
                if self.play:
                    # pop instead of reusing
                    LOGGER.debug(f"Number of scenarios left {len(list(self.all_problem_dict.keys()))}")
                    self.benchmark_id = random.choice(list(self.all_problem_dict.keys()))
                    problem_dict = self.all_problem_dict.pop(self.benchmark_id)
                else:
                    self.benchmark_id, problem_dict = random.choice(list(self.all_problem_dict.items()))
            self._reset_used_planning_problem(problem_dict=problem_dict)
            # Set reset config dictionary
            scenario_id = ScenarioID.from_benchmark_id(self.benchmark_id, "2020a")
            assert (
                str(scenario_id)[-3:] == "T-1"
            ), f"Scenarios with SetBasedPrediction {str(scenario_id)} are not supported!"
            map_id = parse_map_name(scenario_id)
            self.reset_config = self.meta_scenario_reset_dict[map_id]
            # meta_scenario = self.problem_meta_scenario_dict[self.benchmark_id]
            self.scenario = restore_scenario(
                self.reset_config["meta_scenario"],
                problem_dict["obstacle"],
                scenario_id,
            )
            if planning_problem is None:
                self._set_planning_problem(planning_problem_id=planning_problem_id)
            else:
                self.planning_problem: PlanningProblem = planning_problem
        else:
            # NOTE: NOT USE in the code running
            # TODO: calculate reset_config online
            from commonroad_rl.tools.pickle_scenario.preprocessing import (
                generate_reset_config,
            )

            self.reset_config = generate_reset_config(scenario, open_lane_ends=True)
            self.scenario = scenario
            self.planning_problem = planning_problem
            self.benchmark_id = str(scenario.scenario_id)

        # Set name format for visualization
        self.file_name_format = self.benchmark_id + "_ts_%03d"

    def _set_initial_goal_reward(self) -> None:
        """
        Set ego vehicle and initialize its status.

        :return: None
        """
        self.goal = self.observation_collector.goal_observation

    @staticmethod
    def get_ttc_lead_follow(observation_dict):
        idx_follow = 1
        idx_lead = 4

        def get_ttc(p_rel, v_rel):
            if np.isclose(v_rel, 0.0):
                return np.inf
            else:
                return p_rel / -v_rel

        # lane_based_v_rel = v_lead - v_follow
        # ttc: (s_lead - s_follow) / (v_follow - v_lead)
        ttc_follow = get_ttc(
            observation_dict["lane_based_p_rel"][idx_follow],
            observation_dict["lane_based_v_rel"][idx_follow],
        )
        ttc_lead = get_ttc(
            observation_dict["lane_based_p_rel"][idx_lead],
            observation_dict["lane_based_v_rel"][idx_lead],
        )

        return ttc_follow, ttc_lead

    @staticmethod
    def step_plan_trajectory(ego_action, navigator, action):
        """
        NOT used in the code running
        """
        trajectory = ego_action.step_plan_traj(action=action, logger=LOGGER)
        if trajectory is None:
            if len(ego_action.trajectory_history) > 0:
                trajectory = ego_action.trajectory_history[-1]
            else:
                trajectory = ego_action.planner.ReferenceTraj(navigator=navigator)
                # get the closet point on the reference trajectory
                position = np.array([trajectory.cart_x, trajectory.cart_y]).transpose()
                current_position = ego_action.vehicle.state.position
                current_v = ego_action.vehicle.state.velocity
                distance = np.linalg.norm(current_position - position, axis=1)
                min_index = np.argmin(distance)
                end_index = min(min_index + 30, len(trajectory.cart_x))
                trajectory = np.array(
                    [
                        trajectory.cart_x[min_index:end_index],
                        trajectory.cart_y[min_index:end_index],
                        trajectory.cart_theta[min_index:end_index],
                        np.ones(end_index - min_index) * current_v,
                    ]
                ).transpose()
                return trajectory

        trajectory = np.array(
            [
                trajectory.cart_x,
                trajectory.cart_y,
                trajectory.cart_theta,
                trajectory.cart_v,
            ]
        ).transpose()

        return trajectory

    @staticmethod
    def render_vec_env(
        env: gym.vector.VectorEnv = None,
        risk_result_list=None,
        planner_result_list=None,
        reachable_set_interface_list=None,
    ):
        env_render_list = env.call("render", vec_env_show=True)
        if risk_result_list is None:
            risk_result_list = [None] * len(env_render_list)
        if planner_result_list is None:
            planner_result_list = [None] * len(env_render_list)
        if reachable_set_interface_list is None:
            reachable_set_interface_list = [None] * len(env_render_list)
        for render_dict, risk_result, planner_result, reachable_set_interface in zip(
            env_render_list,
            risk_result_list,
            planner_result_list,
            reachable_set_interface_list,
        ):
            if render_dict is not None:
                render = render_dict["render"]
                time_step = render_dict["time_step"]
                # show risk result
                if risk_result is not None:
                    CommonroadEnv.render_risk_result(risk_result, render)
                # show reachable set
                if reachable_set_interface is not None:
                    CommonroadEnv.render_reachable_set(reachable_set_interface, time_step, render)
                # show planner result
                if planner_result is not None:
                    CommonroadEnv.render_planner_result(planner_result, render)
                render.render(
                    show=True,
                    filename=render_dict["filename"],
                    keep_static_artists=render_dict["keep_static_artists"],
                )
            else:
                raise ValueError("render_dict is None, check the render function in ComonroadEnv")

    @staticmethod
    def render_risk_result(risk_result=None, render=None):
        # get vehicle risk field result
        vehicle_risk_field_list = risk_result.vehicle_risk_field_list
        for vehicle_risk_field in vehicle_risk_field_list:
            # get the original trajectory of the surrounding vehicles
            original_traj = vehicle_risk_field.original_traj
            # original_traj = vehicle_risk_field.mean_states[:, :2]
            # construct trajectory for visualization
            traj_state_list = []
            for index, pts in enumerate(original_traj):
                traj_state_list.append(STState(time_step=index, position=pts))
            trajectory = Trajectory(initial_time_step=0, state_list=traj_state_list)
            trajectory_viz_params = TrajectoryParams(
                time_begin=0,
                time_end=int(len(original_traj)),
                draw_continuous=True,
                line_width=1.2,
                facecolor="blue",
            )
            # render.draw_trajectory(trajectory, trajectory_viz_params)
            cvar_x = vehicle_risk_field.gpr_result.CVaR_x
            cvar_y = vehicle_risk_field.gpr_result.CVaR_y
            left_bd_traj_state_list = []
            right_bd_traj_state_list = []
            for index in range(len(cvar_x)):
                left_bd_traj_state_list.append(STState(time_step=index, position=[cvar_x[index][0], cvar_y[index][0]]))
                right_bd_traj_state_list.append(STState(time_step=index, position=[cvar_x[index][1], cvar_y[index][1]]))
            left_bd_trajectory = Trajectory(initial_time_step=0, state_list=left_bd_traj_state_list)
            right_bd_trajectory = Trajectory(initial_time_step=0, state_list=right_bd_traj_state_list)
            bd_trajectory_viz_params = TrajectoryParams(
                time_begin=0,
                time_end=int(len(cvar_x) - 13),
                draw_continuous=True,
                line_width=1.2,
                facecolor="red",
            )
            # render.draw_trajectory(left_bd_trajectory, bd_trajectory_viz_params)
            # render.draw_trajectory(right_bd_trajectory, bd_trajectory_viz_params)
            particals = vehicle_risk_field.particals
            risk_p = particals[2, :, :]
            risk_x = particals[0, :, :]
            risk_y = particals[1, :, :]
            row, col = risk_p.shape
            for i in range(row):
                for j in range(col):
                    center_x = risk_x[i, j]
                    center_y = risk_y[i, j]
                    risk = risk_p[i, j]
                    circle_viz_params = ShapeParams(opacity=risk, facecolor="gold", edgecolor="gold")
                    render.draw_ellipse(
                        center=[center_x, center_y],
                        radius_x=0.25,
                        radius_y=0.25,
                        draw_params=circle_viz_params,
                    )

    @staticmethod
    def render_reachable_set(reach_interface: ReachableSetInterface, step, renderer):
        config = reach_interface.config
        plot_limits = compute_plot_limits_from_reachable_sets(reach_interface)
        palette = sns.color_palette("GnBu_d", 3)
        edge_color = (palette[0][0] * 0.75, palette[0][1] * 0.75, palette[0][2] * 0.75)
        last_step = reach_interface.step_end
        # generate default drawing parameters
        draw_params = generate_default_drawing_parameters(config)
        draw_params.shape.facecolor = palette[0]
        draw_params.shape.edgecolor = edge_color

        # draw reachable set
        for step_i in range(step, last_step):
            list_nodes = reach_interface.reachable_set_at_step(step_i)
            draw_reachable_sets(list_nodes, config, renderer, draw_params)

    @staticmethod
    def render_planner_result(planner_result, render):
        def draw_trajectory(trajectory, render, **kwargs):
            # construct trajectory for visualization
            traj_state_list = []
            for index, pts in enumerate(trajectory):
                traj_state_list.append(STState(time_step=index, position=pts[:2]))
            trajectory_viz = Trajectory(initial_time_step=0, state_list=traj_state_list)
            face_color = kwargs.get("facecolor", "green")
            line_width = kwargs.get("line_width", 1.5)
            viz_param = TrajectoryParams(
                time_begin=0,
                time_end=len(trajectory),
                draw_continuous=True,
                line_width=line_width,
                facecolor=face_color,
            )
            render.draw_trajectory(trajectory_viz, viz_param)

        warm_start_trajectory = planner_result.warm_start_trajectory
        draw_trajectory(warm_start_trajectory, render, facecolor="green")
        draw_trajectory(planner_result.plan_trajectory, render, facecolor="black")
