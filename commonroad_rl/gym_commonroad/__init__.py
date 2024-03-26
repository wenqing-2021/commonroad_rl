"""
CommonRoad Gym environment
"""

import gymnasium as gym
from gymnasium.envs.registration import register

# Notice: this code is run everytime the gym_commonroad module is imported
# this might be pretty shady but seems to be common practice so let's at least catch the errors occurring here
try:
    register(
        id="commonroad-v1",
        entry_point="commonroad_rl.gym_commonroad.commonroad_env:CommonroadEnv",
        kwargs={
            "meta_scenario_path": "data/pickles/merging_secenario/meta_scenario",
            "train_reset_config_path": "data/pickles/merging_secenario/success_navigator_problem",
            "test_reset_config_path": "data/pickles/merging_secenario/success_navigator_problem",
            "visualization_path": "output/gym_commonroad/viz",
            "config_file": "commonroad_rl/gym_commonroad/configs.yaml",
            "logging_mode": 4,
            "use_safe_rl": False,
            "use_reach_set": False,
            "test_env": False,
            "render_mode": "human",
        },
    )
except gym.error.Error:
    print("[gym_commonroad/__init__.py] Error occurs while registering commonroad-v1")
    pass
