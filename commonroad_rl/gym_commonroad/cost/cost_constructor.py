from commonroad_rl.gym_commonroad.cost.cost import Cost, SparseCost


cost_type_to_class = {
    "sparse_cost": SparseCost,
}


def make_cost(configs: dict) -> Cost:
    """
    Initializes the cost class according to the env_configurations

    :param configs: The configuration of the environment
    :return: cost class, either sparse
    """

    cost_type = configs["cost_type"]

    if cost_type in cost_type_to_class:
        return cost_type_to_class[cost_type](configs)
    else:
        raise ValueError(f"Illegal cost type: {cost_type}!")
