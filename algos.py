from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TRPO

# All available training algorithms
ALGOS_DICT = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'ppo1': PPO1,
    'ppo2': PPO2,
    'sac': SAC,
    'trpo': TRPO,
}


def get_algo(algo):
    return ALGOS_DICT[algo]
