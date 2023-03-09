import random
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common import env_checker
from stable_baselines3.common.evaluation import evaluate_policy
from states_generator import StatesGenerator
from rl_env import get_benchmark_rewards, compute_reward, critical_task_reward
from cades_env import CadesEnv
from config import get_config

config, _ = get_config()
import numpy as np
import time

logdir = f"../logs/{int(time.time())}/"
models_dir = f"../models/{int(time.time())}/"


if __name__ == "__main__":
    env = CadesEnv(config)

    check_env(env)
    model = PPO(
        "MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, device="cuda"
    )
    a = env.reset()
    action = env.action_space.sample()
    state = env.step(action)
    print(env.get_env_info())
