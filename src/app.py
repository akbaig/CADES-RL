import random
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO,DQN,A2C
from stable_baselines3.common import env_checker
from stable_baselines3.common.evaluation import evaluate_policy

from rl_env import StatesGenerator, get_benchmark_rewards,compute_reward, critical_task_reward, CustomEnv
from config import get_config
config, _ = get_config()
import numpy as np
import time

logdir = f"../logs/{int(time.time())}/"
models_dir = f"../models/{int(time.time())}/"


if __name__ == '__main__':

    env=CustomEnv(config)

    check_env(env)
    model = PPO('MlpPolicy', env, verbose=1,tensorboard_log=logdir)
    a= env.reset()
    # TIMESTEPS = 10000
    # iters = 0
    # while True:
    #     iters += 1
    #     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    #     # model.save(f"{models_dir}/{TIMESTEPS * iters}")






