import random
import numpy as np
import torch
from utils.config import get_config
from env.cades_env import CadesEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

def initialize_environment():
    # Load configuration
    config = get_config()

    # Initialize and check the environment
    env = CadesEnv(config)
    check_env(env)

    return env, config

    