from utils.config import get_config
# from env.cades_env import CadesEnv
from env.cades_env_oneshot import CadesOneShotEnv
from stable_baselines3.common.env_checker import check_env

def initialize_environment():
    # Load configuration
    config = get_config()

    # Initialize and check the environment
    env = CadesOneShotEnv(config)
    check_env(env)

    return env, config

    