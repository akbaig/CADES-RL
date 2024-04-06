import random
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from cades_env import CadesEnv
from collections import defaultdict
from utils import evaluate
import numpy as np
import time
from config import get_config
import mlflow
import sys
from typing import Any, Dict, Tuple, Union
from stable_baselines3 import SAC
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from utils import MLflowOutputFormat

# Config files
config, _ = get_config()
# Logs and model directories
logdir = f"../logs/{int(time.time())}/"
models_dir = f"../models/{int(time.time())}/"


loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)


if __name__ == "__main__":
    # Experiment Project Name for mlflow
    mlflow.set_experiment(config.experiment_name)
    # Run Name for mlflow
    mlflow.start_run(run_name="Test 2 - Bonus reward")

    # Logging config params
    config_dict = vars(config)
    for key in config_dict:
        mlflow.log_param(key, config_dict[key])

    # Initiate environment variables
    env = CadesEnv(config)
    check_env(env)
    model = RecurrentPPO.load(
        '../models/1705355208/150',
        env=env,
        device=config.device,
    )
    model.set_logger(loggers)
    env.reset()
    info = env.get_env_info()

    # defining an evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=models_dir,
        log_path=logdir,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    is_train = config.train = True
    EPOCHS = config.epochs

    if is_train:
        TIMESTEPS = 10000
        iters = 0
        while iters < EPOCHS:
            iters = iters + 1
            print("Epoch #", iters)
            model.learn(
                total_timesteps=TIMESTEPS,
                log_interval=10000,
                reset_num_timesteps=False,
                tb_log_name=f"PPO",
                callback=eval_callback,
            )
            model.save(f"{models_dir}/{iters}")


    t1 = time.time()
    (
        mean_reward,
        mean_episode_len,
        termination_cause,
        is_success,
        avg_occupancy_ratio,
    ) = evaluate(model, env, 100)
    print(f"mean_reward:{mean_reward:.2f}")
    print(f"mean_episode_len:{mean_episode_len:.2f}")
    print(termination_cause)
    print("Average Occupancy ratio:", np.array(avg_occupancy_ratio).mean())
    print("Average time per input:", (time.time() - t1) / 100)

    mlflow.log_param("avg_occupancy_ratio", np.array(avg_occupancy_ratio).mean())

    # Logging termination cause
    for key in termination_cause.keys():
        mlflow.log_metric(key, termination_cause[key])

    mlflow.log_artifacts(models_dir,"models")
    # End mlflow run session
    mlflow.end_run()
