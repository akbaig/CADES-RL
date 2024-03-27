from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import RecurrentPPO

import random
import shutil
import numpy as np
from cades_env import CadesEnv
from utils import evaluate
import numpy as np
import time
from config import get_config
import torch
import mlflow
import sys
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from utils import MLflowOutputFormat

# Config files
config, _ = get_config()

# Logs and model directories
logs_path = "../logs"
models_path = "../models"
logs_dir = f"{logs_path}/{config.experiment_name}/{config.run_name}/"
models_dir = f"{models_path}/{config.experiment_name}/{config.run_name}/"

loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)

if __name__ == "__main__":

    # Experiment Project Name for mlflow
    mlflow.set_experiment(config.experiment_name)
    # Run Name for mlflow
    mlflow.start_run(run_name="Test: 12 Tasks variable weights - normal RL")

    # Log config params
    config_dict = vars(config)
    for key in config_dict:
        mlflow.log_param(key, config_dict[key])
    
    # Set random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Initiate environment variables
    env = CadesEnv(config)
    # Check environment compilance with stable baselines
    check_env(env)
    # Initiate model
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=1,
        learning_rate=config.lr,
        tensorboard_log=logs_dir,
        batch_size=config.batch_size,
        device=config.device,
        seed=config.seed,
    )
    # Set logger
    model.set_logger(loggers)
    # Initialize Environment (By Resetting it)
    env.reset()

    # Get environment info
    # info = env.get_env_info()

    # Define an evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    # Train the model
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

    # Evaluate the model
    t1 = time.time()
    (
        mean_reward,
        mean_episode_len,
        termination_cause,
        is_success,
        avg_occupancies,
    ) = evaluate(model, env, 100)

    # Calculate Average Occupancy Ratio

    mean_avg_ratio = np.array(avg_occupancies).mean() if len(avg_occupancies) > 0 else 0
    avg_time_per_input = (time.time() - t1) / 100

    # Log Evaluation Metrics in STDOUT
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Mean Episode Length: {mean_episode_len:.2f}")
    print("Termination Cause:", termination_cause)
    print("Average Occupancy Ratio:", mean_avg_ratio)
    print("Average Time per Input:", avg_time_per_input)

    # Log Evaluation Metrics in MLflow
    mlflow.log_metric("avg_occupancy_ratio", mean_avg_ratio)
    mlflow.log_metric("avg_time_per_input", avg_time_per_input)
    for key in termination_cause.keys():
        mlflow.log_metric(key, termination_cause[key])
    mlflow.log_artifacts(logs_dir, "logs")
    mlflow.log_artifacts(models_dir,"models")

    # Clean up local directories models_dir and logs_dir
    shutil.rmtree(models_path, ignore_errors=True)
    shutil.rmtree(logs_path, ignore_errors=True)

    # End mlflow run session
    mlflow.end_run()
