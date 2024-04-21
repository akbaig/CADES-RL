from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

class Sb3Model(ABC):

    def __init__(self, env, config):
        self.metrics_to_eval = ["avg_node_occupancy", "message_channel_occupancy", "empty_nodes"]
        self.env = env
        self.config = config
        self.model = self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def model_name(self):
        pass

    def train(self):
        models_dir = f"../models/{self.config.experiment_name}/{self.config.run_name}/"
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=models_dir,
            log_path=f"../logs/{self.config.experiment_name}/{self.config.run_name}/",
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        EPOCHS = self.config.epochs
        TIMESTEPS = 10000
        iters = 0

        while iters < EPOCHS:
            iters += 1
            print("Epoch #", iters)
            self.model.learn(
                total_timesteps=TIMESTEPS,
                log_interval=10000,
                reset_num_timesteps=False,
                tb_log_name=self.model_name(),
                callback=eval_callback,
            )
            self.model.save(f"{models_dir}/{iters}")

    def evaluate_multiple(self, num_episodes=100):
        all_episode_rewards = []
        all_episodes_len = []
        termination_cause = defaultdict(int)

        # Initialize dictionary to store lists of results for each metric
        metrics_accumulator = {metric: [] for metric in self.metrics_to_eval}

        for _ in range(num_episodes):
            results = self.evaluate()
            all_episode_rewards.append(results["episode_reward"])
            all_episodes_len.append(results["episode_length"])
            termination_cause[results["termination_cause"]] += 1

            # Accumulate each metric's results
            for metric, value in results["metrics"].items():
                metrics_accumulator[metric].append(value)

        # Calculate the mean for each metric
        metrics_means = {metric: np.mean(values) if values else 0 for metric, values in metrics_accumulator.items()}

        return {
            "mean_episode_reward": np.mean(all_episode_rewards),
            "mean_episode_length": np.mean(all_episodes_len) if all_episodes_len else 0,
            "termination_cause": dict(termination_cause),
            "mean_metrics": metrics_means
        }