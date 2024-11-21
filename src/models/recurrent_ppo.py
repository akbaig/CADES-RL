import time
from sb3_contrib import RecurrentPPO
from .model import Sb3Model
import numpy as np

class RecurrentPPOModel(Sb3Model):

    def __init__(self, env, config, model=None):
        super().__init__(env, config, model)

    def model_name(self):
        return "Recurrent_PPO"

    def initialize(self):
        # Initialize the RL model
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            self.env,
            verbose=1,
            learning_rate=self.config.lr,
            tensorboard_log=f"../logs/{self.config.experiment_name}/{self.config.run_name}/",
            batch_size=self.config.batch_size,
            device=self.config.device,
            seed=self.config.seed,
        )
        return model

    @classmethod
    def load(cls, model_path, env, config):
        """
        Usage: RecurrentPPOModel.load(model_path, env, config)
        """
        model = RecurrentPPO.load(
            model_path,
            env,
            verbose=1,
            learning_rate=config.lr,
            tensorboard_log=f"../logs/{config.experiment_name}/{config.run_name}/",
            batch_size=config.batch_size,
            device=config.device,
            seed=config.seed,
        )
        model_instance = cls(env, config, model=model)
        return model_instance

    def evaluate(self, states=None):

        lstm_states = None
        episode_starts = np.array([True], dtype=bool)
        episode_reward = 0
        done = False
        obs = self.env.reset(states, training=False)
        info = {}
        actions = []

        inference_times = []
        while not done:
            inference_times.append(time.time())
            action, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_starts)
            inference_times[-1] = time.time() - inference_times[-1]
            obs, reward, done, info = self.env.step(action, training=False)
            episode_starts = done
            episode_reward += reward
            actions.append(action)

        # Collecting metrics
        metrics_results = {metric: info.get(metric, 0) for metric in self.metrics_to_eval}
        return {
            "obs": obs,
            "actions": actions,
            "episode_reward": episode_reward,
            "episode_length": info.get("episode_len", 0),
            "inference_time": np.sum(inference_times),
            "termination_cause": info.get("termination_cause", "unknown"),
            "metrics": metrics_results
        }