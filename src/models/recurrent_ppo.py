from sb3_contrib import RecurrentPPO
from .model import Sb3Model
import numpy as np

class RecurrentPPOModel(Sb3Model):

    def __init__(self, env, config):
        super().__init__(env, config)

    def model_name(self):
        return "Recurrent_PPO"

    def initialize_model(self):
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

    def load_model(self):
        # Load the model from the saved path
        self.model = RecurrentPPO.load(
            self.config.model_path,
            env=self.env,
            device=self.config.device,
        )

    def evaluate(self):
        lstm_states = None
        episode_starts = np.array([True], dtype=bool)
        episode_reward = 0
        done = False
        obs = self.env.reset()
        info = {}

        while not done:
            action, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_starts)
            obs, reward, done, info = self.env.step(action)
            episode_starts = done
            episode_reward += reward

        # Collecting metrics
        metrics_results = {metric: info.get(metric, 0) for metric in self.metrics_to_eval}
        return {
            "episode_reward": episode_reward,
            "episode_length": info.get("episode_len", 0),
            "termination_cause": info.get("termination_cause", "unknown"),
            "metrics": metrics_results
        }