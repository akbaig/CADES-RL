import time
from sb3_contrib import MaskablePPO
from utils.metrics_callback import MetricsCallback
from utils.seed_update_callback import SeedUpdateCallback
from .model import Sb3Model
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import CallbackList

class MaskablePPOModel(Sb3Model):

    def __init__(self, env, config, model=None):
        super().__init__(env, config, model)

    def model_name(self):
        return "Maskable_PPO"

    def initialize(self):
        # Initialize the RL model
        model = MaskablePPO(
            "MultiInputPolicy",
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
        model = MaskablePPO.load(
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
    
    def _eval_callbacks(self, save_dir):
        metrics_callback = MetricsCallback(
            self.env,
            best_model_save_path=f"{save_dir}/models",
            log_path=f"{save_dir}/logs",
            eval_freq=self.config.eval_timesteps,
            deterministic=True,
            render=False,
            use_masking=True
        )
        callback_list = CallbackList([metrics_callback])
        return callback_list

    def evaluate(self, states=None):

        episode_reward = 0
        done = False
        obs, _ = self.env.reset(states=states, training=False)
        info = {}
        actions = []

        inference_times = []
        while not done:
            inference_times.append(time.time())
            action_masks = get_action_masks(self.env)
            action, _states = self.model.predict(obs, action_masks=action_masks)
            inference_times[-1] = time.time() - inference_times[-1]
            obs, reward, done, _, info = self.env.step(action, training=False)
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