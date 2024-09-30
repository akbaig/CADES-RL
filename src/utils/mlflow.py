import numpy as np
import mlflow
import sys
from typing import Any, Dict, Tuple, Union
from urllib.parse import urlparse
from stable_baselines3.common.logger import KVWriter, HumanOutputFormat, Logger

def setup_logger():
    loggers = Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
    )
    return loggers

def expand_result_dict(result):
        expanded_result = {}
        # Expand termination_cause dictionary
        for cause, count in result.get("termination_cause", {}).items():
            expanded_result[cause] = count
        # Expand mean_metrics dictionary
        for metric, value in result.get("mean_metrics", {}).items():
            expanded_result[metric] = value
        # Add other metrics
        expanded_result["mean_inference_time"] = result.get("mean_inference_time", 0)
        expanded_result["mean_episode_reward"] = result.get("mean_episode_reward", 0)
        expanded_result["mean_episode_length"] = result.get("mean_episode_length", 0)
        return expanded_result

class MLFlowManager:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        mlflow.set_experiment(self.config.experiment_name)

    def log_config(self):
        config_dict = vars(self.config)
        for key in config_dict:
            mlflow.log_param(key, config_dict[key])

    def log_metrics(self, metrics):
        for key in metrics:
            mlflow.log_metric(key, metrics[key])

    def get_run_artifact_uri(self):
        return urlparse(mlflow.get_artifact_uri()).path

    def run(self, run_name=None):
        if run_name is None:
            run_name = self.config.run_name
        with mlflow.start_run(run_name=run_name):
            # Log Config Paramaters
            self.log_config()
            # Setup Logger for Metrics
            logger = setup_logger()
            self.model.set_logger(logger)
            # Train Model
            if self.config.train:
                save_path = self.get_run_artifact_uri()
                self.model.train(save_path)
            # Evaluate Model
            if self.config.inference:
                result = self.model.evaluate_multiple()
                expanded_result = expand_result_dict(result)
                self.log_metrics(expanded_result)
                print(expanded_result)

class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)