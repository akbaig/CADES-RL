from env.init import initialize_environment
from models.ppo import PPOModel
from utils.mlflow import MLFlowManager

if __name__ == "__main__":

    env, config = initialize_environment()
    if config.train is False and config.inference is False:
        raise ValueError("Either train or inference mode should be enabled")
    elif config.train is False and config.inference is True:
        if config.model_path is None or config.model_path == "":
            raise ValueError("model_path argument should be provided for inference mode")
        else:
            model = PPOModel.load(config.model_path, env, config)
    else:
        model = PPOModel(env, config)
    mlflow_manager = MLFlowManager(model, config)
    mlflow_manager.run()