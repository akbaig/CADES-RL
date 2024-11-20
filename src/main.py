from env.init import initialize_environment
from models.ppo import PPOModel
from utils.mlflow import MLFlowManager

if __name__ == "__main__":

    env, config = initialize_environment()
    model = PPOModel(env, config)
    mlflow_manager = MLFlowManager(model, config)
    mlflow_manager.run()