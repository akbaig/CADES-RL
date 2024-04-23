from env.init import initialize_environment
from models.recurrent_ppo import RecurrentPPOModel
from utils.mlflow import MLFlowManager

if __name__ == "__main__":

    env, config = initialize_environment()
    model = RecurrentPPOModel(env, config)
    mlflow_manager = MLFlowManager(model, config)
    mlflow_manager.run()