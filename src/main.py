from env.init import initialize_environment
from models.recurrent_ppo import RecurrentPPOModel

def _output_evaluation_results(result):
    # Extract common results from the dictionary
    mean_episode_reward = result["mean_episode_reward"]
    mean_episode_length = result["mean_episode_length"]
    termination_cause = result["termination_cause"]
    mean_metrics = result["mean_metrics"]

    # Print common results
    print(f"Mean Episode Reward: {mean_episode_reward}")
    print(f"Mean Episode Length: {mean_episode_length}")

    # Dynamically print all metrics
    for metric, value in mean_metrics.items():
        print(f"Mean {metric.replace('_', ' ').title()}: {value}%")

    # Print termination causes
    print("Termination Causes:")
    for cause, count in termination_cause.items():
        print(f"{cause}: {count}")

if __name__ == "__main__":
    env, config = initialize_environment()
    model = RecurrentPPOModel(env, config)
    model.train()
    if(config.inference == True):
        # Call evaluate_multiple to get the aggregated results
        result = model.evaluate_multiple() # by default evaluates for 100 episodes
        _output_evaluation_results(result)
