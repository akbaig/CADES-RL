import numpy as np
from collections import defaultdict
def evaluate(model, env, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_rewards = []
    all_episodes_len = []
    termination_cause = defaultdict(int)
    is_success = 0
    occupancy_ratio = []
    free_space=[]

    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        total_bins = obs['nodes'] * env.norm_factor
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            if info['is_success']:
                is_success += 1
                remaining_node_capacities = obs['nodes'] * env.norm_factor
                occupancy_ratio.append(
                    round(100 - np.mean((np.array(remaining_node_capacities) / np.array(total_bins))) * 100, 2))

            if done:
                termination_cause[info['termination_cause']] += 1
                all_episodes_len.append(info['episode_len'])

            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)

    return mean_episode_reward, (sum(all_episodes_len) / num_episodes), termination_cause, is_success, occupancy_ratio
