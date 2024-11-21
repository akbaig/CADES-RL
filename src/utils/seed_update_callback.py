from stable_baselines3.common.callbacks import BaseCallback
import hashlib
import numpy as np
import random

def generate_seed_name_train(epoch, episode):
    return f"train_epoch_{epoch}_episode_{episode}"

def generate_seed_name_eval(episode):
    return f"eval_episode_{episode}"

def generate_unique_seed(unique_string):
    # Create a unique string identifier for the epoch and iteration
    unique_identifier = unique_string
    # Encode the unique identifier to bytes
    encoded_identifier = unique_identifier.encode()
    # Create a hash object and hash the encoded identifier
    hash_object = hashlib.sha256(encoded_identifier)
    # Get the hexadecimal representation of the hash
    hex_hash = hash_object.hexdigest()
    # Convert the hex hash to an integer having max range 2^32
    seed = int(hex_hash, 16) % (2**32)
    return seed

class SeedUpdateCallback(BaseCallback):

    def __init__(self, train, verbose=0):
        super(SeedUpdateCallback, self).__init__(verbose)
        self.is_train = train
        self.epoch = 0
        self.episode = 0

    def _on_training_start(self) -> None:
        """
        This method is called when a new epoch starts
        """
        self.epoch += 1
        self.episode = 0
        self.on_episode_start()

    def on_episode_start(self) -> None:
        """
        This method is called when a new episode starts
        """
        self.episode += 1
        self.initialize_episode_seed()
    
    def _on_step(self) -> bool:
        """
        This method is called at each step
        """
        if self.locals['dones'][0]:
            # print("Episode Reward", self.locals['infos'][0]["total_reward"])
            # print("Episode Reward Details", self.locals['infos'][0]['reward_type'])
            # print("Episode Length", self.locals['infos'][0]['episode_len'])
            self.on_episode_start()

        return True  # Return True to continue training

    def initialize_episode_seed(self):
        """
        Initialize the seed for the episode
        """
        if self.is_train:
            seed_name = generate_seed_name_train(self.epoch, self.episode)
        else:
            seed_name = generate_seed_name_eval(self.episode)
        seed = generate_unique_seed(seed_name)
        # print("Seed:", seed_name, seed)
        random.seed(seed)
        np.random.seed(seed)
        # if 'env' in self.locals:
        #     self.locals['env'].seed(seed)