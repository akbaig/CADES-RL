
import gym
import numpy as np
from gym import spaces
from enum import Enum
from states_generator import StatesGenerator

class TerminationCause(Enum):
    SUCCESS = 1
    DUBLICATE_PICK = 2
    BIN_OVERFLOW = 3


class CadesEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.config=config
        self.states_generator = StatesGenerator(config)
        self.norm_factor=None

        self.action_space=spaces.MultiDiscrete([config.max_num_items,config.total_bins])
        self.observation_space = spaces.Dict({
            "tasks": spaces.Box(low=0, high=1, shape=(config.max_num_items,), dtype=np.float),
            "nodes": spaces.Box(low=0, high=1, shape = (config.total_bins,), dtype = np.float)
        })

        self.env_stats={}
        self.assignment_status=[]
        for i in range(config.total_bins):
            self.assignment_status.append([])
        self.current_state={}
        self.total_reward=0
        self.done=False



    def step(self, action):
        # observation = self.observation_space
        observation = self.current_state

        done=False

        selected_item_idx = action[0]
        selected_bin_idx = action[1]
        selected_item_cost=self.current_state["tasks"][selected_item_idx]

        # Agent picked the item which is already used
        if selected_item_cost == 0:
            reward = -20
            done = True
            self.info['termination_cause']=TerminationCause.DUBLICATE_PICK.name
        else:
            # Placing the item in bin
            if selected_item_cost<=self.current_state['nodes'][selected_bin_idx]:
                # Mark the selected item as zero
                reward=1
                self.current_state['tasks'][selected_item_idx] = 0
                self.current_state['nodes'][selected_bin_idx]-=selected_item_cost
                self.assignment_status[selected_bin_idx].append(selected_item_idx)
                self.info['episode_len']=self.info['episode_len']+1
                if sum(self.current_state['tasks'])==0:
                    reward=20
                    self.info['termination_cause'] = TerminationCause.SUCCESS.name
                    self.info['is_success']=True
                    done=True
            else:
                reward= -10
                done = True
                self.info['termination_cause'] =  TerminationCause.BIN_OVERFLOW.name

        self.info['assignment_status']=self.assignment_status
        self.total_reward=self.total_reward+reward
        return observation, reward, done, self.info

    def reset(self):

        self.assignment_status = []
        for i in range(self.config.total_bins):
            self.assignment_status.append([])

        self.info={
                "is_success":False,
                "episode_len":0,
                "termination_cause":None
              }
        self.done=False
        self.total_reward=0
        states, states_lens, states_mask, bins_available = self.states_generator.generate_states_batch()
        self.norm_factor= max(list(bins_available[0]))

        observation= {
            'tasks':np.array(list(states[0])/ self.norm_factor),
            'nodes':np.array(list(bins_available[0])/ self.norm_factor)
        }
        self.current_state=observation
        self.env_stats['tasks_total_cost']=sum(observation['tasks']*self.norm_factor)
        self.env_stats['nodes_total_capacity']=sum(observation['nodes']*self.norm_factor)
        self.env_stats['extra_capacity']=round(1-(self.env_stats['tasks_total_cost']/self.env_stats['nodes_total_capacity']),2)*100

        return observation
    def render(self, mode="human"):
        pass
    def get_env_info(self):
        return self.env_stats
    def close(self):
        pass

