"""
Logic to generate new states and compute the reward for each state-action pair.
"""

import numpy as np
from torch import dtype
import random
import torch

class StatesGenerator(object):
    """
    Helper class used to randomly generate batches of states given a set
    of problem conditions, which are provided via the `config` object.
    """

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.min_num_items = config.min_num_items
        self.max_num_items = config.max_num_items
        self.min_item_size = config.min_item_size
        self.max_item_size = config.max_item_size

        self.min_bin_size=config.min_bin_size
        self.max_bin_size=config.max_bin_size
        self.total_bins=config.total_bins

        self.num_critical_items = config.number_of_critical_items
        self.num_critical_copies = config.number_of_copies
        self.ci_groups = []

    def generate_critical_items(self, items_seqs_batch, items_len_mask, items_seq_lens):
        '''
            Generate critical items and their replicas:
            - `items_seqs_batch`: batch of only normal items
            - `items_len_mask`: mask of normal items, list of 1 and 0
            -  `items_seq_lens`: indicates the length of the items in each batch
        '''
        batch_critical_items = []
        critical_copy_mask = []
        items_with_critical = items_seqs_batch.copy()
        batch_ci_groups = []
        for items_seq, len_mask, seq_len in zip(items_with_critical, items_len_mask, items_seq_lens):
            critical_items = [sample[0] for sample in random.sample(list(enumerate(items_seq[:seq_len])), k=self.num_critical_items)]
            batch_critical_items.append(critical_items)
            critical_mask = len_mask.copy()
            ci_groups = []
            for idx, ci in enumerate(critical_items):
                critical_mask[ci] = 2+idx # First Change the mask of the original critical items
            for idx, ci in enumerate(critical_items): # Create copies for the critical items and change their value and mask
                critical_item_copies = random.sample(list(np.where(critical_mask==1.)[0]), k=2)
                critical_mask[critical_item_copies] = 2 + idx
                items_seq[critical_item_copies] = items_seq[ci]
                ci_groups.append([ci]+critical_item_copies)
            critical_copy_mask.append(critical_mask)
            batch_ci_groups.append(ci_groups)
        return (items_with_critical, critical_copy_mask, batch_ci_groups)



    def generate_states_batch(self, batch_size=None):
        """Generate new batch of initial states"""
        if batch_size is None:
            batch_size = self.batch_size
        items_seqs_batch = np.random.randint(
            low=self.min_item_size,
            high=self.max_item_size + 1,
            size=(batch_size, self.max_num_items),
        )

        items_len_mask = np.ones_like(items_seqs_batch, dtype="float32")
        items_seq_lens = np.random.randint(
            low=self.min_num_items, high=self.max_num_items + 1, size=batch_size
        )

        bins_available=[]
        for i in range(self.total_bins):
            bins_available.append(self.min_bin_size+ i*200)
        bins_available=np.array(bins_available)
        bins_available=np.repeat(bins_available[np.newaxis, ...], batch_size, axis=0)

        # bins_available = np.random.randint(
        #     low=self.min_bin_size, high=self.max_bin_size + 1,
        #     size=(batch_size,self.total_bins)
        # )

        for items_seq, len_mask, seq_len in zip(
            items_seqs_batch, items_len_mask, items_seq_lens
        ):
            items_seq[seq_len:] = 0
            len_mask[seq_len:] = 0

        return (
            items_seqs_batch,
            items_seq_lens,
            items_len_mask,
            bins_available
        )



def get_active_bins(heuristic, items_order, items_size, bin_size):
    bins = [0]
    bin_status = [[]]
    for item_idx in items_order:
        if item_idx == -1: # sequence is shorter than max_num_items 
            continue
        item_size = items_size[item_idx]
        if heuristic == "NF":
            if bins[-1] + item_size <= bin_size:
                bins[-1] += item_size
                bin_status[-1].append(item_idx)
            else:
                bin_status.append([item_idx])
                bins.append(item_size)
        elif heuristic == "FF":
            for bin_idx, bin_occupancy in enumerate(bins):
                if bin_occupancy + item_size <= bin_size:
                    bins[bin_idx] += item_size
                    bin_status[bin_idx].append(item_idx)
                    break
            else:
                bins.append(item_size)
                bin_status.append([item_idx])
        
    return bins, bin_status

def avg_occupancy(
    bin_size: int, items_size: tuple[float], items_order: tuple[int], heuristic: str
) -> float:
    """
    Calculates the average occupancy of the used bins given:
    - `bin_size`: the bin size (assumed constantfor all bins)
    - `items_size`: the size of the items to be allocated
    - `items_order`: the order in which the items are allocated (this is the output of the
    pointer network)
    - `heuristic`: the heuristic used to allocate the items in the bins given the order.
      It can be either "NF" (next fit) or "FF" (first fit)
    
    """
    if heuristic not in ("NF", "FF"):
        raise ValueError(f"Unknown heuristic: {heuristic}")
    bins,_ = get_active_bins(heuristic, items_order, items_size, bin_size)
    return np.mean(np.array(bins) / bin_size)

def compute_ci_reward(config, state, item_order, ci_pairs, heuristic):
    bin_size = config.bin_size
    num_copies = config.number_of_copies + 1
    _, bin_status = get_active_bins(heuristic, item_order, state, bin_size)
    reward=0
    for pair in (ci_pairs):
        bins_occupied=0
        for i in range(len(bin_status)):
            res=np.intersect1d(np.array(bin_status[i]),list(pair))
            if len(res):
                bins_occupied=bins_occupied+1
        # reward=reward+bins_occupied/num_copies

    #     Changing reward function a little bit
        if bins_occupied==num_copies:
            reward=reward+1
    reward=reward/len(ci_pairs)
    # print(bin_status,reward)
    return reward


def critical_task_reward(config, critical_items, allocation_order, batch_ci_pairs, heuristic):
    
    allocation_order = allocation_order.numpy().astype(int) if torch.is_tensor(allocation_order) else allocation_order
    ci_reward_batch_avg = []
    for states, actions, ci_pairs in zip(critical_items, allocation_order, batch_ci_pairs):
        reward = compute_ci_reward(config, states, actions, ci_pairs, heuristic)
        ci_reward_batch_avg.append(reward)

    return np.array(ci_reward_batch_avg)
    

def compute_reward(config, states_batch, len_mask, actions_batch):
    """
    Compute the average occupancy ratio for each state-action pair in the batch.
    """
    bin_size = config.bin_size
    # states_batch = states_batch.squeeze(-1).numpy()
    actions_batch = actions_batch.numpy().astype(int)
    avg_occupancy_ratios = []

    for states, actions in zip(states_batch, actions_batch):
        avg_occupancy_ratios.append(avg_occupancy(bin_size, states, actions, heuristic=config.agent_heuristic))
    return np.array(avg_occupancy_ratios)

def get_benchmark_rewards(config, states_generator: StatesGenerator=None, states_batch=None, ci_groups=None):
    """
    Compute the average occupancy ratio following the NF, FF and FFD heuristics. 
    
    If the arg `states_generator` is provided, the states (i.e. sequences of items to allocate) are randomly
    generated during 1,000 loops and the retruend values are the average.
    If no `states_generator`arg is provided and a `states_batch` arg is provided, then the average 
    occupancy ratio is computed for the provided batch.

    Returns a tuple with 3 values corresponding to the average occupancy ratio obtained following
    a NF, FF and FFD heuristic respectively.
    """
    nf_reward, ff_reward, ffd_reward = [], [], []
    nf_ci_reward, ff_ci_reward, ffd_ci_reward = [], [], []
    if states_generator is not None:
        regular_items, states_lens, len_mask = states_generator.generate_states_batch(
            batch_size=1000
        )
        states, ci_copy_mask, ci_groups = states_generator.generate_critical_items(
            regular_items, len_mask, states_lens
        )
    else:
        states = states_batch

    items_order_default = np.arange(config.max_num_items)
    for state, ci_group in zip(states, ci_groups):

        nf_reward.append(
            avg_occupancy(config.bin_size, state, items_order_default, heuristic="NF")
        )
        nf_ci_reward.append(
           compute_ci_reward(config, state, items_order_default, ci_group, heuristic="NF") 
        )
        ff_reward.append(
            avg_occupancy(config.bin_size, state, items_order_default, heuristic="FF")
        )
        ff_ci_reward.append(
           compute_ci_reward(config, state, items_order_default, ci_group, heuristic="FF") 
        )
        items_order_decreasing = np.flip(np.argsort(state))
        ffd_reward.append(
            avg_occupancy(config.bin_size, state, items_order_decreasing, heuristic="FF")
        )
        ffd_ci_reward.append(
            compute_ci_reward(config, state, items_order_decreasing, ci_group, heuristic="FF")
        )
    nf = {"avg_occ": np.mean(nf_reward), "ci": np.mean(nf_ci_reward)}
    ff = {"avg_occ": np.mean(ff_reward), "ci": np.mean(ff_ci_reward)}
    ffd = {"avg_occ": np.mean(ffd_reward), "ci": np.mean(ffd_ci_reward)}
    return nf, ff, ffd


import gym
import numpy as np
from gym import spaces
from enum import Enum

class TerminationCause(Enum):
    SUCCESS = 1
    DUBLICATE_PICK = 2
    BIN_OVERFLOW = 3


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.config=config
        self.states_generator = StatesGenerator(config)


        self.action_space=spaces.MultiDiscrete([config.max_num_items,config.total_bins])
        self.observation_space = spaces.Box(low=-1, high=1, shape=(config.total_bins+config.max_num_items,), dtype=np.float)

        # self.observation_space = spaces.Dict(
        #     {
        #         "tasks": spaces.Discrete(2 ** n_bits),
        #         "bins": spaces.Discrete(2 ** n_bits),
        #         "masks": spaces.Discrete(2 ** n_bits),
        #     }
        # )
        self.assignment_status=[]
        for i in range(config.total_bins):
            self.assignment_status.append([])
        self.current_state=[]
        self.total_reward=0
        self.done=False

    def step(self, action):
        # observation = self.observation_space
        observation = self.current_state
        done=False

        selected_item_idx = action[0]
        selected_bin_idx = action[1]+ self.config.max_num_items
        selected_item_cost=self.current_state[selected_item_idx]

        # Agent picked the item which is already used
        if selected_item_cost == 0:
            reward = -20
            done = True
            self.info['termination_cause']=TerminationCause.DUBLICATE_PICK.name
        else:
            # Placing the item in bin
            if selected_item_cost<=self.current_state[selected_bin_idx]:
                # Mark the selected item as zero
                reward=1
                self.current_state[selected_item_idx] = 0
                self.current_state[selected_bin_idx]-=selected_item_cost
                self.assignment_status[selected_bin_idx%self.config.max_num_items].append(selected_item_idx)
                self.info['episode_len']=self.info['episode_len']+1
                if sum(self.current_state[0:self.config.max_num_items])==0:
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
        observation= np.array(list(states[0])+ list(bins_available[0]))/max(list(bins_available[0]))
        self.current_state=observation
        return observation
    def render(self, mode="human"):
        pass

    def close(self):
        pass