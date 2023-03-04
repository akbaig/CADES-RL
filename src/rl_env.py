from states_generator import StatesGenerator
import numpy as np
import random
import torch



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

