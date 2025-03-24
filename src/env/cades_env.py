import gymnasium as gym
import random
import numpy as np
from gymnasium import spaces
from enum import Enum
from utils.eval_metrics import (
    get_avg_node_occupancy,
    get_avg_active_node_occupancy,
    get_empty_nodes_percentage,
    get_evaluate_message_channel_occupancy
)
from env.extended_states_generator import ExtendedStatesGenerator
import copy


class TerminationCause(Enum):
    SUCCESS = (1, "success")
    DUPLICATE_PICK = (2, "duplicate_pick")
    NODE_OVERFLOW = (3, "node_overflow")
    DUPLICATE_CRITICAL_PICK = (4, "duplicate_critical_pick")

    def __init__(self, id, description):
        self.id = id
        self.description = description

    def __str__(self):
        return self.description

class CadesEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        """
        Initializes observation and action spaces
        """
        super().__init__()

        self.config = config
        self.states_generator = ExtendedStatesGenerator(config, self)
        self.norm_factor = None

        self.action_space = spaces.MultiDiscrete(
            [config.max_num_tasks, config.max_num_nodes]
        )

        self.observation_space = spaces.Dict(
            {
                "tasks": spaces.Box(
                    low=0, high=1, shape=(config.max_num_tasks,), dtype=float
                ),
                "critical_mask": spaces.Box(
                    low=0, high=1, shape=(config.max_num_tasks,), dtype=float
                ),
                "nodes": spaces.Box(
                    low=0, high=1, shape=(config.max_num_nodes,), dtype=float
                ),
                "communications": spaces.MultiBinary(
                   (config.max_num_tasks, config.max_num_tasks)
                ),
            }
        )

        self.env_stats = {}        
        self.current_state = {}

    def _is_task_critical(self, task_index):
        """
        Checks if a task is critical or not i.e. has mask value greater than one
        """
        return self.current_state["critical_mask"][task_index] > 0

    def _is_critical_task_duplicated(self, task_index, node_index):
        """
        Checks if a critical task has any of its replicas in a node
        """
        critical_mask = self.initial_state["critical_mask"]
        # get indices which have same mask value
        replica_indices = list(np.where(critical_mask == critical_mask[task_index])[0])
        # check if these indices are in the assignment status of selected node
        return (
            np.intersect1d(replica_indices, self.assignment_status[node_index]).size > 0
        )

    def _get_task_receivers(self, task_index):
        """
        Gets the receivers for a task, mentioned in communication matrix
        """
        receivers = self.current_state["communications"][task_index]
        return np.where(receivers == 1)[0] # return indices

    def _get_task_senders(self, task_index):
        """
        Gets the senders for a task, mentioned in communication matrix
        """
        senders = self.current_state["communications"][:, task_index]
        return np.where(senders == 1)[0] # return indices

    def _get_tasks_placed_in_node(self, list_of_tasks, node_index):
        """
        Gets the tasks from list_of_tasks which are already present in the node
        """
        return np.intersect1d(list_of_tasks, self.assignment_status[node_index])

    def _get_random_valid_task(self):
        """
        Returns a valid task index from the available tasks
        """
        valid_task_indices = [
            idx for idx, task in enumerate(self.current_state["tasks"]) if task > 0
        ]
        if not valid_task_indices:  # No valid task left
            raise ValueError("No valid task found")
        else:  # Choose a new task from the valid tasks
            new_task_idx = random.choice(valid_task_indices)
        return new_task_idx
    
    def _get_random_valid_node_for_task(self, task_idx):
        """
        Returns a valid node index from the available nodes for the given task
        """
        valid_node_indices = [
            idx for idx, node in enumerate(self.current_state["nodes"]) if node >= self.current_state["tasks"][task_idx]
        ]
        if not valid_node_indices:  # No valid node left
            raise ValueError("No valid node found")
        else:  # Choose a new node from the valid node
            new_node_idx = random.choice(valid_node_indices)
        return new_node_idx

    def _exponential_decay_reward(self, step, max_steps, max_reward, k=2):
        """
        An adjusted exponential decay formula to reach exactly 0 at the final step
        Where k is an exponent that determines how quickly the function approaches zero. 
        Higher values of k will make the decay steeper towards the end.
        """
        reward = max_reward * (1 - (step / max_steps) ** k)
        return reward
    
    def _reward_unit(self):
        # reward can be multiplied with this unit to balance out the effect of different episode lengths
        return self.config.max_num_tasks / self.env_stats["tasks_len"]
    
    def _episode_length_norm_factor(self):
        # Lmax = max_steps_in_all_episodes
        Lmax = self.config.max_num_tasks
        # Lcurrent = max_steps_in_current_episode
        Lcurrent = self.env_stats["tasks_len"]
        # factor = Lmax(Lmax + 1) / Lcurrent(Lcurrent + 1)
        factor = (Lmax * (Lmax + 1)) / (Lcurrent * (Lcurrent + 1))
        # return factor
        return factor

    def _reward(self, action, training=True):
        """
        Reward function for the environment, returns the episode termination signal and reward for the timestep
        """
        done = False
        # Agent outputs two actions, one for task index and one for node index
        selected_task_idx = action[0]
        selected_node_idx = action[1]
        selected_task_cost = self.current_state["tasks"][selected_task_idx]
        reward_type = ""

        # Agent picked the task which is already used
        if selected_task_cost == 0:
            step = self.info["episode_len"]
            max_steps = self.env_stats["tasks_len"]
            max_reward = self.config.DUPLICATE_PICK_reward
            reward = self._exponential_decay_reward(step, max_steps, max_reward)
            reward_type = f"Duplicate Pick Reward on Step {step}: {reward}"
            if training and self.config.invalid_action_replacement is True:
                # Select any other valid action
                    valid_task_idx = self._get_random_valid_task()
                    _, done = self._reward([valid_task_idx, selected_node_idx], training)
            else:
                # If not, terminate the episode (No replacement in evaluation mode)
                done = True
                self.info["termination_cause"] = str(TerminationCause.DUPLICATE_PICK)
        
        # Agent picked the node which is already full
        elif selected_task_cost > self.current_state["nodes"][selected_node_idx]:
            reward = (
                self.config.NODE_OVERFLOW_reward 
                * (self.info["episode_len"] * self.ep_len_norm_factor) 
                * 0.25
            )
            reward_type = f"Node Overflow Reward: {reward}"
            # if training and self.config.invalid_action_replacement is True:
            #     # Select any other valid action
            #     valid_node_idx = self._get_random_valid_node_for_task(selected_task_idx)
            #     _, done = self._reward([selected_task_idx, valid_node_idx], training)
            # else:
            # If not, terminate the episode (No replacement in evaluation mode)
            done = True
            self.info["termination_cause"] = str(TerminationCause.NODE_OVERFLOW)

        # Agent picked the node which already had critical task
        elif self._is_task_critical(
            selected_task_idx
        ) and self._is_critical_task_duplicated(selected_task_idx, selected_node_idx):
            reward = (
                self.config.DUPLICATE_CRITICAL_PICK_reward
                * (self.info["episode_len"] * self.ep_len_norm_factor)
                * 0.15
            )
            reward_type = f"Duplicate Critical Pick Reward: {reward}"
            done = True
            self.info["termination_cause"] = (
                str(TerminationCause.DUPLICATE_CRITICAL_PICK)
            )

        # Agent picked the correct task and node
        else:
            # Assign Rewards
            reward = (self.config.STEP_reward * self.reward_unit)
            reward += (
                (self.config.BONUS_reward * self.reward_unit) 
                * (self.info["episode_len"] * self.ep_len_norm_factor)
            )
            reward_type = f"Step and Bonus Reward: {reward}"
            if self._is_task_critical(selected_task_idx):
                reward += self.config.CRITICAL_reward
                reward_type += f' \nCritical Reward: {reward}'
            # Check if the task is communicating
            task_receivers = self._get_task_receivers(selected_task_idx)
            task_senders = self._get_task_senders(selected_task_idx)
            # if the task is a sender i.e has receivers
            if(len(task_receivers) > 0): 
                # narrow down the receivers to the ones that are already placed in the node
                allocated_receivers = self._get_tasks_placed_in_node(task_receivers, selected_node_idx)
                if(len(allocated_receivers) > 0):
                    # assign reward
                    reward += (self.config.COMM_reward/self.env_stats["comms_len"]) * len(allocated_receivers)
                    reward_type += f' \nCommunication Reward for {selected_task_idx} communicating with {allocated_receivers}: {reward}'
                    # set the communication mask to zero
                    self.current_state["communications"][selected_task_idx, allocated_receivers] = 0
                    # add pair to communication status
                    for receiver in allocated_receivers:
                        self.communication_status.add((selected_task_idx, receiver))
            # if the task is a receiver i.e has senders
            if(len(task_senders) > 0):
                # narrow down the senders to the ones that are already placed in the node
                allocated_senders = self._get_tasks_placed_in_node(task_senders, selected_node_idx)
                if(len(allocated_senders) > 0):
                    # assign reward
                    reward += (self.config.COMM_reward/self.env_stats["comms_len"]) * len(allocated_senders)
                    reward_type += f' \nCommunication Reward for {allocated_senders} communicating with {selected_task_idx}: {reward}'
                    # set the communication mask to zero
                    self.current_state["communications"][allocated_senders, selected_task_idx] = 0
                    # add pair to communication status
                    for sender in allocated_senders:
                        self.communication_status.add((sender, selected_task_idx))
            # Set the selected task mask value as zero
            self.current_state["critical_mask"][selected_task_idx] = 0
            # Mark the selected task as zero
            self.current_state["tasks"][selected_task_idx] = 0
            # Consume the space in selected bin
            self.current_state["nodes"][selected_node_idx] -= selected_task_cost
            # Update Assignment status
            self.assignment_status[selected_node_idx].append(selected_task_idx)
            self.info["episode_len"] = self.info["episode_len"] + 1
            # Check if no task is remaining
            if sum(self.current_state["tasks"]) == 0:
                reward += self.config.SUCCESS_reward
                reward_type += f"\n Success Reward: {reward}"
                self.info["termination_cause"] = str(TerminationCause.SUCCESS)
                self.info["is_success"] = True
                done = True

        self.info["reward_type"] += f'{reward_type}\n'
        return reward, done
    
    def _get_lowest_cost_task(self):
        """
        Returns the index of the task with the lowest cost
        """
        return np.argmin(self.current_state["tasks"])

    def _is_task_valid(self, task_index):
        """
        Checks if the task is valid i.e. has a cost greater than zero
        """
        return self.current_state["tasks"][task_index] > 0
    
    def _is_node_valid_for_task(self, task_index, node_index):
        """
        Check if the selected node has enough space for the selected task
        """
        cost = self.current_state["tasks"][task_index]
        return self.current_state["nodes"][node_index] >= cost
    
    def action_masks(self):
        action_dim1 = self.config.max_num_tasks
        action_dim2 = self.config.max_num_nodes
        mask_dim1 = np.zeros(action_dim1, dtype=bool)
        mask_dim2 = np.zeros(action_dim2, dtype=bool)
        # Generate masks for the tasks
        for i in range(action_dim1):
            mask_dim1[i] = self._is_task_valid(i)
        # Get the lowest cost task and keep only the nodes that can accommodate it
        lowest_cost_task_idx = self._get_lowest_cost_task()
        # Generate masks for the nodes
        for i in range(action_dim2):
            mask_dim2[i] = self._is_node_valid_for_task(lowest_cost_task_idx, i)
        # Concatenate these masks
        masks_np = np.concatenate([mask_dim1, mask_dim2])
        return masks_np

    def _verbose(self, action, reward):
        """
        Prints information about the current timestep
        """
        print(
            f"Observation Space:\n"
            f"Tasks: {self.current_state['tasks']}\n"
            f"Critical Masks: {self.current_state['critical_mask']}\n"
            f"Nodes: {self.current_state['nodes']}\n"
            f"Communications: {self.communication_status}\n"
            f"Assignment Status: {self.assignment_status}"
        )
        print(
            f"Last Action: Selected task: {action[0]} Selected node: {action[1]}\n"
            f"Avg Node Occupancy: {self.info['avg_node_occupancy']}%\n"
            f"Message Channel Occupancy: {self.info['message_channel_occupancy']}%\n"
            f"Empty Nodes: {self.info['empty_nodes']}%\n"
            f"Episode Reward: {reward}\n"
            f"Termination Cause: {self.info['termination_cause']}\n"
        )

    def step(self, action, training=True):
        """
        Advances the episode by one timestep using the given action. 
        """
         # Calc Rewards
        reward, done = self._reward(action, training)
        # Save Info about Episode
        self.info["commnication_status"] = self.communication_status
        self.info["assignment_status"] = self.assignment_status
        self.env_stats["intranode_comms_len"] = len(self.communication_status)

        # Calculate Evaluation Metrics
        self.info["avg_node_occupancy"] = get_avg_node_occupancy(
            self.initial_state["nodes"] * self.norm_factor, # nodes total capacities
            self.current_state["nodes"] * self.norm_factor # nodes remaining capacities
        )
        self.info["avg_active_node_occupancy"] = get_avg_active_node_occupancy(
            self.initial_state["nodes"] * self.norm_factor, # nodes total capacities
            self.current_state["nodes"] * self.norm_factor # nodes remaining capacities
        )
        self.info["message_channel_occupancy"] = get_evaluate_message_channel_occupancy(
            self.env_stats["comms_len"], # total comms
            self.env_stats["intranode_comms_len"] # intranode comms
        )
        self.info["empty_nodes"] = get_empty_nodes_percentage(
            self.assignment_status
        )
        # if done is True:
            # Add reward based on avg active node occupancy
            # reward += self.config.NODE_OCCUPANCY_reward * (self.info["avg_active_node_occupancy"] / 100)
            # Add reward based on message channel occupancy in reverse
            # reward += self.config.MESSAGE_CHANNEL_OCCUPANCY_reward * (1 - (self.info["message_channel_occupancy"] / 100))

        # Update total reward
        self.info["total_reward"] += reward

        # If verbose is toggled, print info about timestep
        if done is True and self.config.verbose is True and self.info["is_success"] is True:
            self._verbose(action, reward)

        observation = self.current_state
        return observation, reward, done, False, self.info # observation, reward, done, truncated, extra_info
    
    def generate_states(self, training=True):
        """
        Generates states for the environment
        """
        (
            tasks,
            num_tasks,
            nodes,
            num_nodes
        ) = self.states_generator.generate_tasks_and_nodes()
        critical_mask = self.states_generator.generate_critical_tasks_and_replicas(
            tasks, num_tasks
        )
        # Use graph during evaluation and also in training if specified
        use_graph = (not training) or self.config.use_comm_graph_in_train
        comms, num_comms = self.states_generator.generate_communications(
            tasks, num_tasks, critical_mask, graph=use_graph
        )
        generated_states = {
            "tasks": tasks,
            "num_tasks": num_tasks,
            "critical_mask": critical_mask,
            "nodes": nodes,
            "num_nodes": num_nodes,
            "communications": comms,
            "num_communications": num_comms,
        }
        return generated_states
    
    def set_states_random_seed(self, seed):
        """
        Sets the seed for base class's random number generator
        """
        super().reset(seed=seed)

    def reset(self, states=None, training=True, seed=None):
        """
        Initializes new states for the start of a new episode.
        """
        # assignment status is an variable-sized 2D Array, having dimensions total_nodes x (size of node)
        # it stores the indices of task assignment on the nodes
        if seed is not None:
            self.set_states_random_seed(seed)
        self.assignment_status = []
        self.communication_status = set()
        self.info = {"is_success": False, "episode_len": 0, "termination_cause": None, "reward_type": "", "total_reward": 0}
        if states is None:
            states = self.generate_states(training)
        for _ in range(states["num_nodes"]):
            self.assignment_status.append([])
        # norm factor is the largest node size
        self.norm_factor = np.max(states["nodes"])
        # critical norm factor is the largest mask value in critical mask
        self.critical_norm_factor = np.max(states["critical_mask"]) or 1 # to avoid division by zero if no critical task
        observation = {
            "tasks": np.array(states["tasks"] / self.norm_factor),
            "critical_mask": np.array(states["critical_mask"] / self.critical_norm_factor),
            "nodes": np.array(states["nodes"] / self.norm_factor),
            "communications": np.array(states["communications"]),
        }
        self.initial_state = copy.deepcopy(observation)
        self.current_state = observation
        self.env_stats["tasks_len"] = states["num_tasks"]
        self.env_stats["comms_len"] = states["num_communications"]
        self.env_stats["tasks_total_cost"] = sum(
            observation["tasks"] * self.norm_factor
        )
        self.env_stats["nodes_total_capacity"] = sum(
            observation["nodes"] * self.norm_factor
        )
        self.env_stats["extra_capacity"] = (
            round(
                1
                - (
                    self.env_stats["tasks_total_cost"]
                    / self.env_stats["nodes_total_capacity"]
                ),
                2,
            )
            * 100
        )
        self.reward_unit = self._reward_unit()
        self.ep_len_norm_factor = self._episode_length_norm_factor()

        return observation, self.info

    def render(self, mode="human"):
        pass

    def get_env_info(self):
        return self.env_stats

    def close(self):
        pass
