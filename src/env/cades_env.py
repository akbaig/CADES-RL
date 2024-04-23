import gym
import random
import numpy as np
from gym import spaces
from enum import Enum
from utils.eval_metrics import get_avg_node_occupancy, get_empty_nodes_percentage, get_evaluate_message_channel_occupancy
from env.states_generator import StatesGenerator
import copy


class TerminationCause(Enum):
    SUCCESS = 1
    DUBLICATE_PICK = 2
    NODE_OVERFLOW = 3
    DUPLICATE_CRITICAL_PICK = 4


class CadesEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        """
        Initializes observation and action spaces
        """
        super().__init__()

        self.config = config
        self.states_generator = StatesGenerator(config)
        self.norm_factor = None

        self.action_space = spaces.MultiDiscrete(
            [config.max_num_tasks, config.total_nodes]
        )

        self.observation_space = spaces.Dict(
            {
                "tasks": spaces.Box(
                    low=0, high=1, shape=(config.max_num_tasks,), dtype=np.float
                ),
                "critical_mask": spaces.MultiDiscrete(
                    [2 + config.number_of_critical_tasks]
                    * np.prod((config.max_num_tasks,))
                ),
                "nodes": spaces.Box(
                    low=0, high=1, shape=(config.total_nodes,), dtype=np.float
                ),
                "communications": spaces.Box(
                    low=0,
                    high=1,
                    shape=(config.max_num_tasks, config.max_num_tasks),
                    dtype=np.uint8,
                ),
            }
        )

        self.env_stats = {}
        self.assignment_status = []
        for i in range(config.total_nodes):
            self.assignment_status.append([])
        self.current_state = {}
        self.total_reward = 0
        self.done = False

    def _is_task_critical(self, task_index):
        """
        Checks if a task is critical or not i.e. has mask value greater than one
        """
        return self.current_state["critical_mask"][task_index] > 1

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

    def _get_valid_action(self, action):
        """
        Prevents the agent from selecting previously chosen indices by masking invalid choices,
        ensuring the agent selects only valid actions.
        """
        task_idx, node_idx = action
        if self.current_state["tasks"][task_idx] > 0:
            return action  # The action is already valid
        valid_task_indices = [
            idx for idx, task in enumerate(self.current_state["tasks"]) if task > 0
        ]
        if not valid_task_indices:  # No valid task left
            raise ValueError("No valid action found")
        else:  # Choose a new task from the valid tasks
            new_task_idx = random.choice(valid_task_indices)
        return [new_task_idx, node_idx]
    
    def _exponential_decay_reward(self, step, max_steps, max_reward, k=2):
        """
        An adjusted exponential decay formula to reach exactly 0 at the final step
        Where k is an exponent that determines how quickly the function approaches zero. 
        Higher values of k will make the decay steeper towards the end.
        """
        reward = max_reward * (1 - (step / max_steps) ** k)
        return reward

    def _reward(self, action):
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
            max_steps = self.config.max_num_tasks
            max_reward = self.config.DUPLICATE_PICK_reward
            reward = self._exponential_decay_reward(step, max_steps, max_reward)
            reward_type = f"Duplicate Pick Reward on Step {step}: {reward}"
            # Select any other valid action
            new_action = self._get_valid_action(action)
            _, done = self._reward(new_action)

        # Agent picked the node which is already full
        elif selected_task_cost > self.current_state["nodes"][selected_node_idx]:
            reward = self.config.NODE_OVERFLOW_reward * self.info["episode_len"] * 0.25
            reward_type = f"Node Overflow Reward: {reward}"
            done = True
            self.info["termination_cause"] = TerminationCause.NODE_OVERFLOW.name

        # Agent picked the node which already had critical task
        elif self._is_task_critical(
            selected_task_idx
        ) and self._is_critical_task_duplicated(selected_task_idx, selected_node_idx):
            reward = (
                self.config.DUPLICATE_CRITICAL_PICK_reward
                * self.info["episode_len"]
                * 0.15
            )
            reward_type = f"Duplicate Critical Pick Reward: {reward}"
            done = True
            self.info["termination_cause"] = (
                TerminationCause.DUPLICATE_CRITICAL_PICK.name
            )

        # Agent picked the correct task and node
        else:
            # Assign Rewards
            reward = self.config.STEP_reward
            reward += self.config.BONUS_reward * self.info["episode_len"]
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
                self.info["termination_cause"] = TerminationCause.SUCCESS.name
                self.info["is_success"] = True
                done = True

        return reward, done

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

    def step(self, action):
        """
        Advances the episode by one timestep using the given action. 
        """
        # Calc Rewards
        reward, done = self._reward(action)
        
        # Save Info about Episode
        self.info["commnication_status"] = self.communication_status
        self.info["assignment_status"] = self.assignment_status
        self.env_stats["intranode_comms_len"] = len(self.communication_status)
        self.total_reward = self.total_reward + reward

        # Calculate Evaluation Metrics
        self.info["avg_node_occupancy"] = get_avg_node_occupancy(
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
        # If verbose is toggled, print info about timestep
        if done is True and self.config.verbose is True and self.info["termination_cause"] is TerminationCause.SUCCESS.name:
            self._verbose(action, reward)

        observation = self.current_state
        return observation, reward, done, self.info # observation, reward, done, extra_info
    
    def generate_states(self):
        """
        Generates states for the environment
        """
        (
            states,
            states_lens,
            states_mask,
            nodes_available,
        ) = self.states_generator.generate_states_batch()
        (states, critical_mask, _) = self.states_generator.generate_critical_tasks(
            states, states_mask, states_lens
        )
        (communications, communications_lens) = (
            self.states_generator.generate_communications(
                states, critical_mask, states_lens
            )
        )
        generated_states = {
            "tasks": states,
            "tasks_lens": states_lens,
            "tasks_mask": states_mask,
            "critical_mask": critical_mask,
            "nodes": nodes_available,
            "communications": communications,
            "communications_lens": communications_lens,
        }
        return generated_states

    def reset(self, states=None):
        """
        Initializes new states for the start of a new episode.
        """
        # assignment status is an variable-sized 2D Array, having dimensions total_nodes x (size of node)
        # it stores the indices of task assignment on the nodes
        self.assignment_status = []
        self.communication_status = set()
        for i in range(self.config.total_nodes):
            self.assignment_status.append([])

        self.info = {"is_success": False, "episode_len": 0, "termination_cause": None}
        self.done = False
        if states is None:
            states = self.generate_states()
        # norm factor is the largest node size of first batch in nodes
        self.norm_factor = max(list(states["nodes"][0]))
        # use first batch of states and nodes_available and normalize the values, this is our observation now
        observation = {
            "tasks": np.array(list(states["tasks"][0]) / self.norm_factor),
            "critical_mask": np.array(states["critical_mask"][0]),
            "nodes": np.array(list(states["nodes"][0]) / self.norm_factor),
            "communications": np.array(states["communications"][0]),
        }
        self.initial_state = copy.deepcopy(observation)
        self.current_state = observation
        self.env_stats["comms_len"] = states["communications_lens"][0]
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

        return observation

    def render(self, mode="human"):
        pass

    def get_env_info(self):
        return self.env_stats

    def close(self):
        pass
