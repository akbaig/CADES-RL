import gym
import random
import numpy as np
from gym import spaces
from enum import Enum
from states_generator import StatesGenerator
from config import get_config


class TerminationCause(Enum):
    SUCCESS = 1
    DUBLICATE_PICK = 2
    NODE_OVERFLOW = 3
    DUPLICATE_CRITICAL_PICK = 4


class CadesEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
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
        # if task has mask value greater than one, it's defined as critical task
        return self.current_state["critical_mask"][task_index] > 1

    def _is_critical_task_duplicated(self, task_index, node_index):
        critical_mask = self.current_state["critical_mask"]
        # get indices which have same mask value
        replica_indices = list(np.where(critical_mask == critical_mask[task_index])[0])
        # check if these indices are in the assignment status of selected node
        return (
            np.intersect1d(replica_indices, self.assignment_status[node_index]).size > 0
        )

    def get_task_receivers(self, task_index):
        receivers = self.current_state["communications"][task_index]
        return np.where(receivers == 1)[0]

    def get_task_senders(self, task_index):
        senders = self.current_state["communications"][:, task_index]
        return np.where(senders == 1)[0]

    def get_tasks_placed_in_node(self, list_of_tasks, node_index):
        return np.intersect1d(list_of_tasks, self.assignment_status[node_index])

    # For Preventing the agent from choosing already chosen indices again
    # This method masks the invalid choices
    # And makes the agent choose a valid action instead
    def get_valid_action(self, action):
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

    def _reward(self, action):

        done = False
        # Agent outputs two actions, one for task index and one for node index
        selected_task_idx = action[0]
        selected_node_idx = action[1]
        selected_task_cost = self.current_state["tasks"][selected_task_idx]
        reward_type = ""

        # Agent picked the task which is already used
        if selected_task_cost == 0:
            # Agent receives reward based on its step number. Highest at early steps, lower at later steps.
            reward = self.config.DUPLICATE_PICK_reward - (
                abs(self.config.max_num_tasks - self.info["episode_len"])
                * 3
                / self.config.max_num_tasks
            )
            # Select any other valid action
            new_action = self.get_valid_action(action)
            _, done = self._reward(new_action)

        # Agent picked the node which is already full
        elif selected_task_cost > self.current_state["nodes"][selected_node_idx]:
            reward = self.config.NODE_OVERFLOW_reward * self.info["episode_len"] * 0.25
            reward_type = "NODE Overflow Reward"
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
            reward_type = "Duplicate Critical Pick Reward"
            done = True
            self.info["termination_cause"] = (
                TerminationCause.DUPLICATE_CRITICAL_PICK.name
            )

        # Agent picked the correct task and node
        else:
            # Assign Rewards
            reward = self.config.STEP_reward
            reward += self.config.BONUS_reward * self.info["episode_len"]
            reward_type = "Step and Bonus Reward"
            if self._is_task_critical(selected_task_idx):
                reward += self.config.CRITICAL_reward
                reward_type += ' \nCritical Reward'
            # Check if the item is communicating
            item_receivers = self.get_task_receivers(selected_task_idx)
            item_senders = self.get_task_senders(selected_task_idx)
            # if the item is a sender i.e has receivers
            if(len(item_receivers) > 0): 
                # narrow down the receivers to the ones that are already placed in the node
                allocated_receivers = self.get_tasks_placed_in_node(item_receivers, selected_node_idx)
                reward += (self.config.COMM_reward/self.env_stats["comms_len"]) * len(allocated_receivers)
                reward_type += f' \nCommunication Reward for {selected_task_idx} communicating with {allocated_receivers}'
                if(len(allocated_receivers) > 0):
                    # set the communication mask to zero
                    self.current_state["communications"][selected_task_idx, allocated_receivers] = 0
                    # add pair to communication status
                    for receiver in allocated_receivers:
                        self.communication_status.add((selected_task_idx, receiver))
            if(len(item_senders) > 0):
                # narrow down the senders to the ones that are already placed in the node
                allocated_senders = self.get_tasks_placed_in_node(item_senders, selected_node_idx)
                reward += (self.config.COMM_reward/self.env_stats["comms_len"]) * len(allocated_senders)
                reward_type += f' \nCommunication Reward for {allocated_senders} communicating with {selected_task_idx}'
                if(len(allocated_senders) > 0):
                    # set the communication mask to zero
                    self.current_state["communications"][allocated_senders, selected_task_idx] = 0
                    # add pair to communication status
                    for sender in allocated_senders:
                        self.communication_status.add((sender, selected_task_idx))
            # Set the selected item mask value as zero
            self.current_state["critical_mask"][selected_task_idx] = 0
            # Mark the selected item as zero
            self.current_state["tasks"][selected_task_idx] = 0
            # Consume the space in selected bin
            self.current_state["nodes"][selected_task_idx] -= selected_task_cost
            # Update Assignment status
            self.assignment_status[selected_node_idx].append(selected_task_idx)
            self.info["episode_len"] = self.info["episode_len"] + 1
            # Check if no task is remaining
            if sum(self.current_state["tasks"]) == 0:
                reward += self.config.SUCCESS_reward
                reward_type += " \n Success Reward"
                self.info["termination_cause"] = TerminationCause.SUCCESS.name
                self.info["is_success"] = True
                done = True

        return reward, done

    def step(self, action):
        observation = self.current_state
        reward, done = self._reward(action)
        if done is True:
            print(
                "Observation Space: \nTasks: ",
                self.current_state["tasks"],
                " \nCritical Masks: ",
                self.current_state["critical_mask"],
                " \nNodes:",
                self.current_state["nodes"],
                " \nPossible Communications: ",
                self.env_stats["comms_len"],
            )
            print(
                "Last Action : Selected task: ",
                action[0],
                " Selected node: ",
                action[1],
            )
            print("Accounted Communications: ", len(self.communication_status))
            print(
                "Episode Reward: ",
                reward,
                " Termination Cause: ",
                self.info["termination_cause"],
            )

        self.info["assignment_status"] = self.assignment_status
        self.total_reward = self.total_reward + reward
        return observation, reward, done, self.info

    def reset(self):
        # assignment status is an variable-sized 2D Array, having dimensions total_nodes x (size of node)
        # it stores the indices of task assignment on the nodes
        self.assignment_status = []
        self.communication_status = set()
        for i in range(self.config.total_nodes):
            self.assignment_status.append([])

        self.info = {"is_success": False, "episode_len": 0, "termination_cause": None}
        self.done = False
        self.total_reward = 0
        # states_batch_generator gives the following variables
        # states - 2D Array (Batch Size x num of tasks) - elements contain size of task
        # states_lens - 1D Array (Batch Size) - elements contain num of valid tasks in each row of states variable
        # states_mask - 2D Array (Batch Size x num of tasks) - elements represent a mask of states variable having all 1 values
        # nodes_available - 2D Array (Batch Size x num of nodes) - elements contain size of nodes (imp: all batches contain same values of node sizes)
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
        # norm factor is the largest node size of first batch in nodes_available
        self.norm_factor = max(list(nodes_available[0]))

        # use first batch of states and nodes_available and normalize the values, this is our observation now
        observation = {
            "tasks": np.array(list(states[0]) / self.norm_factor),
            "critical_mask": np.array(critical_mask[0]),
            "nodes": np.array(list(nodes_available[0]) / self.norm_factor),
            "communications": np.array(communications[0]),
        }
        self.current_state = observation
        self.env_stats["comms_len"] = communications_lens[0]
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
