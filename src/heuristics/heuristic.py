# Abstract class for heuristics
from abc import ABC, abstractmethod
import numpy as np

class Heuristic(ABC):
    def __init__(self, env):
        self.env = env
        self.current_task_idx = None

    @abstractmethod
    def set_state(self, state):
        """
        Abstract method to be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, observation):
        """
        Abstract method to be implemented by subclasses.
        """
        pass

    def precompute_communications(self, observation):
        """
        Precomputes the communication requirements for each task.
        """
        self.communications = {}
        for task_idx in range(len(observation["tasks"])):
            # Get the task senders and receivers
            task_senders = self.env._get_task_senders(task_idx)
            task_receivers = self.env._get_task_receivers(task_idx)
            self.communications[task_idx] = (task_senders, task_receivers)

    def check_reset_state(self):
       if self.env.info["episode_len"] == 0:
            self.set_state(self.env.current_state)
            self.precompute_communications(self.env.current_state)

    def _is_task_critical(self, critical_mask, task_index):
        """
        Checks if a task is critical based on the critical mask.
        """
        return critical_mask[task_index] > 0

    def _is_critical_task_duplicated(self, task_index, node_index):
        """
        Checks if a critical task has already been assigned to the node.
        """
        critical_mask = self.env.initial_state["critical_mask"]
        # Get indices of tasks with the same critical mask value
        replica_indices = np.where(critical_mask == critical_mask[task_index])[0]
        # Check if any of these tasks are already assigned to the node
        return np.intersect1d(replica_indices, self.env.assignment_status[node_index]).size > 0
    
    def _is_node_communication_compatible(self, observation, task_index, node_index):
        """
        Checks if a task is communicative.
        """
        (task_senders, task_receivers) = self.communications[task_index]
        if len(task_receivers) == 0 and len(task_senders) == 0:
            # Task is not communicative
            return True
        else:
            # Task is communicative, ensure that all senders and receivers are assigned to the same node
            # Check if node has any existing senders or receivers
            existence_check = False
            if len(task_receivers) > 0:
                allocation_receivers = self.env._get_tasks_placed_in_node(task_receivers, node_index)
                if len(allocation_receivers) > 0:
                    existence_check = True
            if existence_check is False and len(task_senders) > 0:
                allocation_senders = self.env._get_tasks_placed_in_node(task_senders, node_index)
                if len(allocation_senders) > 0:
                    existence_check = True
            # If the node has any existing senders or receivers, return True (as node suitability has been checked before)
            if existence_check is True:
                return True
            else:
                # Check if node has enough remaining capacity to accommodate all senders and receivers
                node_capacity = observation["nodes"][node_index]
                total_senders_cost = np.sum(observation["tasks"][task_senders])
                total_receivers_cost = np.sum(observation["tasks"][task_receivers])
                return node_capacity >= total_senders_cost + total_receivers_cost