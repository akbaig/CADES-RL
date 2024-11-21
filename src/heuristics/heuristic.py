# Abstract class for heuristics
from abc import ABC, abstractmethod
import numpy as np

class Heuristic(ABC):
    def __init__(self, env):
        self.env = env
        self.current_task_idx = None

    @abstractmethod
    def predict(self, observation):
        """
        Abstract method to be implemented by subclasses.
        """
        pass

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