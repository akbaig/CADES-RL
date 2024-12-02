import numpy as np
from .heuristic import Heuristic

class FirstFitDecreasingHeuristic(Heuristic):
    def __init__(self, env):
        super().__init__(env)
        self.set_state(env.current_state)

    def _get_sorted_tasks(self, tasks):
        task_indices = np.where(tasks > 0)[0]
        task_costs = tasks[task_indices]
        sorted_indices = task_indices[np.argsort(-task_costs)]  # Sort in decreasing order
        return sorted_indices.tolist()
    
    def set_state(self, state):
        self.sorted_tasks = self._get_sorted_tasks(state["tasks"])

    def predict(self, observation):
        # Update sorted tasks if any tasks have been assigned
        self.sorted_tasks = [idx for idx in self.sorted_tasks if observation["tasks"][idx] > 0]

        if not self.sorted_tasks:
            return np.array([None, None]), None

        # Get the next task to assign
        current_task_idx = self.sorted_tasks[0]
        selected_task_cost = observation["tasks"][current_task_idx]

        # Find the first node that can accommodate the task
        nodes = observation["nodes"]
        for node_idx, node_capacity in enumerate(nodes):
            if node_capacity >= selected_task_cost:
                if self._is_task_critical(observation["critical_mask"], current_task_idx):
                    if self._is_critical_task_duplicated(current_task_idx, node_idx):
                        continue  # Skip nodes where critical task duplication occurs
                action = np.array([current_task_idx, node_idx])
                return action, None

        # If no node can accommodate the task
        return np.array([current_task_idx, None]), None