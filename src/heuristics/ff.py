import numpy as np
from .heuristic import Heuristic

class FirstFitHeuristic(Heuristic):
    def __init__(self, env):
        super().__init__(env)
        self.unassigned_tasks = set(range(len(env.current_state["tasks"])))

    def predict(self, observation):
        # Update unassigned tasks
        self.unassigned_tasks = set(np.where(observation["tasks"] > 0)[0])
        
        # If all tasks are assigned, return None
        if not self.unassigned_tasks:
            return np.array([None, None]), None

        # Get the next task to assign
        self.current_task_idx = min(self.unassigned_tasks)
        selected_task_cost = observation["tasks"][self.current_task_idx]
        
        # Find the first node that can accommodate the task
        nodes = observation["nodes"]
        for node_idx, node_capacity in enumerate(nodes):
            if node_capacity >= selected_task_cost:
                if self._is_task_critical(observation["critical_mask"], self.current_task_idx):
                    if self._is_critical_task_duplicated(self.current_task_idx, node_idx):
                        continue  # Skip nodes where critical task duplication occurs
                action = np.array([self.current_task_idx, node_idx])
                return action, None

        # If no node can accommodate the task
        return np.array([self.current_task_idx, None]), None
