import numpy as np
from .heuristic import Heuristic

class FirstFitHeuristic(Heuristic):
    def __init__(self, env):
        super().__init__(env)

    def set_state(self, state):
        self.unassigned_tasks = set(range(len(state["tasks"])))

    def predict(self, observation):
        self.check_reset_state()

        # Update unassigned tasks
        self.unassigned_tasks = set(np.where(observation["tasks"] > 0)[0])
        
        # If all tasks are assigned, return None
        if not self.unassigned_tasks:
            return np.array([None, None]), None

        # Get the next task to assign
        current_task_idx = min(self.unassigned_tasks)
        selected_task_cost = observation["tasks"][current_task_idx]

        nodes = observation["nodes"]
        num_nodes = len(nodes)
        current_node_idx = 0
        skip_communication_check = False
        raise_error = False
        
        # Find the first node that can accommodate the task
        while True:
            node_capacity = nodes[current_node_idx]
            if node_capacity >= selected_task_cost:
                if self._is_task_critical(observation["critical_mask"], current_task_idx):
                    if self._is_critical_task_duplicated(current_task_idx, current_node_idx):
                        pass  # Skip nodes where critical task duplication occurs
                    else:
                        if not skip_communication_check and not self._is_node_communication_compatible(observation, current_task_idx, current_node_idx):
                            pass
                        else:
                            self.unassigned_tasks.remove(current_task_idx)
                            action = np.array([current_task_idx, current_node_idx])
                            return action, None
                else:
                    self.unassigned_tasks.remove(current_task_idx)
                    action = np.array([current_task_idx, current_node_idx])
                    return action, None
            
            # Move to the next node
            current_node_idx = (current_node_idx + 1) % num_nodes
            if current_node_idx == 0:
                # We've looped through all nodes and found no fit
                skip_communication_check = True
                if raise_error:
                    break

        # If no node can accommodate the task
        raise ValueError("No node can accommodate the task")
