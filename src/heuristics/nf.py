import numpy as np
from .heuristic import Heuristic

class NextFitHeuristic(Heuristic):
    def __init__(self, env):
        super().__init__(env)
    
    def set_state(self, state):
        super().set_state(state)
        tasks = state["tasks"]
        task_indices = np.where(tasks > 0)[0]
        self.unassigned_tasks = task_indices.tolist()
        self.current_node_idx = 0  # Start with the first node

    def predict(self, observation):
        self.check_reset_state()
        
        # If all tasks are assigned, return None
        if not self.unassigned_tasks:
            return np.array([None, None]), None

        # Get the next task to assign
        current_task_idx = self.unassigned_tasks[0]
        selected_task_cost = observation["tasks"][current_task_idx]

        nodes = observation["nodes"]
        num_nodes = len(nodes)
        current_node_idx = self.current_node_idx
        skip_communication_check = False
        skip_critical_check = False
        skip_overflow_check = False

        while True:
            node_capacity = nodes[current_node_idx]
            if node_capacity >= selected_task_cost or skip_overflow_check:
                is_critical = self._is_task_critical(observation["critical_mask"], current_task_idx)
                is_critical_duplicated = self._is_critical_task_duplicated(current_task_idx, current_node_idx)
                if not is_critical or (is_critical and not is_critical_duplicated) or skip_critical_check:
                    is_node_comm_compat = self._is_node_communication_compatible(observation, current_task_idx, current_node_idx)
                    if is_node_comm_compat or skip_communication_check:
                        self.unassigned_tasks.pop(0)
                        action = np.array([current_task_idx, current_node_idx])
                        self.current_node_idx = current_node_idx
                        return action, None
            
            # Move to the next node
            current_node_idx = (current_node_idx + 1) % num_nodes
            # We've looped through all nodes and found no fit
            if current_node_idx == 0:
                # In the first iteration, skip communication check
                if skip_communication_check == False:
                    skip_communication_check = True
                # In the second iteration, skip critical check
                elif skip_critical_check == False:
                    skip_critical_check = True
                # In the third iteration, skip overflow check
                elif skip_overflow_check == False:
                    skip_overflow_check = True
                # In the fourth iteration, we've tried all possible combinations, raise an error
                else:
                    # If no node can accommodate the task
                    raise ValueError("No node can accommodate the task")
                # We jump to the start node again as in next fit, the already closed bins are not considered
                current_node_idx = self.current_node_idx
                
