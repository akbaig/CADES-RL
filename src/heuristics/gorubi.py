import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, Tuple, List, Set, Any # Added for type hinting
from .heuristic import Heuristic

class GurobiHeuristic(Heuristic):
    def __init__(self, env, verbose_gurobi=False):
        super().__init__(env)
        self.optimal_assignment = None # Stores result from Gurobi {node: [tasks]}
        self.assignment_plan = []      # Stores sequence of (task, node) actions
        self.verbose_gurobi = verbose_gurobi
        self._initial_state_processed = False # Track if Gurobi has been run for the episode

    def set_state(self, initial_state):
        """
        Solves the entire problem using Gurobi based on the initial state.
        """
        print("GurobiHeuristic: set_state called. Solving problem instance...")
        # Ensure we are using the unnormalized, initial data
        # Access initial state directly if env structure provides it,
        # otherwise, we might need to reconstruct from the passed state (if normalized)
        
        # Assuming self.env holds the necessary structures like initial_state
        # and normalization factors after env.reset()
        try:
            tasks_unnormalized = self.env.initial_state['tasks'] * self.env.norm_factor
            nodes_unnormalized = self.env.initial_state['nodes'] * self.env.norm_factor
            comms_matrix = self.env.initial_state['communications'] # Assume already correct format
            critical_mask = self.env.initial_state['critical_mask'] * self.env.critical_norm_factor
            
            # Add fallback if norm_factor is not present or 1
            if not hasattr(self.env, 'norm_factor') or self.env.norm_factor is None:
                 print("Warning: norm_factor not found on env. Assuming data is unnormalized.")
                 tasks_unnormalized = self.env.initial_state['tasks']
                 nodes_unnormalized = self.env.initial_state['nodes']

            if not hasattr(self.env, 'critical_norm_factor') or self.env.critical_norm_factor is None:
                 print("Warning: critical_norm_factor not found on env. Assuming data is unnormalized.")
                 critical_mask = self.env.initial_state['critical_mask']

        except AttributeError as e:
            print(f"Error accessing environment data needed for Gurobi: {e}")
            print("Ensure env has initial_state and norm_factor attributes after reset.")
            self.optimal_assignment = {}
            self.assignment_plan = []
            return
        except KeyError as e:
             print(f"Error accessing key in initial_state: {e}")
             self.optimal_assignment = {}
             self.assignment_plan = []
             return


        # Call the Gurobi solver function
        try:
             self.optimal_assignment, objective = solve_with_gurobi(
                tasks_data=tasks_unnormalized,
                nodes_data=nodes_unnormalized,
                comms_data=comms_matrix,
                critical_mask_data=critical_mask,
                verbose=self.verbose_gurobi
             )
        except gp.GurobiError as e:
             print(f"Gurobi Error during solve: {e.errno} - {e}")
             self.optimal_assignment = {}
             objective = -1
        except Exception as e:
             print(f"Unexpected Error during Gurobi solve: {e}")
             self.optimal_assignment = {}
             objective = -1


        # Convert the optimal assignment into a sequential plan
        self.assignment_plan = []
        if self.optimal_assignment and objective >= 0: # Check if solver succeeded
            # Create pairs (task_idx, node_idx)
            task_node_pairs = []
            for node_idx, assigned_tasks in self.optimal_assignment.items():
                for task_idx in assigned_tasks:
                    task_node_pairs.append((task_idx, node_idx))

            # Sort the plan, e.g., by task size descending (like FFD)
            # This defines the order the actions will be returned by predict()
            task_node_pairs.sort(key=lambda pair: tasks_unnormalized[pair[0]], reverse=True)
            
            self.assignment_plan = task_node_pairs
            print(f"Gurobi plan created with {len(self.assignment_plan)} steps.")
        else:
            print("Gurobi did not find an optimal solution or failed. No plan created.")
            self.assignment_plan = [] # Ensure plan is empty on failure

    def predict(self, observation):
        """
        Returns the next action from the pre-computed Gurobi plan.
        """
        # Ensure Gurobi runs on the first step by calling check_reset_state
        # check_reset_state now calls set_state internally if needed
        self.check_reset_state() 

        if not self.assignment_plan:
            # No plan available (Gurobi failed or plan is finished)
            print("GurobiHeuristic: No assignment plan available or plan finished.")
            # Return an invalid action or handle as per environment requirements
            # Returning [0, 0] might lead to immediate failure if task 0 is invalid/taken
            # It might be better to return None or raise an error depending on how
            # the evaluation loop handles it. Let's return an invalid index pair.
            num_tasks = len(observation['tasks'])
            num_nodes = len(observation['nodes'])
            # Return indices guaranteed to be out of bounds (or padding) if possible
            # Or just a fixed invalid marker like [-1,-1] if the caller checks
            return np.array([-1, -1]), None # Indicate failure/end of plan


        # Get the next action from the plan
        next_action_tuple = self.assignment_plan.pop(0) # Get and remove first element
        action = np.array(next_action_tuple)

        # print(f"GurobiHeuristic: Predicting action {action}") # Optional debug print
        
        # The second return value (state for recurrent policies) is None for heuristics
        return action, None

def solve_with_gurobi(
    tasks_data: np.ndarray,
    nodes_data: np.ndarray,
    comms_data: np.ndarray,
    critical_mask_data: np.ndarray,
    verbose: bool = False
) -> Tuple[Dict[int, List[int]], float]:
    """
    Solves the task allocation problem using Gurobi.

    Args:
        tasks_data: Array of task sizes (original, not normalized).
                      Includes padding (size 0 for invalid tasks).
        nodes_data: Array of node capacities (original, not normalized).
                     Includes padding (capacity 0 for invalid nodes).
        comms_data: Adjacency matrix (NxN) where comms_data[i, j] = 1
                    if task i sends to task j.
        critical_mask_data: Array where non-zero equal values indicate
                            replicas of the same critical task.
        verbose: If True, Gurobi output is enabled.

    Returns:
        A tuple containing:
        - assignment: A dictionary mapping node index to a list of assigned task indices.
        - objective_value: The value of the objective function (number of satisfied intra-node comms).
        Returns ({}, -1.0) if no optimal solution is found.
    """
    num_all_tasks = len(tasks_data)
    num_all_nodes = len(nodes_data)

    # Identify valid tasks and nodes (non-zero size/capacity)
    valid_tasks = [i for i, size in enumerate(tasks_data) if size > 0]
    valid_nodes = [j for j, cap in enumerate(nodes_data) if cap > 0]

    if not valid_tasks:
        print("No valid tasks to assign.")
        return {}, 0.0

    if not valid_nodes:
        print("No valid nodes available.")
        return {}, -1.0 # Cannot assign tasks

    # --- Create Gurobi Model ---
    env_gurobi = gp.Env(empty=True)
    env_gurobi.setParam('OutputFlag', 1 if verbose else 0)
    env_gurobi.start()
    model = gp.Model("CadesTaskAllocation", env=env_gurobi)

    # --- Define Decision Variables ---
    # x[i, j] = 1 if task i is assigned to node j, 0 otherwise
    x = model.addVars(valid_tasks, valid_nodes, vtype=GRB.BINARY, name="x")

    # --- Define Constraints ---

    # 1. Task Assignment: Each valid task must be assigned to exactly one valid node
    model.addConstrs((x.sum(i, '*') == 1 for i in valid_tasks), name="TaskAssign")

    # 2. Node Capacity: Total size of tasks on a node <= node capacity
    model.addConstrs(
        (gp.quicksum(tasks_data[i] * x[i, j] for i in valid_tasks) <= nodes_data[j]
         for j in valid_nodes), name="NodeCapacity"
    )

    # 3. Critical Task Replication Constraint: Replicas cannot be on the same node
    critical_groups: Dict[int, List[int]] = {}
    for i in valid_tasks:
        mask_val = int(critical_mask_data[i])
        if mask_val > 0:
            if mask_val not in critical_groups:
                critical_groups[mask_val] = []
            critical_groups[mask_val].append(i)

    for group_id, tasks_in_group in critical_groups.items():
        if len(tasks_in_group) > 1: # Only add constraint if there are actual replicas
            for j in valid_nodes:
                 # Sum of assignments for this critical group to node j must be <= 1
                model.addConstr(
                    gp.quicksum(x[i, j] for i in tasks_in_group) <= 1,
                    name=f"CriticalGroup_{group_id}_Node_{j}"
                )

    # --- Define Objective Function ---
    # Maximize the number of communication links satisfied within the same node.
    # We use auxiliary variables for the quadratic terms x[s,j]*x[r,j].
    
    # Identify communicating pairs (s, r) where comms_data[s, r] == 1
    comm_links = []
    for s in valid_tasks:
        for r in valid_tasks:
            if s != r and comms_data[s, r] == 1:
                 # Check if BOTH s and r are valid tasks before adding link
                 # (already ensured by iterating over valid_tasks, but good practice)
                comm_links.append((s, r))
                
    if not comm_links:
         # If no communication links exist, the objective is trivial (0)
         # We still need to find a feasible assignment. Gurobi handles this.
         print("No communication links detected in the problem instance.")
         model.setObjective(0, GRB.MAXIMIZE) # Dummy objective
    else:
        # y[s, r, j] = 1 if task s and task r (communicating pair) are both on node j
        y = model.addVars(comm_links, valid_nodes, vtype=GRB.BINARY, name="comm_active")

        # Link y variables to x variables: y[s,r,j] = x[s,j] * x[r,j]
        # y <= x[s]
        model.addConstrs((y[s, r, j] <= x[s, j] for s, r in comm_links for j in valid_nodes), name="link_y_x_s")
        # y <= x[r]
        model.addConstrs((y[s, r, j] <= x[r, j] for s, r in comm_links for j in valid_nodes), name="link_y_x_r")
        # y >= x[s] + x[r] - 1  (ensures y=1 if both x[s]=1 and x[r]=1)
        # This constraint is actually not strictly needed for a maximization objective
        # because the objective pushes y to 1 when possible.
        # Including it makes the formulation tighter but adds constraints. Optional.
        # model.addConstrs((y[s, r, j] >= x[s, j] + x[r, j] - 1 for s, r in comm_links for j in valid_nodes), name="link_y_x_both")

        # Objective: Maximize sum of y[s, r, j] over all links (s,r) and nodes j
        # This counts each satisfied intra-node communication link.
        model.setObjective(y.sum(), GRB.MAXIMIZE)


    # --- Optimize ---
    print("Starting Gurobi optimization...")
    model.optimize()

    # --- Extract Solution ---
    assignment: Dict[int, List[int]] = {j: [] for j in valid_nodes}
    final_objective = -1.0

    if model.Status == GRB.OPTIMAL:
        print("Optimal solution found.")
        final_objective = model.ObjVal
        solution = model.getAttr('X', x)
        for i in valid_tasks:
            for j in valid_nodes:
                if solution[i, j] > 0.5: # Check if task i is assigned to node j
                    assignment[j].append(i)
                    break # Task assigned to only one node
    elif model.Status == GRB.INFEASIBLE:
        print("Model is infeasible. No solution exists under the given constraints.")
        # You might want to compute an IIS (Irreducible Inconsistent Subsystem)
        # to debug why it's infeasible:
        # model.computeIIS()
        # model.write("model_iis.ilp")
    elif model.Status == GRB.UNBOUNDED:
         print("Model is unbounded.") # Should not happen for this type of problem
    else:
        print(f"Optimization finished with status: {model.Status}")
        # Potentially retrieve best found solution if status is TIME_LIMIT, etc.
        if model.SolCount > 0:
             print("Suboptimal solution might be available.")
             final_objective = model.ObjVal # Get objective of best found solution
             solution = model.getAttr('X', x)
             for i in valid_tasks:
                for j in valid_nodes:
                    if solution[i, j] > 0.5:
                        assignment[j].append(i)
                        break
        else:
             print("No solution found.")


    print(f"Objective value (satisfied intra-node comms): {final_objective}")
    print("Assignment (Node -> [Tasks]):")
    # Filter out empty nodes in the printout for clarity
    print({k: v for k, v in assignment.items() if v}) 

    return assignment, final_objective

