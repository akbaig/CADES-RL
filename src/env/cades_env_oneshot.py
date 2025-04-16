import gymnasium as gym
import numpy as np
from gymnasium import spaces
from enum import Enum
from utils.eval_metrics import ( # Assuming these functions remain relevant for final evaluation
    get_avg_node_occupancy,
    get_avg_active_node_occupancy,
    get_empty_nodes_percentage,
    get_evaluate_message_channel_occupancy
)
from env.extended_states_generator import ExtendedStatesGenerator # Assuming this exists
import copy

class TerminationCause(Enum):
    SUCCESS = (1, "success")
    DUPLICATE_PICK = (2, "duplicate_pick")
    NODE_OVERFLOW = (3, "node_overflow")
    DUPLICATE_CRITICAL_PICK = (4, "duplicate_critical_pick")
    # COMMUNICATION_ABSENCE = (5, "communication_absence")

    def __init__(self, id, description):
        self.id = id
        self.description = description

    def __str__(self):
        return self.description

class CadesOneShotEnv(gym.Env):
    """
    Custom Environment for one-shot task allocation.
    The agent proposes a complete allocation in a single action.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.states_generator = ExtendedStatesGenerator(config, self) # Pass self if needed by generator
        self.norm_factor = None
        self.critical_norm_factor = None

        # Action space: For each task, choose a node.
        # The length of the list is max_num_tasks.
        # Each element's value is between 0 and max_num_nodes-1.
        self.action_space = spaces.MultiDiscrete(
            [config.max_num_nodes] * config.max_num_tasks
        )

        # Observation space remains the same, describing the problem instance.
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
                "communications": spaces.Box(
                    low=0,
                    high=2,
                    shape=(config.max_num_tasks, config.max_num_tasks),
                    dtype=int
                )
            }
        )
        self.initial_problem_state = {} # Stores the raw, unnormalized state
        self.current_observation = {}   # Stores the normalized observation
        self.env_stats = {}

    def _get_task_cost(self, task_index):
        """Gets the actual cost of a task."""
        if self.initial_problem_state and task_index < self.env_stats.get("tasks_len", 0):
             # Use initial_problem_state which holds unnormalized costs
            return self.initial_problem_state["tasks"][task_index]
        return 0 # Or handle error appropriately

    def _get_node_capacity(self, node_index):
        """Gets the actual capacity of a node."""
        if self.initial_problem_state and node_index < self.env_stats.get("nodes_len", 0):
            # Use initial_problem_state which holds unnormalized capacities
            return self.initial_problem_state["nodes"][node_index]
        return 0 # Or handle error appropriately

    def _is_task_critical(self, task_index):
        """Checks if a task is critical based on the initial state."""
        if self.initial_problem_state and task_index < self.env_stats.get("tasks_len", 0):
            # Use initial_problem_state which holds unnormalized mask values
            return self.initial_problem_state["critical_mask"][task_index] > 0
        return False

    def _get_critical_mask_value(self, task_index):
        """Gets the specific critical mask value for replica checking."""
        if self.initial_problem_state and task_index < self.env_stats.get("tasks_len", 0):
            return self.initial_problem_state["critical_mask"][task_index]
        return 0

    def _validate_allocation(self, action):
        """
        Validates the entire allocation proposed by the action.

        Args:
            action (np.ndarray): An array of length max_num_tasks, where action[i]
                                 is the node assigned to task i.

        Returns:
            tuple: (TerminationCause, dict): Validity status and details (e.g., final node loads).
        """
        num_tasks = self.env_stats["tasks_len"]
        num_nodes = self.env_stats["nodes_len"]
        node_loads = {i: 0 for i in range(num_nodes)}
        tasks_per_node = {i: [] for i in range(num_nodes)}
        critical_tasks_per_node = {i: {} for i in range(num_nodes)} # node_idx -> {critical_mask_value: count}

        # --- Basic Action Validity ---
        if len(action) != self.config.max_num_tasks:
            print(f"Warning: Action length {len(action)} doesn't match max_num_tasks {self.config.max_num_tasks}")

        # --- Simulate Allocation and Check Constraints ---
        for task_idx in range(num_tasks): # Only iterate over *actual* tasks
            node_idx = action[task_idx]

            # Check if node index is valid
            if not (0 <= node_idx < num_nodes):
                #  print(f"Warning: Task {task_idx} assigned to invalid node {node_idx}. Treating as overflow.")
                 # Treat as instant failure.
                 return TerminationCause.NODE_OVERFLOW, {"node_loads": node_loads, "tasks_per_node": tasks_per_node}

            task_cost = self._get_task_cost(task_idx)
            if task_cost == 0: # Skip tasks with zero cost (if they exist in the problem)
                continue

            if self.current_observation["nodes"][node_idx] < task_cost / self.norm_factor:
                # Node is not capable of handling this task
                return TerminationCause.NODE_OVERFLOW, {"node_loads": node_loads, "tasks_per_node": tasks_per_node}

            # Check Critical Task Duplication
            if self._is_task_critical(task_idx):
                mask_value = self._get_critical_mask_value(task_idx)
                if mask_value in critical_tasks_per_node[node_idx]:
                    # Found another replica of the same critical task on this node
                    return TerminationCause.DUPLICATE_CRITICAL_PICK, {"node_loads": node_loads, "tasks_per_node": tasks_per_node}
                else:
                    critical_tasks_per_node[node_idx][mask_value] = 1 # Mark this critical task type as present
                    self.current_observation["critical_mask"][task_idx] = 0 # Mark this task as assigned in observation

            node_loads[node_idx] += task_cost # Update node load
            self.current_observation["tasks"][task_idx] = 0 # Mark task as assigned in observation
            self.current_observation["nodes"][node_idx] -= task_cost / self.norm_factor # Update normalized observation
            tasks_per_node[node_idx].append(task_idx)

            # Check for assignment of communications
            comms = self.current_observation["communications"]
            receiver_indices = np.where(comms[task_idx] > 0)[0] # task_idx is the sender
            sender_indices = np.where(comms[:, task_idx] > 0)[0] # task_idx is the receiver
            comms[task_idx, receiver_indices] -= 1 # Mark sender as assigned
            comms[sender_indices, task_idx] -= 1 # Mark receiver as assigneds
            # Count incomplete intranode communications (for reward calculation)
            self.env_stats["incomplete_intranode_comms_len"] += len(sender_indices) + len(receiver_indices)
            # Count completed intranode communications (for metrics)
            completed_receivers = np.where(comms[task_idx, receiver_indices] == 0)[0]
            completed_senders = np.where(comms[sender_indices, task_idx] == 0)[0]
            self.env_stats["intranode_comms_len"] += len(completed_receivers) + len(completed_senders)

        # If all checks passed
        return TerminationCause.SUCCESS, {"node_loads": node_loads, "tasks_per_node": tasks_per_node}

    def _calculate_metrics(self, validity, allocation_details):

        node_loads = allocation_details["node_loads"]
        tasks_per_node = allocation_details["tasks_per_node"]
        num_nodes = self.env_stats["nodes_len"]

        initial_capacities = np.array([self._get_node_capacity(i) for i in range(num_nodes)])
        final_loads = np.array([node_loads[i] for i in range(num_nodes)])
        remaining_capacities = initial_capacities - final_loads

        metrics = {
            "avg_node_occupancy": get_avg_node_occupancy(initial_capacities, remaining_capacities),
            "avg_active_node_occupancy": get_avg_active_node_occupancy(initial_capacities, remaining_capacities),
            "empty_nodes": get_empty_nodes_percentage(list(tasks_per_node.values())),
            "message_channel_occupancy": get_evaluate_message_channel_occupancy(
                self.env_stats["comms_len"], self.env_stats["intranode_comms_len"]
            ),
        }
        
        return metrics

    def _calculate_reward(self, metrics):
        """
        Calculates the reward based on the validity and quality of the allocation.
        """

        reward = 0
        reward += self.config.NODE_OCCUPANCY_reward * (metrics["avg_active_node_occupancy"] / 100.0)
        reward += self.config.MESSAGE_CHANNEL_OCCUPANCY_reward * \
            (self.env_stats["incomplete_intranode_comms_len"] / (self.env_stats["comms_len"] * 2))
        reward += self.config.CRITICAL_TASK_OCCUPANCY_reward * \
            (np.count_nonzero(self.current_observation["critical_mask"][:self.env_stats["tasks_len"]]) / self.env_stats["critical_len"])
        return reward

    def step(self, action, training = True):
        """
        Processes the one-shot allocation action.
        Validates the allocation, calculates reward, and terminates.
        """
        # Ensure action is a numpy array
        action = np.array(action)

        # 1. Validate the proposed allocation
        validity, allocation_details = self._validate_allocation(action)
        allocation_details["action"] = action # Add action to details for reward calc
        metrics = self._calculate_metrics(validity, allocation_details)
        # 2. Calculate the reward based on validation and quality metrics
        reward = self._calculate_reward(metrics)

        # 3. Set done flag - episode always ends after one step
        done = True
        truncated = False # Typically False in one-shot unless there's a time limit outside the env

        # 4. Populate info dictionary
        info = {
            "termination_cause": str(validity),
            "final_node_loads": allocation_details.get("node_loads"),
            "tasks_per_node": allocation_details.get("tasks_per_node"),
            "is_success": validity == TerminationCause.SUCCESS,
            # Add other final metrics if desired
            "avg_node_occupancy": metrics.get("avg_node_occupancy"),
            "avg_active_node_occupancy": metrics.get("avg_active_node_occupancy"),
            "empty_nodes": metrics.get("empty_nodes"),
            "message_channel_occupancy": metrics.get("message_channel_occupancy"), # Calculated inside _calculate_reward
        }

        observation = self.current_observation

        # Return observation (initial state), reward, done, truncated, info
        return observation, reward, done, truncated, info

    def generate_states(self, training=True):
        """Generates states using the provided generator."""
        # Ensure the generator API matches what's needed
        (
            tasks, num_tasks, nodes, num_nodes
        ) = self.states_generator.generate_tasks_and_nodes()
        critical_mask = self.states_generator.generate_critical_tasks_and_replicas(
            tasks, num_tasks
        )
        use_graph = (not training) or self.config.use_comm_graph_in_train
        comms, num_comms = self.states_generator.generate_communications(
            tasks, num_tasks, critical_mask, graph=use_graph
        )
        comms[comms == 1] = 2 # Convert 1s to 2s for communication links
        generated_states = {
            "tasks": tasks, # Should be padded to max_num_tasks
            "num_tasks": num_tasks,
            "critical_mask": critical_mask, # Should be padded to max_num_tasks
            "nodes": nodes, # Should be padded to max_num_nodes
            "num_nodes": num_nodes,
            "communications": comms, # Should be padded to max_num_tasks x max_num_tasks
            "num_communications": num_comms,
        }
        return generated_states

    def set_states_random_seed(self, seed):
        """Sets the seed for the state generator."""
        # Assuming the generator has a method to set its seed
        if hasattr(self.states_generator, 'seed'):
             self.states_generator.seed(seed)
        # Also seed the gym environment's base random number generator if needed elsewhere
        super().reset(seed=seed)


    def reset(self, states=None, training=True, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed) # Seeds the internal RNG

        # Generate new problem state if not provided
        if states is None:
            # Note: The states_generator might depend on the env's RNG, seeded above
            problem_state = self.generate_states(training)
        else:
            problem_state = states # Use provided state (ensure format matches generated)

        # Store the raw (unnormalized) state for internal calculations
        self.initial_problem_state = copy.deepcopy(problem_state)

        # --- Update Environment Stats ---
        self.env_stats["tasks_len"] = problem_state["num_tasks"]
        self.env_stats["nodes_len"] = problem_state["num_nodes"]
        self.env_stats["comms_len"] = problem_state["num_communications"]
        self.env_stats["incomplete_intranode_comms_len"] = 0
        self.env_stats["intranode_comms_len"] = 0
        self.env_stats["critical_len"] = np.count_nonzero(problem_state["critical_mask"][:problem_state["num_tasks"]]) # Count only actual tasks
        self.env_stats["tasks_total_cost"] = np.sum(problem_state["tasks"][:problem_state["num_tasks"]])
        self.env_stats["nodes_total_capacity"] = np.sum(problem_state["nodes"][:problem_state["num_nodes"]])
        if self.env_stats["nodes_total_capacity"] > 0:
             self.env_stats["extra_capacity"] = round(1 - (self.env_stats["tasks_total_cost"] / self.env_stats["nodes_total_capacity"]), 2) * 100
        else:
             self.env_stats["extra_capacity"] = 0 if self.env_stats["tasks_total_cost"] == 0 else -np.inf


        # --- Normalize State for Observation ---
        # Find normalization factors based on *actual* max values in the instance or config maxes
        self.norm_factor = np.max(problem_state["nodes"]) if np.max(problem_state["nodes"]) > 0 else 1.0
        self.critical_norm_factor = np.max(problem_state["critical_mask"]) if np.max(problem_state["critical_mask"]) > 0 else 1.0

        observation = {
            "tasks": np.array(problem_state["tasks"] / self.norm_factor),
            "nodes": np.array(problem_state["nodes"] / self.norm_factor),
            "critical_mask": np.array(problem_state["critical_mask"] / self.critical_norm_factor),
            "communications": np.array(problem_state["communications"]),
        }
        self.current_observation = observation
        # Initial info dictionary
        info = {"env_stats": self.env_stats}

        return observation, info

    def render(self, mode="human"):
        # Basic rendering showing the initial problem state
        if mode == 'human':
            print("-" * 20)
            print("Problem State:")
            print(f"  Tasks (costs): {self.initial_problem_state['tasks'][:self.env_stats['tasks_len']]}")
            print(f"  Nodes (caps):  {self.initial_problem_state['nodes'][:self.env_stats['nodes_len']]}")
            print(f"  Critical Mask: {self.initial_problem_state['critical_mask'][:self.env_stats['tasks_len']]}")
            # Add communication matrix printout if desired (can be large)
            print(f"  Num Tasks: {self.env_stats['tasks_len']}, Num Nodes: {self.env_stats['nodes_len']}")
            print(f"  Total Task Cost: {self.env_stats['tasks_total_cost']}")
            print(f"  Total Node Cap:  {self.env_stats['nodes_total_capacity']}")
            print(f"  Comms Links: {self.env_stats['comms_len']}")
            print("-" * 20)
        else:
            super().render(mode=mode) # Or raise NotImplementedError

    def get_env_info(self):
        return self.env_stats

    def close(self):
        # Clean up any resources if needed
        pass