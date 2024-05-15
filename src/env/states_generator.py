import numpy as np

class StatesGenerator():
    """
    Helper class used to randomly generate batches of states given a set
    of problem conditions, which are provided via the `config` object.
    """

    def __init__(self, config):
        self.min_num_tasks = config.min_num_tasks
        self.max_num_tasks = config.max_num_tasks
        self.min_task_size = config.min_task_size
        self.max_task_size = config.max_task_size

        self.min_node_size = config.min_node_size
        self.max_node_size = config.max_node_size
        self.min_num_nodes = config.min_num_nodes
        self.max_num_nodes = config.max_num_nodes
        self.num_critical_tasks = config.number_of_critical_tasks
        self.num_replicas = config.number_of_replicas

    def generate_tasks_and_nodes(self):
        # Tasks
        num_tasks = np.random.randint(self.min_num_tasks, self.max_num_tasks + 1)
        tasks = np.random.randint(self.min_task_size, self.max_task_size + 1, size=self.max_num_tasks)
        tasks[num_tasks:] = 0  # Zero-padding for invalid tasks
        
        # Nodes
        num_nodes = np.random.randint(self.min_num_nodes, self.max_num_nodes + 1)
        nodes = np.random.randint(self.min_node_size, self.max_node_size + 1, size=self.max_num_nodes)
        nodes[num_nodes:] = 0  # Zero-padding for invalid nodes

        return tasks, num_tasks, nodes, num_nodes

    def generate_tasks_and_nodes_batch(self):
        assert("Not implemented")

    def generate_critical_tasks_and_replicas(self, tasks, num_tasks):
        # Get valid tasks where tasks has value > 0
        valid_tasks = np.where(tasks > 0)[0]
        critical_tasks = np.random.choice(valid_tasks, size=self.num_critical_tasks, replace=False)
        remaining_tasks = np.setdiff1d(valid_tasks, critical_tasks)
        critical_mask = np.zeros(num_tasks)
        # For each critical task, add replicas
        for critical_idx, idx in enumerate(critical_tasks):
            # choose candidates for replicas from remaining
            if(len(remaining_tasks) < self.num_replicas):
                assert("Insufficient candidates for replicas")
            replicas = np.random.choice(remaining_tasks, size=self.num_replicas, replace=False)
            # subtract chosen candidates from remaining tasks
            remaining_tasks = np.setdiff1d(remaining_tasks, replicas)
            # assign unique_id in mask to task and its replicas
            critical_mask[idx] = critical_idx + 1
            critical_mask[replicas] = critical_idx + 1
        return critical_mask

    def generate_critical_tasks_and_replicas_batch(self):
        assert("Not implemented")
