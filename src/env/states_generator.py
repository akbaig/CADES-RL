import numpy as np
import random
import torch
from env.comm_tree import CommunicationTree

"""
Logic to generate new states using .
"""


class StatesGenerator(object):
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
        self.total_nodes = config.total_nodes

        self.num_critical_tasks = config.number_of_critical_tasks
        self.num_critical_copies = config.number_of_copies
        self.min_num_comms = config.min_num_comms
        self.max_num_comms = config.max_num_comms
        self.max_comm_chain = config.max_comm_chain
        self.ci_groups = []

    def generate_critical_tasks(self, tasks_seqs_batch, tasks_len_mask, tasks_seq_lens):
        """
        Generate critical tasks and their replicas:
        - `tasks_seqs_batch`: batch of only normal tasks
        - `tasks_len_mask`: mask of normal tasks, list of 1 and 0
        -  `tasks_seq_lens`: indicates the length of the tasks in each batch
        """
        batch_critical_tasks = []
        critical_copy_mask = []
        tasks_with_critical = tasks_seqs_batch.copy()
        batch_ci_groups = []
        for tasks_seq, len_mask, seq_len in zip(
            tasks_with_critical, tasks_len_mask, tasks_seq_lens
        ):
            critical_tasks = [
                sample[0]
                for sample in random.sample(
                    list(enumerate(tasks_seq[:seq_len])), k=self.num_critical_tasks
                )
            ]
            batch_critical_tasks.append(critical_tasks)
            critical_mask = len_mask.copy()
            ci_groups = []
            for idx, ci in enumerate(critical_tasks):
                critical_mask[ci] = (
                    2 + idx
                )  # First Change the mask of the original critical tasks
            for idx, ci in enumerate(
                critical_tasks
            ):  # Create copies for the critical tasks and change their value and mask
                critical_task_copies = random.sample(
                    list(np.where(critical_mask == 1.0)[0]), k=2
                )
                critical_mask[critical_task_copies] = 2 + idx
                tasks_seq[critical_task_copies] = tasks_seq[ci]
                ci_groups.append([ci] + critical_task_copies)
            critical_copy_mask.append(critical_mask)
            batch_ci_groups.append(ci_groups)
        return (tasks_with_critical, critical_copy_mask, batch_ci_groups)
    
    def valid_senders(self, valid_tasks, comm_tree):
        """
        Get valid senders from communication tree i.e. 
        """
        valid_senders_depths = {}
        for task in valid_tasks:
            depth = comm_tree.get_node_depth(task)
            if depth is None:
                valid_senders_depths[task] = 0
            elif depth < self.max_comm_chain:
                valid_senders_depths[task] = depth
        return valid_senders_depths

    def valid_receivers(self, comm_tree, valid_senders_depths, sender, mask, cost):
        """
        Get valid receivers for a sender
        """
        valid_receivers = list(valid_senders_depths.keys())
        # exclude sender from possible receivers
        valid_receivers.remove(sender)
        # get ancestors of sender
        sender_ancestors = comm_tree.get_all_ancestors(sender)
        # get sender's depth, cost and critical mask value
        sender_depth = valid_senders_depths[sender]
        sender_cost = cost[sender]
        sender_critical_mask = mask[sender]
        # loop through all valid receivers and check if they satisfy all conditions
        final_receivers = []
        for receiver in valid_receivers:
            # 1. Is not ancestor of sender
            if receiver in sender_ancestors:
                continue
            # 2. Longest depth of sender + Longest depth of receiver ≤ max_comm_length
            receiver_depth = valid_senders_depths[receiver]
            if sender_depth + receiver_depth > self.max_comm_chain:
                continue
            # 3. Cost of ancestors of sender + sender cost + Cost of ancestors of receiver + Cost of receiver ≤ avg_node_bin_size
            receiver_ancestors = comm_tree.get_all_ancestors(receiver)
            receiver_cost = cost[receiver]
            sender_ancestors_cost = sum([cost[ancestor] for ancestor in sender_ancestors])
            receiver_ancestors_cost = sum([cost[ancestor] for ancestor in receiver_ancestors])
            if sender_cost + sender_ancestors_cost + receiver_cost + receiver_ancestors_cost > self.max_node_size:
                continue
            # 4. Intersection of critical mask values of (sender + its ancestors) and (receiver + its ancestors) should be None
            receiver_critical_mask = mask[receiver]
            sender_ancestors_critical_mask = [mask[ancestor] for ancestor in sender_ancestors]
            receiver_ancestors_critical_mask = [mask[ancestor] for ancestor in receiver_ancestors]
            combined_sender_masks = np.concatenate(([sender_critical_mask], sender_ancestors_critical_mask))
            combined_receiver_masks = np.concatenate(([receiver_critical_mask], receiver_ancestors_critical_mask))
            # Find intersection between the combined masks of sender and receiver
            intersection = np.intersect1d(combined_sender_masks, combined_receiver_masks)
            intersection = np.setdiff1d(intersection, np.array([0, 1]))
            if len(intersection) > 0:
                continue
            # if all conditions are satisfied, add receiver to final_receivers
            final_receivers.append(receiver)
        return final_receivers
    
    def generate_chained_communications(self, states, critical_mask, states_lens):
        """
        Generate communication matrix for each batch
        """
        batch_comms = []
        batch_comms_count = []
        for state, mask, seq_len in zip(states, critical_mask, states_lens):
            # comms = np.zeros((seq_len, seq_len), dtype="uint8")
            num_comms = np.random.randint(self.min_num_comms, self.max_num_comms + 1)
            valid_tasks = np.where(state != 0)[0] # valid tasks indices
            comm_tree = CommunicationTree(max_depth=self.max_num_tasks)
            for _ in range(num_comms):
                # get valid senders
                valid_senders_depths = self.valid_senders(valid_tasks, comm_tree)
                valid_senders = list(valid_senders_depths.keys())
                # select random sender
                sender = random.choice(valid_senders)
                # get valid receivers for sender
                valid_receivers = self.valid_receivers(comm_tree, valid_senders_depths, sender, mask, state)
                # if there are no valid receivers, continue
                if len(valid_receivers) == 0:
                    continue
                # select random receiver among valid receivers
                receiver = random.choice(valid_receivers)
                # update comms matrix
                comm_tree.add_edge(sender, receiver)
            batch_comms.append(comm_tree.to_matrix())
            batch_comms_count.append(num_comms)
        return batch_comms, batch_comms_count
    
    def generate_communications(self, states, critical_mask, states_lens):
        """
        Generate communication matrix for each batch
        """
        batch_comms = []
        batch_comms_count = []
        for state, mask, seq_len in zip(states, critical_mask, states_lens):
            comms = np.zeros((seq_len, seq_len), dtype="uint8")
            num_comms = np.random.randint(self.min_num_comms, self.max_num_comms + 1)
            valid_tasks = np.where(state != 0)[0]  # valid tasks indices
            for _ in range(num_comms):
                # select random sender
                sender = random.choice(valid_tasks)
                # remove sender from valid_tasks
                valid_receivers = np.setdiff1d(valid_tasks, np.array([sender]))
                # remove receivers which are already communicating with sender (handling duplicate entries)
                valid_receivers = np.setdiff1d(
                    valid_receivers, np.where(comms[sender] == 1)[0]
                )
                # get sender's mask value
                sender_mask = mask[sender]
                # is sender critical
                is_sender_critical = sender_mask > 1
                # if sender is critical, get receivers which are not replicas of sender
                if is_sender_critical:
                    # get all the unselected_tasks which don't have the same mask value as sender
                    valid_receivers = [
                        task for task in valid_receivers if mask[task] != sender_mask
                    ]
                # select random receiver
                receiver = random.choice(valid_receivers)
                comms[sender, receiver] = 1
            batch_comms.append(comms)
            batch_comms_count.append(num_comms)
        return batch_comms, batch_comms_count

    def generate_states_batch(self, batch_size=1):
        """
            Generate new batch of initial states
            Returns:
            states - 2D Array (Batch Size x num of tasks) - elements contain size of task
            states_lens - 1D Array (Batch Size) - elements contain num of valid tasks in each row of states variable
            states_mask - 2D Array (Batch Size x num of tasks) - elements represent a mask of states variable having all 1 values
            nodes_available - 2D Array (Batch Size x num of nodes) - elements contain size of nodes (imp: all batches contain same values of node sizes)
        """
        tasks_seqs_batch = np.random.randint(
            low=self.min_task_size,
            high=self.max_task_size + 1,
            size=(batch_size, self.max_num_tasks),
        )

        tasks_len_mask = np.ones_like(tasks_seqs_batch, dtype="float32")
        tasks_seq_lens = np.random.randint(
            low=self.min_num_tasks, high=self.max_num_tasks + 1, size=batch_size
        )

        nodes_available = []
        node_choices = [
            # using +1 with self.max_node_size to handle using same min and max node size
            node_size
            for node_size in range(self.min_node_size, self.max_node_size + 1, 100)
        ]

        for i in range(self.total_nodes):
            nodes_available.append(random.choice(node_choices))

        nodes_available = np.array(nodes_available)
        nodes_available = np.repeat(
            nodes_available[np.newaxis, ...], batch_size, axis=0
        )

        # nodes_available = np.random.randint(
        #     low=self.min_node_size, high=self.max_node_size + 1,
        #     size=(batch_size,self.total_nodes)
        # )

        for tasks_seq, len_mask, seq_len in zip(
            tasks_seqs_batch, tasks_len_mask, tasks_seq_lens
        ):
            tasks_seq[seq_len:] = 0
            len_mask[seq_len:] = 0

        return (tasks_seqs_batch, tasks_seq_lens, tasks_len_mask, nodes_available)
