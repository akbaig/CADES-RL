import numpy as np
import random
from env.comm_graph import CommunicationGraph
from env.states_generator import StatesGenerator

class ExtendedStatesGenerator(StatesGenerator):
    """
    Adds the capability to generate communication masks
    """

    def __init__(self, config):
        super().__init__(config)
        self.min_num_comms = config.min_num_comms
        self.max_num_comms = config.max_num_comms
        self.max_comm_chain = config.max_comm_chain
        self.non_critical_comm = config.non_critical_comm
        self.critical_comm = config.critical_comm

    def _get_random_comm_count(self):
        # Generate random number of communications
        return np.random.randint(self.min_num_comms, self.max_num_comms + 1)
    
    def _graph_valid_senders(self, comm_graph: CommunicationGraph, valid_tasks):
        """
        Get valid senders from communication graph i.e. 
        """
        valid_senders_depths = {}
        for task in valid_tasks:
            depth = comm_graph.get_node_depth(task)
            if depth is None:
                valid_senders_depths[task] = 0
            elif depth < self.max_comm_chain:
                valid_senders_depths[task] = depth
        return valid_senders_depths

    def _graph_valid_receivers(self, comm_graph: CommunicationGraph, valid_senders_depths, sender, mask, cost):
        """
        Get valid receivers for a sender
        """
        valid_receivers = list(valid_senders_depths.keys())
        # exclude sender from possible receivers
        valid_receivers.remove(sender)
        # get ancestors of sender
        sender_ancestors = comm_graph.get_ancestors(sender)
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
            receiver_ancestors = comm_graph.get_ancestors(receiver)
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
    
    def _generate_graph_comm_matrix(self, tasks, num_tasks, critical_mask, valid_tasks):
        # Initialize communication graph
        comm_graph = CommunicationGraph(max_depth=self.max_num_tasks)
        # Get number of communications
        required_num_comms = self._get_random_comm_count()
        num_comms = 0
        for _ in range(required_num_comms):
            # get valid senders
            valid_senders_depths = self._graph_valid_senders(comm_graph, valid_tasks)
            valid_senders = list(valid_senders_depths.keys())
            # select random sender
            sender = random.choice(valid_senders)
            # get valid receivers for sender
            valid_receivers = self._graph_valid_receivers(comm_graph, valid_senders_depths, sender, critical_mask, tasks)
            # if there are no valid receivers, continue
            if len(valid_receivers) == 0:
                continue
            # select random receiver among valid receivers
            receiver = random.choice(valid_receivers)
            # update comms matrix
            comm_graph.add_edge(sender, receiver)
            num_comms += 1
        return comm_graph.to_matrix(), num_comms
    
    def _generate_simple_comm_matrix(self, tasks, num_tasks, critical_mask, valid_tasks):
        # Initialize communication matrix
        comms = np.zeros((self.max_num_tasks, self.max_num_tasks), dtype="int8")
        # Get number of communications
        required_num_comms = self._get_random_comm_count()
        num_comms = 0
        for _ in range(required_num_comms):
            # select random sender
            sender = random.choice(valid_tasks)
            # remove sender from valid_tasks
            valid_receivers = np.setdiff1d(valid_tasks, np.array([sender]))
            # remove receivers which are already communicating with sender (handling duplicate entries)
            valid_receivers = np.setdiff1d(
                valid_receivers, np.where(comms[sender] == 1)[0]
            )
            # get sender's mask value
            sender_mask = critical_mask[sender]
            # is sender critical
            is_sender_critical = sender_mask > 0
            # if sender is critical, get receivers which are not replicas of sender
            if is_sender_critical:
                # get all the unselected_tasks which don't have the same mask value as sender
                valid_receivers = [
                    task for task in valid_receivers if critical_mask[task] != sender_mask
                ]
            # if there are no valid receivers, continue
            if len(valid_receivers) == 0:
                continue
            # select random receiver
            receiver = random.choice(valid_receivers)
            comms[sender, receiver] = 1
            num_comms += 1
        return comms, num_comms

    def generate_communications(self, tasks, num_tasks, critical_mask, graph = False):
        # Get valid tasks
        valid_tasks = np.where(tasks > 0)[0]
        # Get critical and replica tasks
        critical_tasks = np.where(critical_mask > 0)[0]
        # Define preliminary communication restrictions
        if self.non_critical_comm and not self.critical_comm:
            valid_tasks = np.setdiff1d(valid_tasks, critical_tasks)
        elif not self.non_critical_comm and self.critical_comm:
            valid_tasks = critical_tasks
        # Generate communications
        if(graph):
            comms = self._generate_graph_comm_matrix(tasks, num_tasks, critical_mask, valid_tasks)
        else:
            comms = self._generate_simple_comm_matrix(tasks, num_tasks, critical_mask, valid_tasks)
        return comms
    
    def generate_communications_batch(self):
        assert("Not implemented")