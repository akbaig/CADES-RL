import numpy as np

def get_evaluate_message_channel_occupancy(total_comms, intranode_comms):
    # Calculate the internode communications
    internode_comms = (total_comms - intranode_comms)
    # Calculate occupancy as a percentage of the total communications
    occupancy = (internode_comms / total_comms) * 100
    # Return the percentage, rounded to two decimal places
    return round(occupancy, 2)

def get_avg_node_occupancy(total_capacities, current_capacities):
    # Calculate the used capacities
    used_capacities = (total_capacities - current_capacities) / total_capacities
    # Calculate mean used capacity as percentage
    mean_capacity = np.mean(used_capacities) * 100
    # Return the percentage, rounded to two decimal places
    return round(mean_capacity, 2)

def get_avg_active_node_occupancy(total_capacities, current_capacities):
    # Exclude nodes which are inactive i.e. have zero capacity or full capacity
    used_nodes = total_capacities != current_capacities
    total_capacities = total_capacities[used_nodes]
    current_capacities = current_capacities[used_nodes]
    return get_avg_node_occupancy(total_capacities, current_capacities)

def get_empty_nodes_percentage(assignment_status):
    # Count the number of empty nodes
    empty_nodes = assignment_status.count([])
    # Calculate the percentage of empty nodes
    empty_nodes_percentage = (empty_nodes / len(assignment_status)) * 100
    # Return the percentage, rounded to two decimal places
    return round(empty_nodes_percentage, 2)