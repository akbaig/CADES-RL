from sample_generator import generate_random_graph_with_input

# if given string {t1 -> {t2 -> t4,   t3}, t5 -> {t6 -> t7} } X
# {'t1': ['t4'], 't2': ['t3', 't4', 't5'], 't3': ['t4'], 't4': ['t5']}

def get_evaluation(final_allocation, num_tasks):
    '''
        Get final evaluation as: Total no. of edges - no. of tasks allocated in the same node
        
        Params:
        - final_allocation: 2d list of the final allocation of tasks in nodes
                            e.g. [['t1'], ['t2', 't4'], ['t3']]
        - num_tasks: total no. of tasks
        
        Returns: The total bandwidth
        - 
    '''

    G, graph_dict = generate_random_graph_with_input(num_tasks)
    no_of_tasks_in_same_node = 0

    for successors in graph_dict.values():
        if len(successors) > 2:
            for i in range(len(successors)):
                for j in range(i+1, len(successors)):
                    node_1, node_2 = successors[i], successors[j]
                    if G.has_edge(node_1, node_2) and [node_1, node_2] in final_allocation:
                        no_of_tasks_in_same_node += 1
    return G.number_of_edges() - no_of_tasks_in_same_node


# Todo

# get number of edges in the graph
# get active edges in internode in the graph (current implmentation)
# get percentage of active edges/ total edges
# generate graphs from random
# generate graphs from user input
# generate spaces.Box version of the graph. Makes sure it's the same as cades_env
### Other Notes: Should allow the model to run and infer on the code by calling separate methods (for now)


get_evaluation([['t1'], ['t2', 't4'], ['t3']], 5)