from sample_generator import generate_random_graph_with_input


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