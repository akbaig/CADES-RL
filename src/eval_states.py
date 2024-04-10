from sample_generator import generate_random_graph_with_input

G, graph_dict = generate_random_graph_with_input(4)
print(graph_dict)
print(G)
final_allocation = [['t1'], ['t2', 't4'], ['t3']]


no_of_tasks_in_same_node = 0
for successors in graph_dict.values():
    if len(successors) > 2:
        for i in range(len(successors)):
            for j in range(i+1, len(successors)):
                node_1, node_2 = successors[i], successors[j]
                if G.has_edge(node_1, node_2) and [node_1, node_2] in final_allocation:
                    no_of_tasks_in_same_node += 1

print(no_of_tasks_in_same_node)
print(G.number_of_edges())
