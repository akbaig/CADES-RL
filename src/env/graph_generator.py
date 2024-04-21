import networkx as nx
import matplotlib.pyplot as plt
from random import randint, sample

def generate_random_graph_with_input(num_nodes):
    if num_nodes < 2:  # We need at least two nodes to form a graph
        return "The number of nodes should be at least 2."


    # Create a directed graph
    G = nx.DiGraph()
    graph_dict = {}
    nodes = ['t{}'.format(i+1) for i in range(num_nodes)]
    G.add_nodes_from(nodes)
    
    # Randomly decide the number of edges
    edges = []
    for node in nodes[:-1]:  # Exclude the last node to avoid a self-loop
        successors = sample([x for x in nodes if x > node], 
                            randint(1, len(nodes) - nodes.index(node) - 1))
        graph_dict[node] = successors
        for successor in successors:
            edges.append((node, successor))

    G.add_edges_from(edges)

    # pos = nx.spring_layout(G) 
    # nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=16)
    # plt.show()


    return (G, graph_dict)