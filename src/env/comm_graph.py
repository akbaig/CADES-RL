import networkx as nx
import numpy as np

class CommunicationGraph:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.graph = nx.DiGraph()
        for i in range(max_depth):
            self.graph.add_node(i)

    def get_or_create_node(self, name):
        if name not in self.graph:
            self.graph.add_node(name)
        return name

    def add_edge(self, parent, child):
        self.graph.add_edge(parent, child)

    def add_edges(self, edges):
        for parent, child in edges:
            self.add_edge(parent, child)

    def get_ancestors(self, name):
        
        if name not in self.graph:
            return []
        return list(nx.ancestors(self.graph, name))
    
    def get_node_depth(self, name):
        """
        Find the depth of a node given its name, considering the shortest path from any root node.
        Each root node has a depth of 0. Returns the minimum depth if multiple paths exist.
        """
        if name not in self.graph:
            return None
        
        ancestors = self.get_ancestors(name)
        return len(ancestors) + 1
        
    def to_matrix(self):
        adj_matrix = nx.adjacency_matrix(self.graph)
        # Convert to a dense format for easy handling or printing
        dense_matrix = adj_matrix.todense()
        # Convert the dense matrix to a numpy array of dtype uint8
        return np.array(dense_matrix, dtype=np.uint8)