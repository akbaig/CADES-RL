import numpy as np

class Node:
    def __init__(self, name):
        # assert isinstance(name, int), "Node name must be an integer"
        self.name = name
        self.children = []
        self.parents = []  # List to hold multiple parents

    def add_child(self, node):
        self.children.append(node)
        node.parents.append(self)  # Add this node to the child's list of parents

class CommunicationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.nodes = {}
        self.roots = set()  # Now tracking multiple roots

    def get_or_create_node(self, name):
        # assert isinstance(name, int), "Node name must be an integer"
        if name not in self.nodes:
            new_node = Node(name)
            self.nodes[name] = new_node
            self.roots.add(new_node)  # Assume it's a root until a parent is found
        return self.nodes[name]

    def add_edge(self, parent, child):
        parent_node = self.get_or_create_node(parent)
        child_node = self.get_or_create_node(child)
        parent_node.add_child(child_node)
        self.roots.discard(child_node)  # Remove from roots if it has a parent

    def add_edges(self, edges):
        for parent, child in edges:
            self.add_edge(parent, child)

    def get_all_ancestors(self, node_name):
        """
        Return a set of all ancestor nodes (by name) for a given node name,
        including both direct parents and all levels of grandparents.
        Returns [] if no ancestors are found.
        """
        if node_name not in self.nodes:
            return []

        def find_ancestors(node, ancestors):
            for parent in node.parents:
                ancestors.add(parent.name)
                find_ancestors(parent, ancestors)

        ancestors = set()
        find_ancestors(self.nodes[node_name], ancestors)
        
        # Return None if no ancestors are found
        return list(ancestors) if ancestors else []
    
    def get_node_depth(self, node_name):
        """
        Find the depth of a node given its name, considering the shortest path from any root node.
        Each root node has a depth of 0. Returns the minimum depth if multiple paths exist.
        """
        if node_name not in self.nodes:
            return None  # Node does not exist

        def bfs_min_depth(target):
            visited = set()
            queue = [(node, 0) for node in self.roots]  # Start with all root nodes
            
            while queue:
                current_node, depth = queue.pop(0)
                if current_node.name == target:
                    return depth
                if current_node.name not in visited:
                    visited.add(current_node.name)
                    for child in current_node.children:
                        queue.append((child, depth + 1))
            return None
        #return bfs_min_depth(node_name)
        
        def dfs_max_depth(node, target, current_depth=0, visited=set()):
            if node.name == target:
                return current_depth
            max_depth = None
            visited.add(node.name)
            for child in node.children:
                if child.name not in visited:  # Avoid revisiting
                    child_depth = dfs_max_depth(child, target, current_depth + 1, visited.copy())
                    if child_depth is not None:
                        if max_depth is None or child_depth > max_depth:
                            max_depth = child_depth
            return max_depth

        max_depths = []
        for root in self.roots:
            root_depth = dfs_max_depth(root, node_name)
            if root_depth is not None:
                max_depths.append(root_depth)
        
        return max(max_depths) + 1 if max_depths else None

        
        #max_depths = [dfs_max_depth(root) for root in self.roots]
        #return max(max_depths) + 1 if max_depths else None
        
    def to_matrix(self):
        # initalize np.zeros of size self.max_depth x self.max_depth
        matrix = np.zeros((self.max_depth, self.max_depth), dtype="uint8")
        for node in self.nodes.values():
            node_depth = self.get_node_depth(node.name)
            if node_depth < self.max_depth:
                for child in node.children:
                    child_depth = self.get_node_depth(child.name)
                    if child_depth < self.max_depth:
                        matrix[node.name][child.name] = 1
        return matrix