import random
import heapq
from pyvis.network import Network
from collections import defaultdict


def generate_graph(num_nodes: int, num_edges: int, directed: bool = False, 
                   min_weight: int = 1, max_weight: int = 10):
    """
    Generate a random weighted graph.
    Returns adjacency list: {node: [(neighbor, weight), ...]}
    """
    graph = {i: [] for i in range(num_nodes)}
    all_possible_edges = [(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v]

    chosen_edges = random.sample(all_possible_edges, min(num_edges, len(all_possible_edges)))

    for u, v in chosen_edges:
        weight = random.randint(min_weight, max_weight)
        graph[u].append((v, weight))
        if not directed:
            graph[v].append((u, weight))

    return graph


def dijkstra(graph, start, target):
    """
    Dijkstra algorithm to find shortest path from start to target.
    Returns: (distance, path)
    """
    dist = {node: float("inf") for node in graph}
    dist[start] = 0
    prev = {node: None for node in graph}
    pq = [(0, start)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if u == target:
            break

        if current_dist > dist[u]:
            continue

        for v, weight in graph[u]:
            alt = current_dist + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    # Reconstruct path
    path = []
    u = target
    while u is not None:
        path.append(u)
        u = prev[u]
    path.reverse()

    return dist[target], path


def draw_graph_interactive(graph, path=None, directed=False, output_file="graph.html"):
    """
    Draws the graph interactively with PyVis.
    Highlights the shortest path if provided.
    """
    net = Network(directed=directed, notebook=False, height="600px", width="100%")

    # Add nodes
    for u in graph:
        net.add_node(u, label=str(u), color="lightblue")

    # Add edges
    for u, neighbors in graph.items():
        for v, w in neighbors:
            color = "black"
            width = 1
            if path and u in path and v in path:
                # Highlight edges if they are part of the shortest path
                idx_u, idx_v = path.index(u), path.index(v)
                if abs(idx_u - idx_v) == 1:  
                    color = "red"
                    width = 3
            net.add_edge(u, v, label=str(w), color=color, width=width)

    net.write_html(output_file)
    print(f"✅ Graph saved to {output_file}. Open it in your browser.")

class BlockDS:
    def __init__(self, M, B):
        """
        Initialize the block data structure.
        M: max block size
        B: upper bound on values
        """
        
        self.M = M
        self.B = B
        self.blocks = [[]]
        self.map = {}
        
    def delete(self, a, b):
        """
        delete key a with value b
        """
        
        if a not in self.map:
            return
        block_idx, b0 = self.map[a]
        
        if b0 != b:
            return
        
        block = self.blocks[block_idx]


        for i, (k, v) in enumerate(block):
            if k == a and v == b0:
                block.pop(i)
                del self.map[a]
                break
        if not block and len(self.blocks) > 1:
            self.blocks.pop(block_idx)
        
        
    def insert(self, a, b):
        """
        a=key, b=value
        If key exists, keep the minimum value.
        """
        if a in self.map:
            block_idx, b0 = self.map[a]
            if b >= b0:
                return
            self.delete(a, b0)
            
        inserted = False
        for i, block in enumerate(self.blocks):
            if not block or block[-1][1] >= b:
                block.append((a, b))
                block.sort(key=lambda x: x[1])  # keep block ordered by value
                self.map[a] = (i, b)
                inserted = True
                break
            
        if not inserted:
            self.blocks[-1].append((a, b))
            self.blocks[-1].sort(key=lambda x: x[1])
            self.map[a] = (len(self.blocks) - 1, b)
                
        if len(self.blocks[self.map[a][0]]) > self.M:
            # split block
            self.split(self.map[a][0])
            
            
    def split(self, idx):
        """Split block at index idx into two."""
        block = self.blocks[idx]
        mid = len(block) // 2
        left = block[:mid]
        right = block[mid:]
        self.blocks[idx] = left
        self.blocks.insert(idx + 1, right)
        # rebuild map for affected keys
        for i, (a, b) in enumerate(left):
            self.map[a] = (idx, b)
        for i, (a, b) in enumerate(right):
            self.map[a] = (idx + 1, b)
    
    def batchPrepend(self, L: list[tuple], M: int):
        """
        Prepend list L of (key, value) tuples in batches of size M.
        """
        if len(L) <= M:
            self.blocks.insert(0, L)
            for a, b in L:
                self.map[a] = (0, b)
            return
        
        for i in range(0, len(L), M):
            batch = L[i:i + M]
            self.blocks.insert(0, batch)
            for a, b in batch:
                self.map[a] = (0, b)

    def pull(self):
        """
        Pull the minimum element (key, value) from the structure.
        """
        if not self.blocks or not self.blocks[0]:
            return None
        a, b = self.blocks[0].pop(0) # first one is min as blocks are sorted
        del self.map[a]
        if not self.blocks[0] and len(self.blocks) > 1:
            self.blocks.pop(0)
            # Update map for remaining blocks
            for i, block in enumerate(self.blocks):
                for a, b in block:
                    self.map[a] = (i, b)
        return (a, b)

    def __repr__(self):
        return f"Blocks={self.blocks}, Map={self.map}"
        
def find_pivots(bound_B, complete_vertices, graph_edges, k_steps):
    """
    Implementation of Algorithm 1: Finding Pivots.
    
    Parameters:
    -----------
    bound_B : int or float
        The distance bound (B).
    complete_vertices : set
        Set of complete vertices (S).
    graph_edges : dict
        Graph adjacency list: graph_edges[u] = list of (v, weight_uv).
    k_steps : int
        Number of relaxation steps (k).
    
    Returns:
    --------
    pivot_vertices : set
        Pivot vertices from S.
    reachable_vertices : set
        Set of vertices reachable within distance B.
    """

    # Line 2: Initialize reachable set with complete vertices
    reachable_vertices = set(complete_vertices)
    
    # Line 3: Initial frontier = complete vertices
    current_frontier = set(complete_vertices)

    # Distance dictionary (initialized to ∞)
    distance = defaultdict(lambda: float("inf"))
    for vertex in complete_vertices:
        distance[vertex] = 0

    # Line 4–10: Relax edges for k_steps iterations
    for step in range(1, k_steps + 1):  
        next_frontier = set()
        
        for source in current_frontier:  # for all edges (source, target) with source in previous frontier
            for target, edge_weight in graph_edges.get(source, []):
                
                # Relax edge
                if distance[source] + edge_weight < distance[target]:
                    distance[target] = distance[source] + edge_weight
                
                # If still within bound_B, add target to next frontier
                if distance[source] + edge_weight < bound_B:
                    next_frontier.add(target)
        
        current_frontier = next_frontier
        reachable_vertices |= current_frontier  # Union with global reachable set
    
    # Line 11–13: If reachable set is too large
    if len(reachable_vertices) > k_steps * len(complete_vertices):
        return set(complete_vertices), reachable_vertices
    
    # Line 14–15: Build directed forest F (shortest-path edges)
    forest_edges = {
        (u, v) for u in reachable_vertices 
        for (v, weight) in graph_edges.get(u, [])
        if abs(distance[u] + weight - distance[v]) < 1e-9  # d[v] = d[u] + w
    }
    
    # Build adjacency from forest edges
    forest_children = defaultdict(list)
    for parent, child in forest_edges:
        forest_children[parent].append(child)
    
    # DFS to compute subtree sizes
    def compute_subtree_size(node, visited):
        if node in visited:
            return 0
        visited.add(node)
        size = 1
        for child in forest_children[node]:
            size += compute_subtree_size(child, visited)
        return size

    # Line 16: Define pivot vertices
    pivot_vertices = set()
    for vertex in complete_vertices:
        subtree_size = compute_subtree_size(vertex, set())
        if subtree_size >= k_steps:
            pivot_vertices.add(vertex)

    # Line 17: Return pivot and reachable sets
    return pivot_vertices, reachable_vertices


# Example usage
if __name__ == "__main__":
    # num_edges = 100
    # num_nodes = 60
    # g = generate_graph(num_nodes=num_nodes, num_edges=num_edges, directed=False)
    # print("Graph adjacency list:")
    # for node, edges in g.items():
    #     print(f"{node}: {edges}")

    # start, target = 0, num_nodes - 1
    
    
    # dist, path = dijkstra(g, start, target)
    # print(f"\nShortest path from {start} to {target}: {path} with distance {dist}")

    # draw_graph_interactive(g, path=path, directed=False, output_file="graph.html")
# Initialize with block size M=3 and bound B=100
    ds = BlockDS(3, 100)

    # Insert elements
    ds.insert("x", 10)
    ds.insert("y", 5)
    ds.insert("z", 20)
    ds.insert("w", 15)  # causes block split

    print("After insertions:", ds)

    # Update existing key with smaller value
    ds.insert("z", 8)
    print("After updating z:", ds)

    # Delete element
    ds.delete("y", 5)
    print("After deleting y:", ds)
