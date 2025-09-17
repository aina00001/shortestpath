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
        self.D0 = []  # list of blocks for D0
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

    def splitMedian(self, L: list[tuple]):
        """
        Split list L of (key, value) tuples into two lists around the median value.
        Returns two lists: left (<= median) and right (> median).
        """
        if not L:
            return [], []
        # L.sort(key=lambda x: x[1])  # sort by value
        mid = len(L) // 2 # median index
        
        left =  L[:mid+1] # first half including median
        right = L[mid+1:] # second half excluding median
        
        return left, right
    
    def batchPrepend(self, L: list[tuple], M: int):
        """
        Prepend list L of (key, value) tuples in batches of size M into D0.
        """
        if len(L) > M:
            left, right = self.splitMedian(L)
            self.batchPrepend(right, M)
            self.batchPrepend(left, M)
        else:
            self.D0.insert(0, L)

    def pull(self, M : int) -> list[tuple]:
        """
        Pull M the minimum element (key, value) from blocks U D0
        """
        # get one block from D0 and one block from D1
        sp0 = self.D0[0].copy() if len(self.D0) > 1 else []
        sp1 = self.blocks[0].copy() if len(self.blocks) > 0 else []
        # Merge sp0 and sp1 into a single sorted list s
        i, j = 0, 0
        s = []
        while len(s) < M and i < len(sp0) and j < len(sp1):
            print("i, j", i, j)
            if sp0[i][1] <= sp1[j][1]:
                s.append(sp0[i])
                # remove from D0
                self.D0[0].pop(0)
                if self.D0[0] == []:
                    self.D0.pop(0) # remove empty block
                i += 1 
            else:
                s.append(sp1[j])
                self.delete(sp1[j][0], sp1[j][1])
                j += 1
        # If we still need more elements, take from the remainder of sp0 or sp1
        while len(s) < M and i < len(sp0):
            s.append(sp0[i])
            # remove from D0
            self.D0[0].pop(0)
            if self.D0[0] == []:
                self.D0.pop(0) # remove empty block
            i += 1
        while len(s) < M and j < len(sp1):
            s.append(sp1[j])
            self.delete(sp1[j][0], sp1[j][1])
            j += 1
        return s
        

    def __repr__(self):
        return f"Blocks={self.blocks}, Map={self.map}, D0={self.D0}"


def find_pivots(B, S, adj, d, block_ds: BlockDS):
    """
    Algorithm 2: FindPivots(B, S) tied to BlockDS.
    
    B : distance threshold
    S : set of complete vertices
    adj : adjacency list {u: [(v, w), ...]}
    d : dict of current shortest distances {v: dist}
    block_ds : BlockDS instance
    
    Returns:
        P : set of pivot vertices
        W : set of visited vertices
    """
    P = set()
    W = set()

    # Initialize with S
    for u in S:
        P.add(u)
        W.add(u)
        block_ds.insert(u, d[u])

    # Expand until no more nodes within bound B
    while True:
        # Pull one minimum element
        pulled = block_ds.pull(1)
        if not pulled:
            break
        u, du = pulled[0]

        # If above bound, stop
        if du >= B:
            break

        # For each edge (u, v) with weight w
        for v, w in adj[u]:
            if du + w < d.get(v, float("inf")) and du + w < B:
                d[v] = du + w
                block_ds.insert(v, d[v])
                P.add(v)
                W.add(v)

    return P, W


def find_pivots_old(bound_B, complete_vertices, graph_edges, k_steps):
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

    ds.batchPrepend([("a", 1), ("c", 2), ("b", 3), ("d", 4)], M=3)

    print("After batchPrepend:", ds)

    # Update existing key with smaller value
    ds.insert("z", 8)
    print("After updating z:", ds)

    # Delete element
    ds.delete("y", 5)
    print("After deleting y:", ds)

    res = ds.pull(4)
    print("Pulled elements:", res)
    print("After pulling 4 elements:", ds)

# Example graph
    # adj = {
    #     "s": [("a", 2), ("b", 5)],
    #     "a": [("c", 2)],
    #     "b": [("c", 1)],
    #     "c": []
    # }

    # # Distances initialized
    # d = {"s": 0, "a": float("inf"), "b": float("inf"), "c": float("inf")}

    # # Create BlockDS
    # block_ds = BlockDS(M=3, B=2)

    # print("mandeha")
    # # Call Algorithm 2
    # P, W = find_pivots(B=10, S={"s"}, adj=adj, d=d, block_ds=block_ds)

    # print("Pivots:", P)
    # print("W:", W)
    # print("Distances:", d)
    # print("BlockDS state:", block_ds)
