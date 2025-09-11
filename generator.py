import random
import heapq
from pyvis.network import Network


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
                block.sort(key=lambda x: x[1])  # keep block ordered
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

    def __repr__(self):
        return f"Blocks={self.blocks}, Map={self.map}"
        

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
