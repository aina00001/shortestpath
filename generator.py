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

from collections import defaultdict

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
    num_edges = 100
    num_nodes = 60
    g = generate_graph(num_nodes=num_nodes, num_edges=num_edges, directed=False)
    print("Graph adjacency list:")
    for node, edges in g.items():
        print(f"{node}: {edges}")

    start, target = 0, num_nodes - 1
    
    
    dist, path = dijkstra(g, start, target)
    print(f"\nShortest path from {start} to {target}: {path} with distance {dist}")

    draw_graph_interactive(g, path=path, directed=False, output_file="graph.html")
