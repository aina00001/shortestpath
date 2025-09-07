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
