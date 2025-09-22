import heapq
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple, Optional
from generator import generate_graph, dijkstra, draw_graph_interactive
import time

# -------------------------
# SimpleBlockDS (heap-backed)
# -------------------------
class SimpleBlockDS:
    """
    Simplified Block DS backed by a min-heap and a map for lazy deletion.
    Provides: initialize(M,B), insert(a,b), delete(a,b_or_none), batch_prepend(list_of_pairs),
    pull(M) -> (Bi, list_of_pairs) where pulled items are removed from the DS.
    """
    def __init__(self, M: int = 16, B: float = float("inf")):
        self.initialize(M, B)

    def initialize(self, M: int, B: float):
        self.M = M
        self.B = B
        self.heap: List[Tuple[float, str]] = []   # (value, key)
        self.map: Dict[str, float] = {}           # key -> current valid value

    def insert(self, a: str, b: float):
        # keep only smaller value for a
        if a in self.map and b >= self.map[a]:
            return
        self.map[a] = b
        heapq.heappush(self.heap, (b, a))

    def delete(self, a: str, b: Optional[float] = None):
        # lazy deletion: just remove key from map
        if a in self.map:
            del self.map[a]

    def batch_prepend(self, pairs: Iterable[Tuple[str, float]]):
        # semantics: insert these key/value pairs (they may be smaller values)
        for a, b in pairs:
            self.insert(a, b)

    def pull(self, M: int) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Remove and return up to M smallest valid pairs.
        Returns (Bi, Si) where Si is list of (key, value) popped, and
        Bi is the smallest remaining value (upper bound) or self.B if none.
        """
        S: List[Tuple[str, float]] = []
        while len(S) < M and self.heap:
            val, key = heapq.heappop(self.heap)
            # ignore outdated entries
            if key not in self.map or self.map[key] != val:
                continue
            # valid entry: remove it (pop) and append to S
            del self.map[key]
            S.append((key, val))

        # find next smallest valid value (Bi)
        Bi = self.B
        while self.heap:
            val, key = self.heap[0]
            if key in self.map and self.map[key] == val:
                Bi = val
                break
            heapq.heappop(self.heap)  # discard outdated top

        return Bi, S

    def empty(self) -> bool:
        # clean top outdated
        while self.heap:
            val, key = self.heap[0]
            if key in self.map and self.map[key] == val:
                return False
            heapq.heappop(self.heap)
        return True

    def __repr__(self):
        # show the valid contents (not cleaning heap fully)
        valid = [(k, v) for k, v in self.map.items()]
        return f"SimpleBlockDS(M={self.M},B={self.B},items={valid})"


# -------------------------
# Algorithm 2: BaseCase
# -------------------------
def base_case(B: float,
              S: Set[str],
              adj: Dict[str, List[Tuple[str, float]]],
              d_hat: Dict[str, float],
              k: int) -> Tuple[float, Set[str]]:
    """
    Algorithm 2 (BaseCase) — S must be singleton {x} and x assumed complete.
    Mutates d_hat in place. Returns (B_prime, U).
    """
    if len(S) != 1:
        raise ValueError("base_case expects S to be a singleton set {x}.")

    x = next(iter(S))
    U0: Set[str] = set(S)

    heap: List[Tuple[float, str]] = []
    # push starting node x with its current d_hat[x]
    heapq.heappush(heap, (d_hat.get(x, float("inf")), x))

    while heap and len(U0) < (k + 1):
        du, u = heapq.heappop(heap)
        # lazy skip
        if du != d_hat.get(u, float("inf")):
            continue

        U0.add(u)
        # relax outgoing edges
        for v, w in adj.get(u, []):
            newd = du + w
            # paper uses <= and requires newd < B for adding to heap
            if newd <= d_hat.get(v, float("inf")) and newd < B:
                d_hat[v] = newd
                heapq.heappush(heap, (newd, v))

    if len(U0) <= k:
        return B, U0
    else:
        B_prime = max(d_hat[v] for v in U0)
        U = {v for v in U0 if d_hat[v] < B_prime}
        return B_prime, U


# -------------------------
# Algorithm 1: FindPivots
# -------------------------
def find_pivots(B: float,
                S: Set[str],
                adj: Dict[str, List[Tuple[str, float]]],
                d_hat: Dict[str, float],
                k: int) -> Tuple[Set[str], Set[str]]:
    """
    Algorithm 1 (FindPivots).
    - B: bound
    - S: set of (complete) vertices
    - adj: adjacency list
    - d_hat: dict of tentative distances (mutated in place)
    - k: number of relaxation steps
    Returns: (P, W)
    """
    # W = S ∪ W1 ∪ ... ; W0 = S
    W: Set[str] = set(S)
    Wi: Set[str] = set(S)

    for i in range(1, k + 1):
        Wi_next: Set[str] = set()
        # relax from nodes in Wi
        for u in list(Wi):
            du = d_hat.get(u, float("inf"))
            for v, w in adj.get(u, []):
                newd = du + w
                # paper uses <= here
                if newd <= d_hat.get(v, float("inf")):
                    d_hat[v] = newd
                    if newd < B:
                        Wi_next.add(v)
                    W.add(v)
        Wi = Wi_next
        if len(W) > k * len(S):
            # large workload case
            return set(S), W

    # Build forest F with edges (u,v) where u,v in W and d_hat[v] == d_hat[u] + w
    forest_children: Dict[str, List[str]] = defaultdict(list)
    for u in W:
        for v, w in adj.get(u, []):
            if v in W and abs(d_hat.get(v, float("inf")) - (d_hat.get(u, float("inf")) + w)) <= 1e-12:
                forest_children[u].append(v)

    # compute subtree sizes (DFS)
    def subtree_size(root: str, visited: Set[str]) -> int:
        # returns size of subtree rooted at root (in F)
        size = 1
        for child in forest_children.get(root, []):
            if child not in visited:
                visited.add(child)
                size += subtree_size(child, visited)
        return size

    P: Set[str] = set()
    for x in S:
        visited_local: Set[str] = set()
        # Only if x is present in forest_children or reachable
        size = subtree_size(x, visited_local)
        if size >= k:
            P.add(x)

    return P, W


# -------------------------
# Algorithm 3: BMSSP (recursive)
# -------------------------
def bmssp(l: int,
          B: float,
          S: Set[str],
          adj: Dict[str, List[Tuple[str, float]]],
          d_hat: Dict[str, float],
          k: int,
          t: int) -> Tuple[float, Set[str]]:
    """
    Algorithm 3 BMSSP(l, B, S)
    Returns (B_prime, U) and mutates d_hat in-place.
    """
    # print("BMSSP called with l =", l, "B =", B, "S =", S)
    # Base case
    if l == 0:
        return base_case(B, S, adj, d_hat, k)

    # Find pivots
    P, W = find_pivots(B, S, adj, d_hat, k)

    # Initialize D (block DS) with M := 2*(l-1)*t and upper bound B
    M = max(1, 2 * (l - 1) * t)
    D = SimpleBlockDS(M=M, B=B)
    # Insert pivots into D
    for x in P:
        D.insert(x, d_hat.get(x, float("inf")))

    i = 0
    B0_prime = min((d_hat[x] for x in P), default=B)
    U: Set[str] = set()
    threshold = k * (2 ** l) * t  # k * 2^l * t

    # main loop
    while len(U) < threshold and not D.empty():
        i += 1
        Bi, Si_pairs = D.pull(M)   # Si_pairs: list[(key, val)] popped from D
        Si_keys = {k0 for (k0, _) in Si_pairs}
        # recursive call with level l-1
        B_i_prime, Ui = bmssp(l - 1, Bi, Si_keys, adj, d_hat, k, t)
        # add returned Ui nodes to U
        U.update(Ui)

        # prepare a set K to be batch-prepended later
        K_list: List[Tuple[str, float]] = []

        # relax outgoing edges from nodes in Ui
        for u in Ui:
            du = d_hat.get(u, float("inf"))
            for v, w in adj.get(u, []):
                newd = du + w
                # check relaxation condition (<=)
                if newd <= d_hat.get(v, float("inf")):
                    d_hat[v] = newd
                    # if newd in [Bi, B) insert into D
                    if Bi <= newd < B:
                        D.insert(v, newd)
                    # else if newd in [B_i_prime, Bi) collect for batch prepend
                    elif B_i_prime <= newd < Bi:
                        K_list.append((v, newd))

        # Also some x in Si that have d_hat[x] in [B_i_prime, Bi) should be prepended
        # Si_pairs contain the values at the time of pull; but d_hat might have changed.
        for x0, old_val in Si_pairs:
            cur = d_hat.get(x0, float("inf"))
            if B_i_prime <= cur < Bi:
                K_list.append((x0, cur))

        # batch prepend all K_list plus the Si items with d_hat in [B_i_prime, Bi)
        if K_list:
            D.batch_prepend(K_list)

        # check termination condition (partial execution)
        if len(U) >= threshold:
            B_prime = min(B_i_prime, B)
            # add those x in W with d_hat[x] < B_prime
            extra = {x for x in W if d_hat.get(x, float("inf")) < B_prime}
            U.update(extra)
            return B_prime, U

    # If loop exits because D is empty -> successful execution
    B_prime = min((B0_prime, B))
    # include W nodes with d_hat[x] < B_prime
    extra = {x for x in W if d_hat.get(x, float("inf")) < B_prime}
    U.update(extra)
    return B_prime, U


# -------------------------
# High-level driver to run SSSP via BMSSP
# -------------------------
def sssp_via_bmssp(adj: Dict[str, List[Tuple[str, float]]],
                   source: str) -> Dict[str, float]:
    """
    Top-level wrapper that runs BMSSP to compute d_hat distances from source to all nodes.
    This computes k and t from n = number of vertices using the paper's style:
      k = floor((log n)^(1/3)), t = floor((log n)^(2/3))
    and sets l = ceil((log n)/t).
    Returns final d_hat dictionary with distances.
    """
    # build vertex set
    vertices = set(adj.keys())
    for u in list(adj.keys()):
        for v, _ in adj[u]:
            vertices.add(v)
    n = max(2, len(vertices))
    logn = max(1.0, math.log(n))  # natural log
    k = max(1, int(math.floor(logn ** (1 / 3.0))))
    t = max(1, int(math.floor(logn ** (2 / 3.0))))
    l = max(0, int(math.ceil(math.log(n) / t))) if t > 0 else 0

    # initialize d_hat
    d_hat: Dict[str, float] = {v: float("inf") for v in vertices}
    d_hat[source] = 0.0

    # run BMSSP
    B_final, U_final = bmssp(l, float("inf"), {source}, adj, d_hat, k, t)

    # After successful run, d_hat should contain final distances for reachable nodes.
    return d_hat


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # small sample graph
    # adj_example = {
    #     "s": [("a", 2.0), ("b", 5.0)],
    #     "a": [("c", 2.0)],
    #     "b": [("c", 1.0)],
    #     "c": [("d", 3.0)],
    #     "d": []
    # }
    
    num = 200

    num_nodes = num
    num_edges = num*100
    g = generate_graph(num_nodes=num_nodes, num_edges=num_edges, directed=True)
    print("Graph adjacency list:")
    for node, edges in g.items():
        print(f"{node}: {edges}")

    source = 0
    target = num-1
    
    start_bmssp = time.time()

    print("dijkstra----------------------------------------------------------------")
    start_dijkstra = time.time()
    dist, dist_target, path = dijkstra(g, source, target)
    end_dijkstra = time.time() - start_dijkstra
    print(f"dijkstra time: {end_dijkstra}")
    print(dist)
    print(f"\nShortest path from {source} to {target}: {path} with distance {dist_target}")
    
    
    print("bmssp-------------------------------------------------------------------")
    print("Running SSSP via BMSSP on a small graph...")
    distances = sssp_via_bmssp(g, source)
    print("Distances:", distances)
    
    end_bmssp = time.time() - start_bmssp
    print(f"bmssp time: {end_bmssp}")


    draw_graph_interactive(g, path=path, directed=False, output_file="graph_new.html")


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