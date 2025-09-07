import heapq
import random
import time
from collections import defaultdict, deque
from math import inf, log2, floor, ceil

# ---------- Utilities: Dijkstra for correctness checking ----------
def dijkstra(n, adj, src):
    dist = [inf]*n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d,u = heapq.heappop(pq)
        if d!=dist[u]:
            continue
        for v,w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq,(nd,v))
    return dist

# ---------- Research-inspired Band-Partitioned SSSP ----------
def band_sssp(n, adj, src, base=1.0, k=2):
    """
    Experimental band-partitioned SSSP.
    n: number of nodes (0..n-1)
    adj: adjacency list: list of lists of (v, weight)
    src: source node
    base: initial band width (float > 0). Bands are [base*2^i, base*2^(i+1))
    k: number of local relaxation rounds per band (small int >=1)
    Returns dist list (length n) with shortest distances (or inf).
    """
    if base <= 0:
        raise ValueError("base must be > 0")
    dist = [inf]*n
    dist[src] = 0.0

    # Map band index -> deque of nodes to process
    buckets = defaultdict(deque)

    # helper: compute band id for a distance value d
    def band_id(d):
        if d == 0:
            return -1000000  # special band for zero (src)
        # band index = floor(log2(d/base))
        return floor(log2(d/base))

    # initialize with source in its band (special)
    buckets[band_id(0)].append(src)

    # track which band indices exist in increasing order
    # We'll pop bands by increasing band index
    active_band_indices = set(buckets.keys())

    # process bands in increasing order
    while active_band_indices:
        bi = min(active_band_indices)
        active_band_indices.remove(bi)
        q = buckets.pop(bi, deque())

        # We'll perform up to k rounds of relaxations within this band.
        # (This is the "local limited Bellman-Ford" idea.)
        for round_idx in range(k):
            if not q:
                break
            next_q = deque()
            while q:
                u = q.popleft()
                du = dist[u]
                # relax outgoing edges
                for v,w in adj[u]:
                    nd = du + w
                    if nd + 1e-15 < dist[v]:  # small epsilon for floating issues
                        dist[v] = nd
                        bi_v = band_id(nd)
                        # If it belongs to same band, enqueue to next_q for same-band rounds
                        if bi_v == bi:
                            next_q.append(v)
                        else:
                            # else push into its appropriate bucket (future band)
                            buckets[bi_v].append(v)
                            active_band_indices.add(bi_v)
            q = next_q

        # After k rounds, if q still has items (not emptied), push them back into the same band
        while q:
            v = q.popleft()
            buckets[bi].append(v)
            active_band_indices.add(bi)

    return dist

# ---------- Small randomized test + benchmark ----------
def random_graph(n, m, max_w=100, directed=True):
    adj = [[] for _ in range(n)]
    edges = set()
    while len(edges) < m:
        u = random.randrange(n)
        v = random.randrange(n)
        if u == v:
            continue
        if (u,v) in edges:
            continue
        edges.add((u,v))
        w = random.random()*max_w
        adj[u].append((v,w))
        if not directed:
            adj[v].append((u,w))
    return adj

if __name__ == "__main__":
    random.seed(1)
    n = 200    # nodes
    m = 1200   # edges
    src = 0
    adj = random_graph(n,m,max_w=100,directed=True)

    # run Dijkstra
    t0 = time.time()
    d_ref = dijkstra(n, adj, src)
    t1 = time.time()

    # run banded SSSP (try couple parameter settings)
    t2 = time.time()
    d_band = band_sssp(n, adj, src, base=90.0, k=20)
    t3 = time.time()

    # correctness check (within small epsilon)
    eps = 1e-8
    ok = True
    for i in range(n):
        a = d_ref[i]
        b = d_band[i]
        if (a == inf and b == inf):
            continue
        if abs(a-b) > 1e-6:
            print(f"Mismatch node {i}: dijkstra={a}, band={b}")
            ok = False
            break

    print("Correctness:", ok)
    print(f"Dijkstra time: {t1-t0:.4f}s, Band-SSSP time: {t3-t2:.4f}s (params base=1.0,k=2)")
