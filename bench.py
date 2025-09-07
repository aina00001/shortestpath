"""
Partial‑Ordering Single‑Source Shortest Paths in Python
=======================================================

This is a **practical, educational implementation** capturing the core idea behind
"breaking the sorting barrier": avoid totally ordering all vertices by exact
distance (as in Dijkstra). Instead, we process nodes in **distance bands** using
buckets — a technique often called **δ‑stepping**. The actual STOC 2025 paper by
Duan–Mao–Shu–Yin includes sophisticated ingredients to achieve the
O(m log^{2/3} n) bound; reproducing that theory‑tight code is beyond the
scope of a short snippet. This version is designed to be **readable and usable**.

We also add **benchmark comparisons** vs.:
- Dijkstra with a naive O(n) min scan (no heap)
- Dijkstra with a binary heap (heapq)

"""
from collections import defaultdict
from math import inf
import heapq
import random
import time
from typing import List, Tuple

Adj = List[List[Tuple[int, float]]]


def sssp_delta_stepping(n: int, edges: Adj, source: int, delta: float):
    if delta <= 0:
        raise ValueError("delta must be positive")
    dist = [inf] * n
    parent = [-1] * n
    dist[source] = 0.0

    B: dict[int, set[int]] = defaultdict(set)
    B[0].add(source)

    def min_nonempty_bucket_index():
        if not B:
            return None
        return min(B.keys())

    def relax(u: int, v: int, w: float) -> bool:
        du = dist[u]
        dv = dist[v]
        nd = du + w
        if nd < dv:
            dist[v] = nd
            parent[v] = u
            return True
        return False

    while True:
        i = min_nonempty_bucket_index()
        if i is None:
            break

        R = set()
        S = set(B[i])
        del B[i]

        while S:
            R.update(S)
            next_S = set()
            for u in S:
                for v, w in edges[u]:
                    if w <= delta and relax(u, v, w):
                        next_S.add(v)
            S = next_S

        heavy_enqueued = []
        for u in R:
            for v, w in edges[u]:
                if w > delta and relax(u, v, w):
                    b = int(dist[v] // delta)
                    heavy_enqueued.append((b, v))

        for b, v in heavy_enqueued:
            B[b].add(v)

    return dist, parent


def dijkstra_naive(n: int, edges: Adj, source: int):
    dist = [inf] * n
    parent = [-1] * n
    dist[source] = 0.0
    visited = [False] * n
    for _ in range(n):
        # find min distance among unvisited
        u = -1
        best = inf
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1:
            break
        visited[u] = True
        for v, w in edges[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
    return dist, parent


def dijkstra_heap(n: int, edges: Adj, source: int):
    dist = [inf] * n
    parent = [-1] * n
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in edges[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent


def choose_delta(edges: Adj, quantile: float = 0.5) -> float:
    ws = [w for u in range(len(edges)) for _, w in edges[u] if w > 0]
    if not ws:
        return 1.0
    ws.sort()
    k = max(0, min(len(ws) - 1, int(quantile * (len(ws) - 1))))
    return max(1e-12, float(ws[k]))


# ---------------------------------------------------------------
# Benchmark demo
# ---------------------------------------------------------------
if __name__ == "__main__":
    n = 2000
    m = 10000
    random.seed(0)
    edges: Adj = [[] for _ in range(n)]
    for _ in range(m):
        u = random.randrange(n)
        v = random.randrange(n)
        if u != v:
            w = random.uniform(1, 10)
            edges[u].append((v, w))

    source = 0
    delta = choose_delta(edges)

    for name, func in [
        ("Dijkstra naive", dijkstra_naive),
        ("Dijkstra heapq", dijkstra_heap),
        ("Delta-stepping", lambda n,e,s: sssp_delta_stepping(n,e,s,delta)),
    ]:
        t0 = time.time()
        dist, _ = func(n, edges, source)
        t1 = time.time()
        print(f"{name:<20} time={t1-t0:.4f}s, reachable={sum(d<inf for d in dist)}")
