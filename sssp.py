"""
Partial‑Ordering Single‑Source Shortest Paths in Python
=======================================================

This is a **practical, educational implementation** capturing the core idea behind
"breaking the sorting barrier": avoid totally ordering all vertices by exact
distance (as in Dijkstra). Instead, we process nodes in **distance bands** using
buckets — a technique often called **δ‑stepping** (Meyer & Sanders), which is a
well‑known partial‑ordering SSSP and aligns with the spirit of the more
advanced algorithm by Duan–Mao–Shu–Yin (STOC 2025). The actual STOC paper
includes additional, sophisticated ingredients to achieve the
O(m log^{2/3} n) bound; reproducing that theory‑tight code is beyond the
scope of a short snippet. This version is designed to be **readable and usable**.

Key properties of this implementation:
- Works on directed graphs with **non‑negative** edge weights (floats allowed).
- Avoids priority queues and total vertex sorting.
- Uses **buckets** of width δ to process vertices in partial order.
- Handles both **light** edges (≤ δ) and **heavy** edges (> δ) as per δ‑stepping.
- Pure Python, no third‑party deps; adjacency list input.

API
---
sssp_delta_stepping(n, edges, source, delta)
    n:        number of vertices (0..n-1)
    edges:    adjacency list: list[ list[(v, w)] ] where edges[u] = [(v, weight), ...]
    source:   source vertex id
    delta:    positive bucket width; try heuristics like median edge weight or mean.

Returns: (dist, parent)
    dist[i] = shortest distance from source to i (float('inf') if unreachable)
    parent[i] = predecessor on some shortest path (or -1 for source/unreachable)

Complexity (practical):
    Dependent on δ and graph structure. For many sparse graphs and a reasonable
    δ, this is competitive with binary‑heap Dijkstra, and it illustrates the
    partial‑ordering paradigm clearly.
"""
from collections import defaultdict, deque
from math import inf
from typing import List, Tuple

Adj = List[List[Tuple[int, float]]]


def sssp_delta_stepping(n: int, edges: Adj, source: int, delta: float):
    if delta <= 0:
        raise ValueError("delta must be positive")
    dist = [inf] * n
    parent = [-1] * n
    dist[source] = 0.0

    # Buckets: map bucket index -> set of vertices scheduled in that band
    B: dict[int, set[int]] = defaultdict(set)
    B[0].add(source)

    # Helper: find current minimum nonempty bucket index
    def min_nonempty_bucket_index():
        if not B:
            return None
        return min(B.keys())

    # Relax an edge u->v with weight w if it improves dist[v]
    def relax(u: int, v: int, w: float) -> bool:
        du = dist[u]
        dv = dist[v]
        nd = du + w
        if nd + 0.0 < dv:  # strict improvement
            dist[v] = nd
            parent[v] = u
            return True
        return False

    processed = set()  # for debug/inspection; not required for correctness

    while True:
        i = min_nonempty_bucket_index()
        if i is None:
            break

        # R will accumulate vertices whose **light‑edge** relaxations are closed
        R = set()
        S = set(B[i])  # working set for the current bucket band i
        del B[i]

        # 1) Repeatedly relax **light** edges (w ≤ δ) from S until no changes
        #    (This is the partial‑order expansion within a band.)
        while S:
            # Move S to R and collect newly improved vertices via light edges
            R.update(S)
            next_S = set()
            for u in S:
                for v, w in edges[u]:
                    if w <= delta and relax(u, v, w):
                        b = int(dist[v] // delta)
                        next_S.add(v)
                        # Ensure v is in the current or later bucket; for light
                        # relaxations we keep it in the working set instead of
                        # committing to a specific bucket immediately.
            S = next_S

        # 2) After closing light relaxations, relax **heavy** edges (w > δ) once
        heavy_enqueued = []
        for u in R:
            for v, w in edges[u]:
                if w > delta and relax(u, v, w):
                    b = int(dist[v] // delta)
                    heavy_enqueued.append((b, v))

        # Insert heavy‑relaxed vertices into their buckets
        for b, v in heavy_enqueued:
            B[b].add(v)

        processed.update(R)

    return dist, parent


# ---------------------------------------------------------------
# Convenience utilities
# ---------------------------------------------------------------

def reconstruct_path(parent: List[int], target: int) -> List[int]:
    """Reconstruct one shortest path to 'target' using the parent array."""
    path = []
    cur = target
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def choose_delta(edges: Adj, quantile: float = 0.5) -> float:
    """Heuristic: choose δ as a quantile of edge weights (default median)."""
    ws = [w for u in range(len(edges)) for _, w in edges[u] if w > 0]
    if not ws:
        return 1.0
    ws.sort()
    k = max(0, min(len(ws) - 1, int(quantile * (len(ws) - 1))))
    return max(1e-12, float(ws[k]))


# ---------------------------------------------------------------
# Example usage (run this module directly)
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Small demo graph (directed, non‑negative weights)
    n = 7
    edges: Adj = [[] for _ in range(n)]
    def add(u, v, w):
        edges[u].append((v, float(w)))

    add(0, 1, 2)
    add(0, 2, 1)
    add(1, 2, 1)
    add(1, 3, 2)
    add(2, 3, 3)
    add(2, 4, 13)
    add(3, 5, 1)
    add(4, 5, 1) # 5
    add(5, 6, 1)
    # add(5, 4, 1) 
    delta = choose_delta(edges)  # median weight
    dist, parent = sssp_delta_stepping(n, edges, source=0, delta=delta)

    print("delta=", delta)
    print("dist= ", dist)
    for t in range(n):
        if dist[t] < inf:
            print(f"path to {t}: ", reconstruct_path(parent, t))
        else:
            print(f"{t} unreachable")
