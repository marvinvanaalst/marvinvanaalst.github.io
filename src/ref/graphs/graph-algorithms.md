---
title: 'Graph algorithms'
description: 'BFS/DFS, topological sort, strongly connected components, shortest paths (Dijkstra, Bellman–Ford, A*), minimum spanning trees, max-flow/min-cut, matchings and union-find: the everyday toolbox with costs.'
aliases:
  [
    graph algorithms,
    BFS,
    breadth-first search,
    DFS,
    depth-first search,
    topological sort,
    strongly connected components,
    Tarjan,
    Kosaraju,
    shortest path,
    Dijkstra,
    Bellman-Ford,
    A*,
    Floyd-Warshall,
    Johnson,
    minimum spanning tree,
    Kruskal,
    Prim,
    union-find,
    disjoint set,
    max flow,
    min cut,
    Ford-Fulkerson,
    Edmonds-Karp,
    Dinic,
    push-relabel,
    bipartite matching,
    Hopcroft-Karp,
    Hungarian algorithm,
    PageRank,
    adjacency list,
    CSR graph
  ]
tags: [graphs, algorithms, data structures]
order: 1
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Representation:* adjacency lists (or CSR arrays) for sparse graphs; adjacency matrices only for dense or small graphs, or for linear-algebra formulations.
- *Traversal:* BFS (unweighted shortest paths, levels), DFS (cycles, topological order, SCCs, articulation points). Both $`O(V + E)`$.
- *Shortest paths:* unweighted → BFS; non-negative weights → *Dijkstra* with a binary heap $`O((V + E)\log V)`$; negative weights → Bellman–Ford $`O(VE)`$ (detects negative cycles); DAG → relax in topological order $`O(V + E)`$; all pairs → Floyd–Warshall $`O(V^3)`$ (dense) or Johnson (sparse); goal-directed → A\* with an admissible heuristic.
- *Connectivity / merging:* *union-find* with path compression and union by rank, practically $`O(1)`$ per operation.
- *MST:* Kruskal (sort edges + union-find) or Prim (heap). *Flows / cuts / matchings:* Dinic or push-relabel for max-flow; Hopcroft–Karp for bipartite matching; Hungarian for assignment.

## Decision table

| Problem | Algorithm | Complexity | Notes |
| --- | --- | --- | --- |
| Reachability, unweighted distances | BFS | $`O(V + E)`$ | queue; levels = hop distances |
| Cycle detection, DFS order | DFS | $`O(V + E)`$ | iterative to avoid stack overflow |
| Topological sort | Kahn (in-degrees) or DFS postorder | $`O(V + E)`$ | fails iff the graph has a cycle |
| Strongly connected components | Tarjan / Kosaraju | $`O(V + E)`$ | condensation is a DAG |
| Shortest paths, weights ≥ 0 | Dijkstra (binary heap) | $`O((V + E)\log V)`$ | no negative weights |
| Shortest paths, negative weights | Bellman–Ford | $`O(VE)`$ | detects negative cycles |
| Shortest paths in a DAG | DP in topological order | $`O(V + E)`$ | also longest paths (critical path) |
| Point-to-point with heuristic | A\* | depends on heuristic | admissible (and consistent) heuristic for optimality |
| All-pairs shortest paths | Floyd–Warshall / Johnson | $`O(V^3)`$ / $`O(VE\log V)`$ | dense / sparse |
| Connected components (dynamic merges) | union-find | \~$`O(\alpha(n))`$ per operation | also Kruskal, unification |
| Minimum spanning tree | Kruskal / Prim | $`O(E\log E)`$ / $`O(E\log V)`$ | Kruskal for sparse, edges sorted |
| Max flow / min cut | Dinic / push-relabel | $`O(V^2E)`$ / $`O(V^3)`$, fast in practice | min cut = max flow (duality) |
| Bipartite matching | Hopcroft–Karp | $`O(E\sqrt V)`$ | unweighted |
| Assignment (weighted bipartite) | Hungarian / auction | $`O(n^3)`$ | optimal transport special case |
| Ranking / stationary distribution | PageRank (power iteration) | $`O(E)`$ per iteration | Markov chain on the graph |

## Variants

### Representations

- *Adjacency list / CSR:* offsets + neighbour indices (+ weights); cache-friendly, $`O(V + E)`$ memory. Same as CSR sparse matrices (see [sparse matrices](/ref/solve/sparse-matrices)).
- *Edge list:* good for sorting (Kruskal) and building other formats.
- *Adjacency matrix:* $`O(V^2)`$; enables linear-algebra formulations (powers count walks, Laplacian eigenvectors give clustering).

### Dijkstra

1. Distances $`d[s] = 0`$, others $`\infty`$; priority queue with $`(0, s)`$.
2. Pop the vertex $`u`$ with smallest tentative distance (skip stale entries).
3. For each edge $`(u, v, w)`$: if $`d[u] + w \lt d[v]`$, set $`d[v] = d[u] + w`$, record predecessor, push $`(d[v], v)`$.
4. Repeat until the queue is empty (or the target is popped).

- A\* = Dijkstra with priority $`d[v] + h(v)`$; with a consistent heuristic it expands far fewer vertices.

### Union-find

- `find` with path compression, `union` by rank/size: amortised inverse-Ackermann time.
- Uses: connected components under edge insertions, Kruskal, percolation/cluster labelling, type unification (see [type inference](/ref/programs/type-inference-unification)), congruence closure (see [e-graphs](/ref/programs/term-rewriting-egraphs)).

### SCCs and DAGs

- *Tarjan:* one DFS with low-link values and a stack.
- Condense SCCs to a DAG; process in topological order: dependency resolution, fixpoint iteration per SCC (dataflow, DAE structural analysis, build systems).

### Flows and cuts

- *Max-flow min-cut theorem:* maximum flow value = minimum cut capacity (LP duality).
- Augmenting paths (Edmonds–Karp), blocking flows (Dinic), push-relabel. Applications: bipartite matching, image segmentation (graph cuts), project selection, network reliability.

## Other lenses

- *Linear algebra.* BFS = sparse matrix–vector products over the boolean semiring; shortest paths = min-plus products; spectral graph theory uses Laplacian eigenvectors for partitioning and clustering (see [algebraic structures](/ref/programs/algebraic-structures), [eigenproblems](/ref/solve/eigenproblems)).
- *Optimisation.* Shortest paths, flows and matchings are LPs with totally unimodular constraint matrices (integral optima); assignment is discrete optimal transport (see [constrained optimisation](/ref/optimise/constrained)).
- *Probability.* Random walks, PageRank and Markov chains on graphs (see [stochastic processes](/ref/dynamics/stochastic-processes)).
- *Sparse direct solvers.* Fill-reducing orderings (minimum degree, nested dissection) are graph algorithms (see [sparse matrices](/ref/solve/sparse-matrices)).

## Pitfalls & diagnostics

- *Negative weights with Dijkstra* give wrong answers silently.
- *Recursion depth* in DFS on large graphs; use an explicit stack.
- *Priority-queue variants:* decrease-key vs lazy insertion (lazy is simpler and usually faster).
- *Directed vs undirected* confusion (adding both edge directions for undirected graphs).
- *Big graphs:* prefer CSR and libraries (igraph, networkx for prototyping, graph-tool, GraphBLAS) over pointer-heavy structures.

## See also

- [Spatial indexes](/ref/graphs/spatial-indexes)
- [Algebraic structures](/ref/programs/algebraic-structures)
- [Sparse matrices](/ref/solve/sparse-matrices)
- [Stochastic processes](/ref/dynamics/stochastic-processes)

*References:*

- Cormen, Leiserson, Rivest & Stein, _Introduction to Algorithms_ (4th ed.).
- Sedgewick & Wayne, _Algorithms_ (4th ed.).
- Ahuja, Magnanti & Orlin, _Network Flows: Theory, Algorithms, and Applications_.
