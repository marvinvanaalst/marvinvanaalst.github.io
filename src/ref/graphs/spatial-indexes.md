---
title: 'Spatial indexes & neighbour search'
description: 'Uniform grids and cell lists, k-d trees, octrees, BVHs, R-trees, space-filling curves and approximate nearest neighbours: which structure for which query, dimension and workload.'
aliases:
  [
    spatial index,
    nearest neighbor search,
    kNN search,
    range query,
    radius search,
    uniform grid,
    cell list,
    spatial hashing,
    k-d tree,
    ball tree,
    octree,
    quadtree,
    BVH,
    bounding volume hierarchy,
    SAH,
    R-tree,
    Morton order,
    Z-order,
    Hilbert curve,
    space-filling curve,
    locality sensitive hashing,
    approximate nearest neighbors,
    HNSW,
    FAISS,
    vector database
  ]
tags: [data structures, geometry, algorithms]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Particles with a fixed interaction radius** (MD, SPH, agent-based models): **uniform grid / cell lists** (or spatial hashing), $O(n)$ build and query.
- **Exact kNN / radius queries in low dimension** ($d \lesssim 10$–20): **k-d tree** (points) or ball tree.
- **Ray tracing, collision detection, objects with extent:** **BVH** (built with SAH or LBVH via Morton codes).
- **Adaptive density, hierarchical summaries** (FMM, Barnes–Hut, adaptive meshes): octrees/quadtrees.
- **Database-style rectangles with updates** (GIS): R-trees.
- **High-dimensional vectors** (embeddings): exact search degrades to brute force; use **approximate** nearest neighbours (HNSW graphs, IVF + product quantisation).
- **Memory locality:** sort data along a Morton/Hilbert curve; often a bigger win than the tree choice.

## Decision table

| Structure                | Build                               | Query                              | Dynamic updates | Best for                                 |
| ------------------------ | ----------------------------------- | ---------------------------------- | --------------- | ---------------------------------------- |
| Brute force              | none                                | $O(n)$ per query                   | trivial         | small $n$, high $d$, GPU batches         |
| Uniform grid / cell list | $O(n)$                              | $O(1)$ cells × neighbours          | cheap (rebin)   | fixed-radius queries, uniform density    |
| Spatial hash             | $O(n)$                              | as grid, unbounded domains         | cheap           | sparse/unbounded domains                 |
| k-d tree                 | $O(n\log n)$                        | ~$O(\log n)$ low $d$               | poor (rebuild)  | static points, kNN, low dimension        |
| Ball tree / cover tree   | $O(n\log n)$                        | better than k-d in moderate $d$    | poor            | non-Euclidean metrics, moderate $d$      |
| Octree / quadtree        | $O(n\log n)$                        | $O(\log n)$ adaptive               | moderate        | adaptive density, multipole methods      |
| BVH                      | $O(n\log n)$ (SAH) or $O(n)$ (LBVH) | ~$O(\log n)$ ray/overlap           | refit cheaply   | rays, collisions, objects with extent    |
| R-tree                   | $O(n\log n)$                        | $O(\log n)$ typical                | good            | rectangles, GIS, databases               |
| HNSW (graph-based ANN)   | $O(n\log n)$                        | sublinear, approximate             | inserts OK      | high-dimensional embeddings, high recall |
| IVF + PQ (FAISS)         | clustering                          | sublinear, approximate, compressed | batch           | billion-scale vectors, memory-limited    |
| LSH                      | $O(n)$                              | sublinear, approximate             | easy            | theoretical guarantees, Jaccard/cosine   |

## Variants

### Uniform grids and cell lists

1. Choose cell size ≥ interaction radius $r$.
2. Bin particles by cell index (counting sort by cell: two passes, contiguous storage).
3. For each particle, check its own and neighbouring cells ($3^d$ cells).

- Verlet neighbour lists (radius $r$ + skin) reused for several time steps.
- Non-uniform density wastes memory/time; use hashing or trees.

### k-d trees

- Recursively split points at the median along alternating (or widest) dimensions.
- **kNN query:** descend to the leaf, then backtrack, pruning subtrees whose bounding region is farther than the current $k$-th best distance.
- Degrades towards brute force as dimension grows (pruning fails beyond ~10–20 dimensions for uniform data).

### BVHs

- Tree of bounding boxes around groups of primitives.
- **SAH** (surface area heuristic) builds give the best ray-tracing performance; **LBVH**: sort primitives by Morton code and build in parallel in $O(n)$ (GPUs).
- Dynamic scenes: refit boxes bottom-up after motion, rebuild occasionally.

### Space-filling curves

- **Morton (Z-order):** interleave the bits of quantised coordinates; **Hilbert:** better locality, slightly more expensive.
- Sorting by curve index clusters nearby points in memory: better caching for neighbour loops, load balancing (domain decomposition), linear octrees.

### Approximate nearest neighbours

- **HNSW:** multi-layer proximity graph; greedy search from the top layer down; tunable recall/speed via `ef` and `M`.
- **IVF-PQ:** cluster vectors (inverted file), store residuals compressed with product quantisation; search a few clusters.
- Evaluate recall@k against exact search on a sample; choose parameters for the recall you need.

## Other lenses

- **Fast algorithms.** Trees enable Barnes–Hut and FMM for $N$-body problems and hierarchical matrices (see [hierarchical matrices & FMM](/ref/solve/hierarchical-matrices-fmm)).
- **Statistics / ML.** kNN classifiers, kernel density estimation, KSG mutual-information estimators, t-SNE/UMAP neighbour graphs and DBSCAN all rely on neighbour search (see [unsupervised learning](/ref/learn/unsupervised), [entropy & MI](/ref/info/entropy-kl-mi)).
- **Graphics.** BVHs and spatial grids accelerate ray tracing and collision detection; robust primitive tests sit at the leaves (see [geometry primitives](/ref/space/geometry-primitives)).

## Pitfalls & diagnostics

- **High dimensions:** distances concentrate, trees stop pruning; check whether brute force (vectorised/GPU) is faster.
- **Periodic boundaries** (simulation boxes) need wrapped cell neighbours / minimum-image distances.
- **Floating-point ties and duplicates** in tree construction (many identical points) degrade balance.
- **Measure:** build vs query cost trade-off depends on how many queries you run per build.
- **Metric mismatch:** k-d trees assume coordinate-aligned splits (Euclidean-like metrics); use ball/cover trees for general metrics.

## See also

- [Graph algorithms](/ref/graphs/graph-algorithms)
- [Hierarchical matrices & FMM](/ref/solve/hierarchical-matrices-fmm)
- [Geometry primitives](/ref/space/geometry-primitives)
- [Unsupervised learning](/ref/learn/unsupervised)

**References:**

- Samet, _Foundations of Multidimensional and Metric Data Structures_.
- Karras, "Maximizing parallelism in the construction of BVHs, octrees, and k-d trees", HPG (2012).
- Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs", IEEE TPAMI (2020).
