---
title: 'Hierarchical matrices & fast multipole'
description: 'Exploiting low-rank off-diagonal blocks of dense kernel matrices: FMM, Barnes–Hut, H- and H²-matrices, HSS, and fast direct solvers in near-linear time.'
aliases:
  [
    fast multipole method,
    FMM,
    Barnes-Hut,
    treecode,
    H-matrix,
    hierarchical matrix,
    H2-matrix,
    HSS,
    HODLR,
    butterfly,
    ACA,
    adaptive cross approximation,
    kernel matrix,
    boundary element method,
    BEM,
    N-body,
    Ewald summation
  ]
tags: [linear algebra, fast algorithms, kernel methods, N-body]
order: 9
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Dense matrices from smooth kernels $K(x_i, y_j)$ (gravity, electrostatics, BEM, Gaussian-process covariances) have **low-rank off-diagonal blocks** for well-separated clusters.
- **Matrix-vector products:** Barnes–Hut $O(n\log n)$, FMM $O(n)$, at controllable accuracy.
- **Factorisation / solves / inverses:** H-matrices ($O(n\log^2 n)$), H²/HSS ($O(n)$ in favourable cases). HODLR is the simplest special case.
- Periodic systems (molecular dynamics, plasmas): Ewald summation with FFT (PME/P3M), $O(n\log n)$.
- The kernel must be smooth away from the diagonal; oscillatory kernels (high-frequency Helmholtz) need directional or butterfly variants.

## Decision table

| Method                  | Operation           | Cost                       | Kernel requirements                              | Use when                                      |
| ----------------------- | ------------------- | -------------------------- | ------------------------------------------------ | --------------------------------------------- |
| Direct summation        | matvec              | $O(n^2)$                   | none                                             | $n \lesssim 10^4$, GPUs push this higher      |
| Barnes–Hut treecode     | matvec              | $O(n\log n)$               | multipole expansion (or none: centre of mass)    | N-body, moderate accuracy                     |
| FMM (analytic)          | matvec              | $O(n)$                     | analytic expansions (Laplace, Helmholtz, Stokes) | high accuracy, huge $n$                       |
| Kernel-independent FMM  | matvec              | $O(n)$                     | kernel evaluations only                          | general smooth kernels                        |
| PME / P3M (Ewald + FFT) | matvec, periodic    | $O(n\log n)$               | Coulomb-like, periodic box                       | molecular dynamics                            |
| HODLR                   | solve, logdet       | $O(n\log^2 n)$             | low-rank off-diagonal (1D-like ordering)         | GPs in 1D, time series                        |
| H-matrix (ACA)          | matvec, LU, inverse | $O(n\log^2 n)$ storage/ops | asymptotically smooth                            | BEM, dense solvers, preconditioners           |
| H² / HSS                | matvec, solve       | $O(n)$                     | nested bases                                     | best constants, fast direct solvers           |
| Butterfly               | matvec              | $O(n\log n)$               | oscillatory (Fourier-like)                       | high-frequency scattering, special transforms |

## Variants

### Admissibility and low rank

- Partition points into a tree of clusters (octree, k-d tree).
- A block (cluster $s$, cluster $t$) is **admissible** if $\min(\operatorname{diam} s, \operatorname{diam} t) \le \eta\,\operatorname{dist}(s, t)$. Such blocks are approximated by rank-$k$ factors $UV^\top$ with $k \sim \log^d(1/\varepsilon)$.
- Inadmissible blocks (near field) are stored densely at the leaves.

### Barnes–Hut and FMM

- **Barnes–Hut:** for each target, walk the tree; if a cell is far enough (opening angle $\theta$), use its multipole (or just centre-of-mass) approximation; otherwise open it.
- **FMM** adds local expansions so far-field interactions are aggregated on both sides:
  1. Upward pass: particle → multipole expansions (P2M), merge children to parents (M2M).
  2. Interaction lists: translate multipole to local expansions between well-separated cells (M2L), the dominant cost.
  3. Downward pass: shift local expansions to children (L2L), evaluate at targets (L2P).
  4. Near field: direct summation (P2P).
- Accuracy set by expansion order $p$; cost $O(p^2 n)$ to $O(p^3 n)$ in 3D depending on translation operators.

### Hierarchical matrices

- **Construction:** adaptive cross approximation (ACA) builds $UV^\top$ from a few rows and columns of each admissible block without forming it; or interpolation / proxy-point methods.
- **Arithmetic:** add, multiply, LU and inverse in H-format with recompression (truncated SVD) after each operation. An approximate H-LU makes an excellent preconditioner.
- **H² / HSS:** bases are nested across levels (parent bases from children), removing the log factors.

### Fast direct solvers

- Recursive skeletonisation / interpolative decompositions: compress the far field, eliminate, recurse. Factor once in $O(n)$–$O(n^{3/2})$, solve in $O(n)$, also log-determinants.
- Particularly strong for integral equations (BEM) where iterative methods may need many iterations.

## Other lenses

- **Linear algebra.** These are data-sparse rather than sparse formats: structured low-rank approximation of blocks (see [low-rank](/ref/approx/low-rank)). The ranks relate to separation in the kernel: $K(x, y) \approx \sum_{k} u_k(x)v_k(y)$.
- **Gaussian processes.** Covariance matrices of smooth kernels are H-matrices; HODLR/HSS solvers give exact-ish likelihoods and log-determinants in near-linear time (see [Gaussian processes](/ref/approx/gaussian-processes)).
- **PDEs.** Inverses of elliptic PDE operators (Green's functions) are also hierarchically low rank, which is why H-LU of sparse FE matrices works as a preconditioner and why nested dissection fronts compress.

## Pitfalls & diagnostics

- **Geometry matters:** clustered or highly non-uniform point sets need adaptive trees; poor partitions blow up ranks.
- **High-frequency kernels** have ranks growing with frequency; standard H-matrices degenerate. Use directional or butterfly methods.
- **Accuracy vs cost:** errors are relative to the whole matrix; check the relative matvec error against direct summation on a subset.
- **Implementation effort is high;** use established libraries (e.g. FMM3D, ExaFMM, H2Lib, STRUMPACK, hm-toolbox) rather than writing your own for production.

## See also

- [Low-rank approximation](/ref/approx/low-rank)
- [Gaussian processes](/ref/approx/gaussian-processes)
- [Spatial indexes](/ref/graphs/spatial-indexes)
- [Multigrid & domain decomposition](/ref/solve/multigrid-domain-decomposition)

**References:**

- Greengard & Rokhlin, "A fast algorithm for particle simulations", J. Comput. Phys. (1987).
- Hackbusch, _Hierarchical Matrices: Algorithms and Analysis_.
- Martinsson, _Fast Direct Solvers for Elliptic PDEs_.
