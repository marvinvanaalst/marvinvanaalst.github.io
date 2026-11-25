---
title: 'Sparse matrices'
description: 'Storage formats, fill-in, fill-reducing orderings and symbolic vs numeric factorisation: when sparse direct solvers win and what they cost.'
aliases:
  [
    CSR,
    CSC,
    COO,
    ELL,
    BSR,
    fill-in,
    AMD,
    COLAMD,
    nested dissection,
    METIS,
    reverse Cuthill-McKee,
    RCM,
    symbolic factorisation,
    supernodal,
    multifrontal,
    sparsity pattern
  ]
tags: [linear algebra, sparse, data structures]
order: 2
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- Store in *CSC/CSR* for computation; build with *COO* (triplets) and convert.
- Sparse direct solvers cost what the *fill-in* costs. Always use a fill-reducing ordering: AMD for general use, nested dissection (METIS) for large 2D/3D meshes.
- Split factorisation into *symbolic* (pattern, once) and *numeric* (values, every time). Newton iterations and implicit ODE steps reuse the symbolic phase.
- Direct is unbeatable for 2D problems up to millions of unknowns and for many right-hand sides; for large 3D problems memory explodes and iterative methods take over.

## Decision table

| Format | Layout | Fast at | Slow at | Use when |
| --- | --- | --- | --- | --- |
| COO | (row, col, value) triplets | assembly, appending, duplicates summed | arithmetic, lookup | building matrices (FE assembly) |
| CSR | row pointers + column indices + values | row slicing, $`Ax`$ | column slicing, inserting | SpMV, row-oriented algorithms (C, Python) |
| CSC | column pointers + row indices + values | column slicing, $`A^\top x`$, factorisations | row slicing, inserting | direct solvers (SuiteSparse), Julia, MATLAB |
| BSR / BCSR | CSR of dense $`b\times b`$ blocks | multi-component PDEs, SIMD | irregular patterns | systems with several unknowns per node |
| ELL / SELL-C-σ | padded fixed width per row | GPU/SIMD SpMV | rows of very different length | GPU kernels |
| DIA | stored diagonals | banded, stencils on structured grids | anything irregular | structured-grid stencils |
| Matrix-free | function computing $`Ax`$ | memory, huge problems | factorisation (impossible) | Krylov methods, Jacobian-free Newton |

| Ordering | Goal | Cost | Use when |
| --- | --- | --- | --- |
| AMD / COLAMD | minimise fill (greedy) | \~linear in nnz | default for Cholesky/LU |
| Nested dissection (METIS) | minimise fill and allow parallelism | higher, graph partitioning | large 2D/3D meshes |
| RCM | reduce bandwidth / profile | linear | banded solvers, cache locality, ILU |
| Natural | none | free | already banded or tiny |

## Variants

### Fill-in

- Eliminating a variable connects all of its remaining neighbours in the matrix graph. New non-zeros are *fill*.
- Fill depends entirely on the elimination order. Arrow matrix example: eliminating the hub first fills everything, eliminating it last fills nothing.
- *Scaling for grid problems* with nested dissection:
  - 2D ($`n`$ unknowns): fill $`O(n\log n)`$, flops $`O(n^{3/2})`$.
  - 3D: fill $`O(n^{4/3})`$, flops $`O(n^2)`$. This is why 3D goes iterative.

### Factorisation phases

1. *Ordering:* compute a permutation $`P`$ (AMD, ND).
2. *Symbolic analysis:* elimination tree, column counts, supernodes. Depends only on the pattern.
3. *Numeric factorisation:* the actual $`LL^\top`$ / $`LU`$ / $`LDL^\top`$. For LU, partial pivoting may alter the pattern; threshold pivoting balances stability and sparsity.
4. *Solve:* triangular solves, cost proportional to nnz of the factors.

- *Supernodal / multifrontal* methods group columns with the same pattern into dense blocks and use BLAS-3, which is where performance comes from (CHOLMOD, UMFPACK, MUMPS, PARDISO).

### Sparse matrix–vector product (SpMV)

- Cost: $`2\,\mathrm{nnz}`$ flops, but *memory-bound*: \~12 bytes moved per non-zero (CSR, Float64 + Int32 index), so \~0.15 flop/byte.
- Improvements: reordering for locality (RCM), blocking (BSR), smaller index types, SELL-C-σ on GPUs.

### Sparsity pattern of Jacobians

- *Detection:* from the model's structure (symbolic tracing, operator overloading) instead of dense finite differences.
- *Colouring:* structurally orthogonal columns can be perturbed together, so a sparse Jacobian needs only $`\chi`$ (number of colours) evaluations instead of $`n`$. See [automatic differentiation](/ref/diff/automatic-differentiation).

## Other lenses

- *Graphs.* A symmetric sparse matrix is a graph; elimination is vertex elimination; the filled graph is chordal; minimum fill is NP-hard and related to treewidth. Nested dissection is recursive graph bisection by small separators.
- *Probability.* Sparse precision matrices are Gaussian Markov random fields: the pattern encodes conditional independence, and sparse Cholesky gives exact sampling and log-determinants.
- *Compilers.* Sparse kernels are irregular loops over indirection arrays; sparse tensor compilers (TACO) generate them from index notation and format descriptions.

## Pitfalls & diagnostics

- *Inserting into CSR/CSC* one element at a time is $`O(\mathrm{nnz})`$ per insertion. Assemble in COO, convert once.
- *Accidental densification:* $`A^{-1}`$, $`A^\top A`$ with a dense row, or adding a dense rank-1 term. Use Woodbury instead of forming the dense matrix.
- *Explicit zeros* stored in the pattern still cost memory and flops; but keeping them lets you reuse the symbolic factorisation when values change.
- *Measure fill:* nnz(L) / nnz(A). Ratios above \~50–100 suggest a better ordering or an iterative method.
- *Index type overflow:* Int32 indices limit nnz to $`2^{31}`$.

## See also

- [Solve Ax = b](/ref/solve/linear-systems)
- [Krylov methods & preconditioning](/ref/solve/krylov-preconditioning)
- [Multigrid & domain decomposition](/ref/solve/multigrid-domain-decomposition)
- [Graph algorithms](/ref/graphs/graph-algorithms)

*References:*

- Davis, _Direct Methods for Sparse Linear Systems_.
- Davis, Rajamanickam & Sid-Lakhdar, "A survey of direct methods for sparse linear systems", Acta Numerica (2016).
- Saad, _Iterative Methods for Sparse Linear Systems_, ch. 3.
