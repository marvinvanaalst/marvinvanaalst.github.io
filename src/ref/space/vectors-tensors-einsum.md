---
title: 'Vectors, tensors & einsum'
description: 'Index notation and einsum, broadcasting rules, memory layout and strides, contraction order, and the covariant/contravariant distinction when it actually matters.'
aliases:
  [
    tensor,
    einsum,
    Einstein summation,
    index notation,
    broadcasting,
    strides,
    memory layout,
    row-major,
    column-major,
    contraction,
    tensor contraction order,
    opt_einsum,
    outer product,
    Kronecker product,
    batched matrix multiply,
    reshape,
    transpose,
    views,
    BLAS,
    GEMM,
    covariant,
    contravariant,
    dual vector
  ]
tags: [linear algebra, arrays, performance]
order: 1
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *einsum* writes array operations in index notation: `ij,jk->ik` is matrix multiply, `bij,bjk->bik` batched matmul, `ii->` trace. Unambiguous and shape-checked.
- *Broadcasting:* align shapes from the right; dimensions of size 1 stretch. Make singleton axes explicit to avoid silent mistakes.
- *Performance* is about memory: contiguous access in the storage order (row-major in C/NumPy/PyTorch, column-major in Julia/Fortran/MATLAB), fusing operations, and mapping work onto BLAS-3 (GEMM).
- *Contraction order* for multi-tensor products can change cost by orders of magnitude; let `opt_einsum`-style optimisers choose.
- Upper/lower indices (contravariant/covariant) matter with non-Euclidean metrics and changes of basis (differential geometry, physics); in plain numerics with orthonormal bases they coincide.

## Decision table

| Operation | einsum | Cost | Notes |
| --- | --- | --- | --- |
| Dot product | `i,i->` | $`O(n)`$ | memory-bound |
| Outer product | `i,j->ij` | $`O(mn)`$ | often avoidable (keep factors) |
| Matrix–vector | `ij,j->i` | $`O(mn)`$ | BLAS-2, memory-bound |
| Matrix–matrix | `ij,jk->ik` | $`O(mnk)`$ | BLAS-3, compute-bound: fast |
| Batched matmul | `bij,bjk->bik` | $`O(bmnk)`$ | GPUs love it |
| Trace / diagonal | `ii->` / `ii->i` | $`O(n)`$ | don't form the full product first |
| Trace of product | `ij,ji->` | $`O(n^2)`$ | instead of $`O(n^3)`$ matmul then trace |
| Bilinear form $`x^\top Ay`$ | `i,ij,j->` | $`O(n^2)`$ | order: $`Ay`$ first |
| Kronecker-structured product $`(A\otimes B)\operatorname{vec}(X)`$ | `ai,bj,ij->ab` style | $`O(n^3)`$ instead of $`O(n^4)`$ | never form $`A\otimes B`$ |
| Attention scores | `bhqd,bhkd->bhqk` | $`O(bhq kd)`$ | then softmax, `bhqk,bhkd->bhqd` |

## Variants

### Index notation and einsum

- Repeated indices are summed (Einstein convention); output indices listed after `->`.
- Implicit vs explicit output: always write the output explicitly for clarity.
- Equivalent tools: `einsum` (NumPy, PyTorch, JAX), `@tullio`/`@einsum` (Julia), `einops` (`rearrange`, `reduce`, `repeat`) for reshapes with named axes.

### Broadcasting

- Shapes are compared from the trailing dimension; equal or 1 is compatible; missing leading dimensions are treated as 1.
- `x[:, None] - y[None, :]` gives all pairwise differences (`(n,1)` and `(1,m)` → `(n,m)`).
- Silent broadcasting bugs: `(n,)` vs `(n,1)` shapes producing `(n,n)` results. Assert shapes.

### Memory layout and strides

- An array = buffer + shape + strides (bytes to move per index step). Transposes, slices and broadcasting are often free *views* that change strides.
- Iterate over the contiguous axis innermost (last axis in row-major, first in column-major).
- Non-contiguous views passed to BLAS or GPU kernels may trigger copies; call `contiguous()` / `copy()` deliberately.
- *Struct of arrays vs array of structs:* SoA is usually better for SIMD/GPU.

### Contraction order

- For $`ABCv`$: $`A(B(Cv))`$ costs $`O(n^2)`$ matvecs; $`((AB)C)v`$ costs $`O(n^3)`$.
- Multi-tensor networks: choosing the pairwise contraction order is NP-hard in general; greedy and dynamic-programming optimisers (`opt_einsum`, `cotengra`) find good paths. Reuse the path for repeated contractions.

### Covariant and contravariant

- Vectors $`v^i`$ transform with the inverse of basis changes; covectors (gradients, linear functionals) $`w_i`$ transform with the basis change. Their pairing $`w_iv^i`$ is invariant.
- Metric tensor $`g_{ij}`$ raises/lowers indices. In Euclidean orthonormal coordinates $`g = I`$, so the distinction disappears — which is why numerics can usually ignore it, and why gradients transform differently from displacements under rescaling (preconditioning, natural gradient).

## Other lenses

- *Linear algebra.* einsum covers products, traces, transposes and Kronecker identities; $`\operatorname{vec}(AXB) = (B^\top\otimes A)\operatorname{vec}(X)`$ is an einsum (see [matrix calculus](/ref/diff/matrix-calculus)).
- *Compilers.* Tensor compilers (XLA, TVM, TACO, polyhedral tools) turn index expressions into fused, tiled loops (see [polyhedral loop transforms](/ref/programs/polyhedral-loop-transforms)).
- *Physics.* Tensor networks (MPS/TT) are big einsum graphs; contraction order is the central computational problem (see [low-rank](/ref/approx/low-rank)).

## Pitfalls & diagnostics

- *Memory blow-up* from intermediate broadcast arrays (e.g. `(n, m, d)` pairwise differences); use specialised kernels (e.g. distance matrices via $`\lVert x\rVert^2 + \lVert y\rVert^2 - 2x^\top y`$) or chunking.
- *Loop order* against the storage order costs 10× or more.
- *Explicit loops in Python* over large arrays; vectorise or JIT-compile.
- *Reshape vs transpose confusion:* reshape reinterprets the buffer in storage order; it doesn't move axes.
- *Assert shapes* at function boundaries; most numerical array bugs are shape bugs.

## See also

- [Matrix calculus](/ref/diff/matrix-calculus)
- [Rotations](/ref/space/rotations)
- [Polyhedral loop transforms](/ref/programs/polyhedral-loop-transforms)
- [Low-rank approximation](/ref/approx/low-rank)

*References:*

- Harris et al., "Array programming with NumPy", Nature (2020).
- Rogozhnikov, "Einops: clear and reliable tensor manipulations with Einstein-like notation", ICLR (2022).
- Gray & Kourtis, "Hyper-optimized tensor network contraction", Quantum (2021).
