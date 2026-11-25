---
title: 'High-dimensional numerics'
description: 'The curse of dimensionality and how to dodge it: sparse grids (Smolyak), tensor-train methods, low effective dimension, and when Monte Carlo beats grids.'
aliases:
  [
    curse of dimensionality,
    sparse grids,
    Smolyak,
    combination technique,
    tensor train,
    TT-cross,
    low-rank tensor methods,
    effective dimension,
    ANOVA decomposition,
    Monte Carlo vs grids,
    high-dimensional integration,
    high-dimensional approximation,
    parameter scans,
    concentration of measure,
    active subspaces
  ]
tags: [high dimensions, approximation, integration, tensors]
order: 7
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Tensor grids* need $`n^d`$ points: 10 points per axis in 10 dimensions is $`10^{10}`$. Hopeless beyond $`d \approx 4`$–6.
- *Monte Carlo* converges as $`N^{-1/2}`$ regardless of $`d`$: the default for integrals in many dimensions. *QMC* approaches $`N^{-1}`$ when the effective dimension is low.
- *Sparse grids (Smolyak):* $`O(n\log^{d-1}n)`$ points for functions with bounded mixed derivatives; excellent for smooth problems up to \~10–20 dimensions.
- *Tensor trains:* if the function or tensor has low TT ranks, storage and work scale linearly in $`d`$; build from samples with TT-cross.
- *Find structure first:* low effective dimension (few important inputs or low-order interactions), active subspaces (few important directions), additivity. Most practical high-dimensional success comes from exploiting it.

## Decision table

| Method | Cost for accuracy $`\varepsilon`$ | Requires | Use when |
| --- | --- | --- | --- |
| Tensor-product grids | $`\varepsilon^{-d/s}`$ | smoothness $`s`$ | $`d \le 3`$–4 |
| Monte Carlo | $`\varepsilon^{-2}`$ | finite variance | any $`d`$, rough integrands |
| Quasi-Monte Carlo | \~$`\varepsilon^{-1}`$ (log factors) | low effective dimension, smoothness | integration, moderate–high $`d`$ |
| Sparse grids | $`\varepsilon^{-1/s}\lvert\log\varepsilon\rvert^{(d-1)(\dots)}`$ | bounded mixed derivatives | smooth, $`d \lesssim 10`$–20 |
| Adaptive / dimension-adaptive sparse grids | depends on anisotropy | anisotropic importance | some inputs much more important |
| Tensor train (TT-cross) | $`O(dnr^2)`$ samples | low TT ranks | parametric PDEs, high-order tensors, CME |
| Active subspaces | gradient samples + low-dim method | few dominant directions | gradients available, ridge-like functions |
| ANOVA / HDMR truncation | sum of low-order component functions | low-order interactions | additive-ish models |
| Neural networks | data-hungry | compositional structure | lots of data, very high $`d`$ |

## Variants

### Why grids fail

- Error of a tensor rule with $`n`$ points per dimension: $`O(n^{-s}) = O(N^{-s/d})`$ with $`N = n^d`$. Accuracy per point collapses with $`d`$.
- Volume concentrates near the boundary and corners of hypercubes; nearest-neighbour distances become nearly equal (concentration of measure). Intuition from 2D/3D fails.

### Sparse grids (Smolyak)

- Combine 1D rules $`U^{(l)}`$ of increasing level with difference operators $`\Delta^{(l)} = U^{(l)} - U^{(l-1)}`$:
  $`A(q, d) = \sum_{\lVert\mathbf l\rVert_1 \le q + d - 1}\Delta^{(l_1)}\otimes\dots\otimes\Delta^{(l_d)}`$.
- Keeps only tensor products whose total level is small: drops high-order interactions at fine resolution.
- *Nested 1D rules* (Clenshaw–Curtis, Gauss–Patterson, Leja) keep the point count low.
- *Combination technique:* the same approximation as a signed sum of small full tensor grids; easy to parallelise and to reuse existing solvers.
- *Dimension-adaptive:* refine in directions where surpluses are large (Gerstner–Griebel).

### Tensor trains

- Represent $`f(x_1, \dots, x_d)`$ on a grid as a TT: $`G_1(x_1)\cdots G_d(x_d)`$ with ranks $`r_k`$.
- *TT-cross:* build from $`O(dnr^2)`$ adaptively chosen function evaluations (maximum-volume submatrices), without touching the full grid.
- Arithmetic (sums, products, integration, solving linear systems with DMRG/AMEn) stays in TT format with rank truncation.
- Applications: parametric PDEs, chemical master equation, high-dimensional densities, quantum chemistry (MPS).

### Effective dimension and active subspaces

- *Truncation dimension:* number of leading inputs carrying most variance. *Superposition dimension:* the order of interactions that matter. Compute via Sobol indices (see [global sensitivity](/ref/dynamics/global-sensitivity)).
- *Active subspaces:* eigenvectors of $`C = \mathbb E[\nabla f\nabla f^\top]`$; if a few eigenvalues dominate, $`f(x) \approx g(W_1^\top x)`$ with low-dimensional $`g`$. Estimate $`C`$ from gradient samples.

## Other lenses

- *Probability.* Monte Carlo's dimension independence comes from the CLT; the constant (variance) still depends on the problem, which is where variance reduction and QMC help (see [variance reduction](/ref/random/variance-reduction)).
- *Linear algebra.* Tensor trains are repeated SVDs of unfoldings; ranks measure correlation between groups of variables (see [low-rank](/ref/approx/low-rank)).
- *Learning theory.* Neural networks escape the curse only for functions with compositional or low-dimensional structure; no method beats the curse for generic smooth functions in high dimension.

## Pitfalls & diagnostics

- *Assuming smoothness* that isn't there (thresholds, bifurcations): sparse grids and polynomial methods degrade badly; Monte Carlo doesn't care.
- *Parameter scans in many dimensions:* don't grid; use Sobol sequences, LHS or adaptive sampling.
- *Sparse grid negative weights* can amplify noise; prefer non-negative variants for noisy integrands.
- *TT rank growth* during arithmetic: truncate often with controlled tolerance; monitor ranks.
- *Check effective dimension early;* it decides which method can work at all.

## See also

- [Quasi-Monte Carlo](/ref/random/quasi-monte-carlo)
- [Low-rank approximation](/ref/approx/low-rank)
- [Quadrature](/ref/integrate/quadrature)
- [Surrogate models](/ref/approx/surrogates)
- [Global sensitivity analysis](/ref/dynamics/global-sensitivity)

*References:*

- Bungartz & Griebel, "Sparse grids", Acta Numerica (2004).
- Oseledets & Tyrtyshnikov, "TT-cross approximation for multidimensional arrays", Linear Algebra Appl. (2010).
- Constantine, _Active Subspaces_.
