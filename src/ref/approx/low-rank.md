---
title: 'Low-rank approximation'
description: 'SVD and the Eckart–Young theorem, truncated and randomised SVD, PCA, NMF, and tensor decompositions (CP, Tucker, tensor train).'
aliases:
  [
    SVD,
    singular value decomposition,
    truncated SVD,
    randomized SVD,
    Eckart-Young,
    PCA,
    principal component analysis,
    NMF,
    non-negative matrix factorization,
    CUR,
    interpolative decomposition,
    matrix completion,
    tensor decomposition,
    CP decomposition,
    PARAFAC,
    Tucker,
    HOSVD,
    tensor train,
    TT,
    MPS,
    low-rank adaptation,
    LoRA
  ]
tags: [linear algebra, dimensionality reduction, tensors]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Best rank-$`k`$ approximation* in spectral and Frobenius norm: truncated SVD (Eckart–Young). Error = the discarded singular values.
- *Dense, need all singular values:* LAPACK SVD ($`O(mn^2)`$). *Need top $`k`$ of a large matrix:* *randomised SVD* ($`O(mnk)`$, a few passes over the data) or Lanczos (Golub–Kahan).
- *PCA* = SVD of the centred data matrix. Scale variables first if units differ.
- *Non-negative parts-based factors:* NMF (no uniqueness, local optima). *Interpretable rows/columns:* CUR / interpolative decomposition.
- *Tensors:* CP (sum of rank-1, compact but ill-posed), Tucker / HOSVD (core + factor matrices, stable), *tensor train* (scales linearly in the order $`d`$; the workhorse for high-dimensional problems).

## Decision table

| Method | Cost | Optimal? | Use when |
| --- | --- | --- | --- |
| Full SVD (dense) | $`O(mn\min(m, n))`$ | yes | moderate dense matrices |
| Truncated SVD via Lanczos | $`O(k)`$ matvecs × iterations | yes (to tolerance) | sparse/large, few singular values |
| Randomised SVD | $`O(mnk)`$ + small SVD, 2–4 passes | near-optimal (power iterations help) | large dense or streaming, moderate accuracy |
| PCA (via SVD) | as SVD | yes (variance) | dimensionality reduction, decorrelation |
| Incremental / streaming PCA | $`O(nk)`$ per sample | approximate | data that doesn't fit in memory |
| NMF (multiplicative updates, HALS) | iterative, $`O(mnk)`$ per sweep | local optimum | non-negative data (spectra, counts, images) |
| CUR / interpolative decomposition | $`O(mnk)`$ | near-optimal | interpretability, sparse structure preserved |
| Matrix completion (nuclear norm, ALS) | iterative | depends | missing entries (recommenders) |
| CP / PARAFAC (ALS) | iterative | NP-hard, may be ill-posed | interpretable multilinear components (chemometrics) |
| Tucker / HOSVD | SVDs of unfoldings | quasi-optimal | compressing moderate-order tensors |
| Tensor train (TT-SVD, TT-cross) | $`O(dnr^3)`$ | quasi-optimal | high-order tensors, parametric functions |

## Variants

### SVD and Eckart–Young

- $`A = U\Sigma V^\top`$, $`\sigma_1 \ge \sigma_2 \ge \dots \ge 0`$.
- *Eckart–Young–Mirsky:* $`A_k = U_k\Sigma_kV_k^\top`$ minimises $`\lVert A - B\rVert`$ over rank-$`k`$ $`B`$; errors $`\lVert A - A_k\rVert_2 = \sigma_{k+1}`$, $`\lVert A - A_k\rVert_F^2 = \sum_{i \gt k}\sigma_i^2`$.
- *Choosing $`k`$:* energy fraction, a gap in the spectrum, or optimal hard thresholding for noisy data (Gavish–Donoho: $`\tau \approx 2.858\,\sigma_{\text{med}}`$ for square matrices).

### Randomised SVD (Halko–Martinsson–Tropp)

1. Draw a Gaussian test matrix $`\Omega \in \mathbb R^{n\times(k + p)}`$ (oversampling $`p \approx 5`$–10).
2. Sample the range: $`Y = (AA^\top)^qA\Omega`$ ($`q = 1`$–2 power iterations for slowly decaying spectra), orthonormalise $`Q = \operatorname{qr}(Y)`$.
3. Project: $`B = Q^\top A`$ (small), SVD $`B = \tilde U\Sigma V^\top`$.
4. $`U = Q\tilde U`$.

- Two passes over $`A`$ (plus $`2q`$); embarrassingly parallel matrix products; GPU friendly.

### PCA

- Centre the data $`X`$ (rows = samples), SVD $`X = U\Sigma V^\top`$. Principal directions = columns of $`V`$, scores = $`U\Sigma`$, explained variance $`\sigma_i^2/(n - 1)`$.
- *Standardise* variables with different units, otherwise the largest-unit variable dominates.
- *Kernel PCA* for nonlinear structure; *probabilistic PCA* as a latent-variable model with a likelihood.

### NMF

- $`A \approx WH`$ with $`W, H \ge 0`$. Algorithms: multiplicative updates (Lee–Seung, simple, slow), alternating non-negative least squares, HALS (fast).
- Non-convex: initialise with NNDSVD, run several times. Not unique in general.

### Tensor decompositions

- *CP:* $`\mathcal T \approx \sum_{r=1}^R a_r\otimes b_r\otimes c_r`$. Unique under mild conditions (Kruskal) — useful for interpretation — but the best rank-$`R`$ approximation may not exist (degenerate solutions).
- *Tucker:* $`\mathcal T \approx \mathcal G\times_1 U_1\times_2 U_2\times_3 U_3`$; HOSVD (SVD of each unfolding) is quasi-optimal, HOOI refines. Storage $`r^d`$ for the core.
- *Tensor train:* $`\mathcal T(i_1, \dots, i_d) = G_1(i_1)G_2(i_2)\cdots G_d(i_d)`$ with $`r\times r`$ matrices; storage $`O(dnr^2)`$. Built by sequential SVDs (TT-SVD) or from samples (TT-cross). Same as matrix product states in physics.

## Other lenses

- *Statistics.* PCA maximises retained variance; it is the MLE of probabilistic PCA. Factor analysis adds per-variable noise. Truncated SVD regularises least squares by discarding noisy directions (see [least squares](/ref/solve/least-squares)).
- *Dynamics.* POD = PCA of trajectories; DMD = low-rank linear dynamics (see [timescale reduction](/ref/dynamics/timescale-reduction)).
- *ML.* Low-rank structure is everywhere: embeddings, LoRA fine-tuning ($`W + BA`$ with small rank), attention approximations, compressed layers.
- *Kernel methods.* Nyström approximation and random features give low-rank kernel matrices; hierarchical matrices exploit low rank of blocks (see [hierarchical matrices](/ref/solve/hierarchical-matrices-fmm)).

## Pitfalls & diagnostics

- *Forgetting to centre* before PCA puts the mean into the first component.
- *Sign and rotation ambiguity:* singular vectors are defined up to sign (and rotation within repeated singular values); don't compare components across runs naively.
- *Randomised SVD accuracy* degrades for flat spectra; use power iterations and check $`\lVert A - QQ^\top A\rVert`$ by sampling.
- *Interpreting components* as physical factors is risky (PCA directions are orthogonal by construction, physics isn't).
- *CP rank selection* is hard; check core consistency and stability across random initialisations.

## See also

- [Least squares](/ref/solve/least-squares)
- [Eigenproblems](/ref/solve/eigenproblems)
- [High-dimensional numerics](/ref/approx/high-dimensional-numerics)
- [Unsupervised learning](/ref/learn/unsupervised)
- [Timescale separation & model reduction](/ref/dynamics/timescale-reduction)

*References:*

- Halko, Martinsson & Tropp, "Finding structure with randomness", SIAM Review (2011).
- Kolda & Bader, "Tensor decompositions and applications", SIAM Review (2009).
- Oseledets, "Tensor-train decomposition", SIAM J. Sci. Comput. (2011).
