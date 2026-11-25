---
title: 'Multigrid & domain decomposition'
description: 'Geometric and algebraic multigrid, smoothers and cycles, multigrid as solver vs preconditioner, and Schwarz and Schur-complement domain decomposition.'
aliases:
  [
    multigrid,
    geometric multigrid,
    algebraic multigrid,
    AMG,
    V-cycle,
    W-cycle,
    F-cycle,
    full multigrid,
    smoother,
    Gauss-Seidel,
    prolongation,
    restriction,
    coarse grid correction,
    smoothed aggregation,
    domain decomposition,
    additive Schwarz,
    restricted additive Schwarz,
    FETI,
    BDDC,
    Schur complement method
  ]
tags: [linear algebra, PDE, iterative solver, parallel]
order: 8
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Multigrid* solves elliptic problems (diffusion, Poisson, elasticity) in $`O(n)`$ work: smoothing kills high-frequency error, a coarse grid kills the low-frequency error.
- *Geometric MG* (GMG) when you control a hierarchy of meshes; *algebraic MG* (AMG) builds the hierarchy from the matrix alone. AMG as a preconditioner for CG/GMRES is the robust default.
- *Domain decomposition* (DD) splits the domain into subdomains solved independently (parallelism); a *coarse space* is needed for scalability, which is where DD and multigrid meet.
- Neither works out of the box for strongly convective, indefinite (Helmholtz) or saddle-point problems; those need specialised smoothers, coarse spaces or block preconditioners.

## Decision table

| Method | Setup | Iterations vs $`n`$ | Parallel | Use when |
| --- | --- | --- | --- | --- |
| Geometric MG (V-cycle) | mesh hierarchy, cheap | $`O(1)`$ | good | structured grids, matrix-free high-order methods |
| Full multigrid (FMG) | as GMG | one pass to discretisation error | good | best-possible $`O(n)`$ solve on nested grids |
| Classical AMG (Ruge–Stüben) | moderate–high | $`O(1)`$ for M-matrices | moderate | scalar elliptic, unstructured meshes |
| Smoothed aggregation AMG | moderate | $`O(1)`$ | good | elasticity (with near-nullspace = rigid modes) |
| Additive Schwarz (one level) | subdomain factorisations | grows with #subdomains | perfect | small numbers of subdomains |
| Two-level Schwarz | + coarse problem | \~independent | good | scalable DD with overlap |
| FETI-DP / BDDC | subdomain + coarse primal dofs | $`O((1 + \log(H/h))^2)`$ | excellent | very large elliptic problems, non-overlapping |
| Sparse direct | factorisation | 1 | limited | 2D, moderate size (see [sparse matrices](/ref/solve/sparse-matrices)) |

## Variants

### The two-grid idea

1. *Pre-smooth:* a few sweeps of Gauss–Seidel or damped Jacobi ($`\omega \approx 2/3`$) on $`Ax = b`$. Removes oscillatory error components quickly; smooth error barely changes.
2. *Restrict* the residual $`r = b - Ax`$ to the coarse grid: $`r_c = R r`$.
3. *Coarse solve* $`A_c e_c = r_c`$ with the Galerkin operator $`A_c = R A P`$ (often $`R = P^\top`$).
4. *Prolongate and correct:* $`x \leftarrow x + P e_c`$.
5. *Post-smooth.*

Recursing on step 3 gives a V-cycle (one recursive call), W-cycle (two) or F-cycle (in between). Work per V-cycle is \~$`\tfrac43\times`$ (2D) the finest-level work.

### Smoothers

- Damped Jacobi and Chebyshev: perfectly parallel, standard on GPUs.
- Gauss–Seidel (lexicographic, red-black, symmetric): stronger, less parallel.
- ILU or block smoothers for anisotropic problems; line/plane relaxation for strong anisotropy on structured grids.
- Vanka / Braess–Sarazin smoothers for saddle points (Stokes).

### Algebraic multigrid

- *Strength of connection:* $`j`$ strongly influences $`i`$ if $`-a_{ij} \ge \theta\max_k(-a_{ik})`$, typically $`\theta \approx 0.25`$.
- *Coarsening:* classical (C/F splitting) or aggregation (group strongly connected nodes).
- *Interpolation:* built so that "algebraically smooth" error (small $`Ae`$) is well represented; smoothed aggregation smooths a tentative piecewise-constant prolongator.
- *Near-nullspace:* for systems, supply the modes with $`Ae \approx 0`$ (rigid body modes for elasticity), otherwise convergence degrades badly.

### Domain decomposition

- *Overlapping Schwarz:* solve on overlapping subdomains, add (additive) or apply sequentially (multiplicative). Restricted additive Schwarz (RAS) is the common preconditioner.
- *Non-overlapping / substructuring:* eliminate subdomain interiors, leaving the interface Schur complement $`S = A_{\Gamma\Gamma} - A_{\Gamma I}A_{II}^{-1}A_{I\Gamma}`$; solve it iteratively with Neumann–Neumann, FETI or BDDC preconditioners.
- *Coarse space:* one-level methods need more iterations as the number of subdomains grows (information travels one subdomain per iteration). A coarse problem (one dof per subdomain, or spectral coarse spaces like GenEO for heterogeneous coefficients) restores scalability.

## Other lenses

- *Fourier analysis.* Local Fourier analysis predicts smoothing factors and two-grid convergence rates exactly for constant-coefficient problems; it explains why Jacobi needs damping.
- *Subspace correction.* Multigrid and DD are both "subspace correction" methods: decompose the space, solve projected problems, combine. Additive = parallel (Jacobi-like), multiplicative = sequential (Gauss–Seidel-like).
- *Hierarchical models.* Coarse levels resemble multiresolution analysis (wavelets) and hierarchical bases; in ML, U-Nets share the down/up-sampling structure.

## Pitfalls & diagnostics

- *Measure the convergence factor* (residual reduction per cycle). Good MG gives 0.05–0.2 per V-cycle, independent of the mesh size. If it degrades with refinement, the coarse space or smoother is wrong.
- *Anisotropy and jumping coefficients* break point smoothers and naive coarsening; use semi-coarsening, line smoothers or operator-dependent interpolation (AMG handles coefficient jumps well).
- *Indefinite problems* (Helmholtz at high frequency): standard MG diverges; use shifted-Laplacian preconditioners or specialised methods.
- *AMG setup cost* can dominate for problems that are solved only once with few iterations; reuse the setup across Newton steps or time steps.
- *Use as a preconditioner* for CG/GMRES rather than a standalone solver; it is more robust for small amounts of extra work.

## See also

- [Krylov methods & preconditioning](/ref/solve/krylov-preconditioning)
- [Sparse matrices](/ref/solve/sparse-matrices)
- [Method of lines](/ref/integrate/pdes-method-of-lines)
- [Hierarchical matrices & FMM](/ref/solve/hierarchical-matrices-fmm)

*References:*

- Briggs, Henson & McCormick, _A Multigrid Tutorial_ (2nd ed.).
- Trottenberg, Oosterlee & Schüller, _Multigrid_.
- Toselli & Widlund, _Domain Decomposition Methods: Algorithms and Theory_.
