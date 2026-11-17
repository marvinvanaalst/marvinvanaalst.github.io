---
title: 'Krylov methods & preconditioning'
description: 'CG, MINRES, GMRES, BiCGSTAB and friends, convergence in terms of spectra, and the preconditioners that actually make them work.'
aliases:
  [
    CG,
    conjugate gradient,
    MINRES,
    GMRES,
    BiCGSTAB,
    QMR,
    IDR,
    LSQR,
    LSMR,
    Arnoldi,
    Lanczos,
    preconditioner,
    Jacobi,
    ILU,
    incomplete Cholesky,
    AMG,
    block preconditioner,
    restart
  ]
tags: [linear algebra, iterative solver, sparse]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Krylov methods find the best approximation to $x$ in $\mathcal K_k = \operatorname{span}\{b, Ab, \dots, A^{k-1}b\}$. Each iteration costs one matrix-vector product plus a few vector operations.
- **SPD → CG. Symmetric indefinite → MINRES. Nonsymmetric → GMRES** (restarted) or **BiCGSTAB** (short recurrences, less robust). Least squares → LSQR/LSMR.
- Convergence depends on the spectrum. Without a **preconditioner** the iteration count is usually unacceptable; the preconditioner is the real design decision.
- Preconditioner ladder: Jacobi → (block) Gauss–Seidel/SSOR → ILU / incomplete Cholesky → **algebraic multigrid** (elliptic PDEs) → physics-based block preconditioners (saddle points).

## Decision table

| Method              | Matrix                       | Memory         | Work / iteration    | Optimality                             | Pitfalls                                       |
| ------------------- | ---------------------------- | -------------- | ------------------- | -------------------------------------- | ---------------------------------------------- |
| CG                  | SPD                          | 4 vectors      | 1 matvec, 2 dots    | minimises $A$-norm error               | breaks on indefinite; needs SPD preconditioner |
| MINRES              | symmetric (indefinite)       | ~7 vectors     | 1 matvec            | minimises residual                     | needs SPD preconditioner                       |
| GMRES($m$)          | general                      | $m$ vectors    | 1 matvec, $k$ dots  | minimises residual over $\mathcal K_k$ | restarts can stagnate; cost grows with $k$     |
| BiCGSTAB            | general                      | ~8 vectors     | 2 matvecs           | none (short recurrence)                | erratic convergence, breakdowns                |
| IDR($s$)            | general                      | $O(s)$ vectors | $s+1$ matvecs       | none                                   | tune $s$                                       |
| CGLS / LSQR / LSMR  | rectangular (least squares)  | few vectors    | $Ax$ and $A^\top x$ | minimises $\lVert Ax - b\rVert$        | needs $A^\top$; LSMR has smoother residuals    |
| Chebyshev iteration | SPD, known eigenvalue bounds | few vectors    | 1 matvec, no dots   | minimax polynomial                     | needs spectral bounds; good as smoother        |

| Preconditioner           | Setup cost                   | Effectiveness                       | Parallel              | Use when                                     |
| ------------------------ | ---------------------------- | ----------------------------------- | --------------------- | -------------------------------------------- |
| Jacobi (diagonal)        | trivial                      | weak; fixes bad scaling             | perfect               | baseline, strongly diagonal problems         |
| Block Jacobi             | small dense LUs              | moderate                            | perfect               | multi-component nodes, subdomains            |
| SSOR / Gauss–Seidel      | none                         | moderate                            | poor (coloring helps) | smoothers in multigrid                       |
| ILU(0) / ILU(k) / ILUT   | like a partial factorisation | good for many nonsymmetric problems | poor                  | general sparse, moderate size                |
| Incomplete Cholesky      | as ILU                       | good for SPD                        | poor                  | SPD, not elliptic enough for AMG             |
| Algebraic multigrid      | moderate                     | iteration count ~independent of $n$ | good                  | elliptic PDEs (diffusion, pressure)          |
| Block / Schur complement | problem-specific             | can be mesh-independent             | good                  | saddle points (Stokes, KKT), coupled physics |

## Variants

### Conjugate gradients

1. $r_0 = b - Ax_0$, $p_0 = r_0$.
2. $\alpha_k = r_k^\top r_k / p_k^\top A p_k$, then $x_{k+1} = x_k + \alpha_k p_k$, $r_{k+1} = r_k - \alpha_k A p_k$.
3. $\beta_k = r_{k+1}^\top r_{k+1} / r_k^\top r_k$, $p_{k+1} = r_{k+1} + \beta_k p_k$.
4. Stop when $\lVert r_k\rVert \le \mathrm{tol}\cdot\lVert b\rVert$.

- **Convergence:** $\lVert e_k\rVert_A \le 2\left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^k \lVert e_0\rVert_A$, i.e. about $\tfrac12\sqrt\kappa\,\ln(2/\varepsilon)$ iterations. Clustered eigenvalues converge much faster than this bound.
- **Preconditioned CG** with SPD $M \approx A$: replace $r$ by $z = M^{-1}r$ in the inner products. Converges like $\kappa(M^{-1}A)$.

### GMRES

- Arnoldi builds an orthonormal basis $V_k$ of $\mathcal K_k$ with $AV_k = V_{k+1}\bar H_k$ (Hessenberg).
- Minimise $\lVert\beta e_1 - \bar H_k y\rVert$ (small least-squares problem via Givens rotations), set $x_k = x_0 + V_k y$.
- Work and memory grow linearly with $k$, so restart after $m$ steps (GMRES($m$), $m$ ~ 20–100). Restarting loses optimality and can stagnate.
- **Left vs right preconditioning:** right preconditioning ($AM^{-1}y = b$) minimises the true residual; flexible GMRES allows a changing preconditioner (e.g. an inner iterative solve).

### Short-recurrence nonsymmetric methods

- BiCG, CGS, BiCGSTAB, QMR, IDR($s$): constant memory and work per iteration, but no minimisation property and possible breakdown.
- Usually tried after GMRES($m$) runs out of memory.

### Building preconditioners

- **ILU(0):** Gaussian elimination keeping only entries in the pattern of $A$. ILU($k$) allows fill up to level $k$; ILUT drops by magnitude threshold.
- **AMG:** builds coarse levels automatically from the matrix (strength of connection, aggregation or classical coarsening), then applies multigrid cycles. See [multigrid](/ref/solve/multigrid-domain-decomposition).
- **Block preconditioners** for $\begin{pmatrix} A & B^\top \\ B & 0\end{pmatrix}$: use $\operatorname{diag}(\hat A, \hat S)$ with $\hat S \approx BA^{-1}B^\top$ (Schur complement approximation). With good approximations MINRES/GMRES converge in a handful of iterations.

## Other lenses

- **Polynomial approximation.** After $k$ steps, $r_k = p_k(A)r_0$ with $p_k(0) = 1$, and the method picks the best such polynomial. Convergence = how small a degree-$k$ polynomial can be on the spectrum. Clusters and small spectral intervals are good; eigenvalues near 0 are bad.
- **Optimisation.** CG is the optimal first-order method for minimising $\tfrac12 x^\top A x - b^\top x$, and nonlinear CG generalises it. Lanczos is CG's bookkeeping view.
- **Model reduction.** Arnoldi/Lanczos project $A$ onto a small subspace; the same machinery gives eigenvalues ([eigenproblems](/ref/solve/eigenproblems)), matrix functions ([matrix functions](/ref/solve/matrix-functions)) and moment matching.

## Pitfalls & diagnostics

- **Residual vs error:** the residual can be small while the error is large if $\kappa$ is large. Monitor both if you can (e.g. against a reference solution on a small case).
- **Loss of orthogonality** in Lanczos/CG with finite precision delays convergence; it rarely prevents it. GMRES uses modified Gram–Schmidt or Householder Arnoldi.
- **Stagnation of restarted GMRES:** increase $m$, add deflation (GCRO-DR), or improve the preconditioner.
- **Preconditioner must match the method:** CG and MINRES need SPD preconditioners.
- **Nullspaces:** singular systems (pure Neumann, pressure) need the nullspace projected out or the right-hand side made consistent.
- **Tolerance:** for Newton–Krylov use inexact (Eisenstat–Walker) forcing terms; solving each linear system to $10^{-12}$ wastes work.

## See also

- [Solve Ax = b](/ref/solve/linear-systems)
- [Sparse matrices](/ref/solve/sparse-matrices)
- [Multigrid & domain decomposition](/ref/solve/multigrid-domain-decomposition)
- [Nonlinear roots](/ref/solve/nonlinear-roots) (Newton–Krylov)
- [Eigenproblems](/ref/solve/eigenproblems)

**References:**

- Saad, _Iterative Methods for Sparse Linear Systems_ (2nd ed.).
- Shewchuk, "An introduction to the conjugate gradient method without the agonizing pain" (1994).
- Benzi, "Preconditioning techniques for large linear systems: a survey", J. Comput. Phys. (2002).
