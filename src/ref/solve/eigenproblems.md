---
title: 'Eigenproblems'
description: 'Dense QR algorithm vs iterative Lanczos/Arnoldi, power and inverse iteration, shift-invert, generalised problems and the SVD: which eigensolver for which question.'
aliases:
  [
    eigenvalues,
    eigenvectors,
    power iteration,
    inverse iteration,
    Rayleigh quotient iteration,
    QR algorithm,
    Hessenberg,
    Schur decomposition,
    Lanczos,
    Arnoldi,
    implicitly restarted Arnoldi,
    ARPACK,
    Krylov-Schur,
    LOBPCG,
    Jacobi-Davidson,
    shift-invert,
    generalized eigenproblem,
    SVD,
    pseudospectrum
  ]
tags: [linear algebra, eigenvalues, spectral]
order: 6
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *All eigenvalues of a dense matrix* ($`n \lesssim 10^4`$): the QR algorithm (LAPACK). Symmetric: $`\tfrac43 n^3`$ for values only, \~$`9n^3`$ with vectors; nonsymmetric: \~$`10n^3`$ (values) to $`25n^3`$ (with vectors).
- *A few eigenvalues of a large sparse matrix:* Lanczos (symmetric) or implicitly restarted Arnoldi / Krylov–Schur (general). Extreme eigenvalues converge first.
- *Eigenvalues near a target $`\sigma`$:* *shift-invert*, i.e. run Lanczos/Arnoldi on $`(A - \sigma I)^{-1}`$; needs a sparse factorisation.
- *One dominant eigenvector* (PageRank, Perron vector): power iteration. One eigenpair near a good guess: Rayleigh quotient iteration.
- Symmetric problems are perfectly conditioned; nonsymmetric eigenvalues can be extremely sensitive (non-normality).

## Decision table

| Method | Finds | Cost | Use when |
| --- | --- | --- | --- |
| QR algorithm (dense) | all eigenvalues (+ vectors) | $`O(n^3)`$ | dense, $`n \lesssim 10^4`$ |
| Divide & conquer / MRRR | all, symmetric tridiagonal | $`O(n^2)`$–$`O(n^3)`$, fast in practice | symmetric with vectors (default in LAPACK drivers) |
| Power iteration | dominant eigenpair | 1 matvec / iteration, rate $`\lvert\lambda_2/\lambda_1\rvert`$ | well-separated dominant eigenvalue |
| Inverse iteration | eigenpair nearest $`\sigma`$ | 1 solve / iteration | refine a known approximate eigenvalue |
| Rayleigh quotient iteration | nearest eigenpair | 1 solve with a new shift each step | cubic convergence (symmetric), good start |
| Lanczos (thick restart) | few extreme eigenpairs, symmetric | matvecs + reorthogonalisation | large sparse symmetric |
| Arnoldi / Krylov–Schur (ARPACK) | few eigenpairs, general | matvecs, $`O(nm)`$ memory | large sparse nonsymmetric |
| Shift-invert + Krylov | eigenvalues near $`\sigma`$, interior | one sparse factorisation + solves | interior eigenvalues, smallest magnitude |
| LOBPCG | few extreme eigenpairs, symmetric | block matvecs, preconditioned | good preconditioner, blocks, no factorisation |
| Jacobi–Davidson | interior eigenpairs | inexact inner solves | no factorisation possible, interior spectrum |
| FEAST / contour integral | all eigenvalues in a region | several shifted solves (parallel) | eigenvalues in an interval or disc |

## Variants

### The dense QR algorithm

1. *Reduce* to Hessenberg form (nonsymmetric) or tridiagonal form (symmetric) by Householder similarity transforms, $`O(n^3)`$, once.
2. *Iterate* shifted QR steps $`H - \mu I = QR`$, $`H \leftarrow RQ + \mu I`$ implicitly (bulge chasing, $`O(n^2)`$ per step), with Wilkinson or Francis double shifts.
3. *Deflate* as subdiagonal entries become negligible.

- Result: Schur form $`A = QTQ^*`$ ($`T`$ triangular), backward stable. Eigenvectors from $`T`$ by back substitution.
- Symmetric tridiagonal: QR, divide and conquer, or MRRR (Multiple Relatively Robust Representations, $`O(n^2)`$ for all vectors).

### Power, inverse and Rayleigh quotient iteration

- *Power:* $`x_{k+1} = Ax_k / \lVert Ax_k\rVert`$. Converges to the dominant eigenvector at rate $`\lvert\lambda_2/\lambda_1\rvert`$.
- *Inverse iteration:* power iteration with $`(A - \sigma I)^{-1}`$; converges to the eigenvalue nearest $`\sigma`$ at rate $`\lvert\lambda_{\text{near}} - \sigma\rvert / \lvert\lambda_{\text{next}} - \sigma\rvert`$.
- *Rayleigh quotient* $`\rho(x) = x^\top A x / x^\top x`$ is the best eigenvalue estimate for a given vector (error quadratic in the vector error, symmetric case). *RQI* uses $`\sigma_k = \rho(x_k)`$ as shift each step: cubic convergence for symmetric matrices.

### Krylov eigensolvers

- *Lanczos* (symmetric): three-term recurrence builds an orthonormal basis $`V_k`$ and tridiagonal $`T_k = V_k^\top A V_k`$. Ritz values (eigenvalues of $`T_k`$) approximate extreme eigenvalues first.
- *Arnoldi* (general): Hessenberg $`H_k`$ instead of $`T_k`$, full orthogonalisation.
- *Restarting:* keep the wanted Ritz vectors and throw away the rest (implicit restarts, Krylov–Schur, thick restart) so memory stays bounded.
- *Finite precision:* Lanczos loses orthogonality as Ritz values converge, producing spurious copies; selective or full reorthogonalisation fixes it.

### Spectral transformations

- *Shift-invert:* eigenvalues of $`(A - \sigma I)^{-1}`$ are $`1/(\lambda - \sigma)`$, so eigenvalues near $`\sigma`$ become extreme and converge fast.
- *Polynomial filtering* (Chebyshev): amplify an interval of the spectrum without factorisation.
- *Cayley transform* $`(A - \sigma I)^{-1}(A - \tau I)`$ for stability analysis (rightmost eigenvalues).

### Generalised and related problems

- *$`Ax = \lambda Bx`$:* symmetric-definite ($`B`$ SPD): reduce via Cholesky of $`B`$, or use $`B`$-inner products in Lanczos. General: QZ algorithm (generalised Schur).
- *Quadratic eigenproblems* $`(\lambda^2 M + \lambda C + K)x = 0`$: linearise to a $`2n`$ generalised problem.
- *SVD:* eigenvalues of $`A^\top A`$ without forming it. Dense: bidiagonalisation + implicit QR / divide and conquer. Large: Golub–Kahan–Lanczos or randomised SVD (see [low-rank](/ref/approx/low-rank)).

## Other lenses

- *Dynamics.* Eigenvalues of the Jacobian decide local stability ($`\operatorname{Re}\lambda \lt 0`$) and timescales ($`1/\lvert\lambda\rvert`$). The rightmost eigenvalue is the one that matters, which is an interior problem for spectra with large negative values. See [steady states & stability](/ref/dynamics/steady-states-stability).
- *Optimisation.* Eigenvalues of a symmetric matrix are stationary values of the Rayleigh quotient: $`\lambda_{\min} = \min_x \rho(x)`$. LOBPCG is a preconditioned optimiser for it.
- *Probability.* The stationary distribution of a Markov chain is the left Perron eigenvector; the spectral gap sets the mixing time (see [stochastic processes](/ref/dynamics/stochastic-processes)). PCA = eigenvectors of the covariance.
- *Graphs.* The Laplacian's second eigenvector (Fiedler vector) gives spectral partitioning and clustering.

## Pitfalls & diagnostics

- *Non-normal matrices:* eigenvalues can move by $`\gg\lVert\Delta A\rVert`$ under perturbations, and transient growth can happen even when all eigenvalues are stable. Inspect the pseudospectrum or the condition numbers of eigenvalues ($`1/\lvert y^* x\rvert`$ for unit left/right eigenvectors).
- *Clustered eigenvalues* make individual eigenvectors ill-conditioned; the invariant subspace is still well defined. Ask for the subspace.
- *Smallest eigenvalues* by plain Lanczos/Arnoldi converge slowly; use shift-invert at 0 or a preconditioned method.
- *Don't compute eigenvalues to solve linear systems or compute matrix functions* when a factorisation or Krylov method is cheaper.
- *Check residuals* $`\lVert Ax - \lambda x\rVert / (\lVert A\rVert\lVert x\rVert)`$ for every returned pair.

## See also

- [Krylov methods & preconditioning](/ref/solve/krylov-preconditioning)
- [Matrix functions](/ref/solve/matrix-functions)
- [Low-rank approximation](/ref/approx/low-rank)
- [Steady states & stability](/ref/dynamics/steady-states-stability)

*References:*

- Golub & Van Loan, _Matrix Computations_, ch. 7–8, 10.
- Saad, _Numerical Methods for Large Eigenvalue Problems_ (2nd ed.).
- Trefethen & Embree, _Spectra and Pseudospectra_.
