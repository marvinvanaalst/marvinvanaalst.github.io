---
title: 'Solve Ax = b'
description: 'Dense and banded direct solvers: which factorisation for which matrix, what it costs, and when to switch to sparse or iterative methods.'
aliases:
  [
    LU,
    Cholesky,
    LDLT,
    QR,
    Householder,
    Thomas algorithm,
    tridiagonal,
    Gaussian elimination,
    iterative refinement,
    Woodbury
  ]
tags: [linear algebra, factorisation, direct solver]
order: 1
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

Factor once, then solve each right-hand side in $O(n^2)$. Never form $A^{-1}$.

- **SPD** → Cholesky.
- **General square** → LU with partial pivoting.
- **Ill-conditioned or rectangular** → QR.
- **Banded** → banded LU (tridiagonal: Thomas, $O(n)$).
- **Large and sparse** → sparse direct while memory allows (2D problems), Krylov + preconditioner beyond that (3D, matrix-free).

## Decision table

| Variant                        | Use when                                    | Cost (flops)                                       | Stability                                                | Pitfalls                                     |
| ------------------------------ | ------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------- |
| Cholesky $LL^\top$             | symmetric positive definite                 | $\tfrac13 n^3$                                     | stable, no pivoting needed                               | fails if not SPD (doubles as an SPD test)    |
| LU, partial pivoting $PA = LU$ | general square                              | $\tfrac23 n^3$                                     | backward stable in practice (growth factor rarely large) | singular or near-singular $A$                |
| $LDL^\top$ (Bunch–Kaufman)     | symmetric indefinite (KKT systems, saddles) | $\tfrac13 n^3$                                     | stable with symmetric pivoting                           | $D$ has 2×2 blocks                           |
| Householder QR                 | ill-conditioned square, least squares       | $\tfrac43 n^3$ (square), $2mn^2 - \tfrac23 n^3$    | backward stable, no $\kappa^2$ penalty                   | 2× LU cost for square systems                |
| Banded LU, bandwidth $p$       | banded (1D discretisations, splines)        | $O(np^2)$                                          | as LU; pivoting widens the upper band to $2p$            | fill stays inside the band                   |
| Thomas (tridiagonal)           | tridiagonal, diagonally dominant            | $\approx 8n$                                       | stable if diagonally dominant or SPD                     | no pivoting, can break otherwise             |
| Sparse direct (multifrontal)   | sparse, $n \lesssim 10^6$, many RHS         | depends on fill: 2D grid $O(n^{3/2})$, 3D $O(n^2)$ | as dense counterpart                                     | memory from fill-in; ordering matters        |
| Krylov + preconditioner        | huge sparse, matrix-free                    | $O(\text{nnz})$ per iteration                      | depends on preconditioner                                | no convergence without a good preconditioner |
| Explicit inverse               | almost never                                | $2n^3$                                             | less accurate than solving                               | wasteful even for many RHS                   |

Dense memory check: $n = 10^4$ in Float64 is 800 MB. Above that, dense is out unless you have a cluster.

## Variants

### LU with partial pivoting

Gaussian elimination written as a factorisation, $PA = LU$, with $L$ unit lower triangular and $U$ upper triangular.

- **Factorisation**, for each column $k$:
  1. Pivot: swap row $k$ with the row holding the largest $\lvert a_{ik}\rvert$, $i \ge k$. This keeps $\lvert l_{ij}\rvert \le 1$.
  2. Multipliers: $l_{ik} = a_{ik} / a_{kk}$ for $i > k$.
  3. Schur complement update of the trailing block: $a_{ij} \mathrel{-}= l_{ik}\,a_{kj}$ for $i, j > k$.
- **Solve:** forward substitution $Ly = Pb$, then back substitution $Ux = y$, $n^2$ flops each.
- **Speed:** real implementations are blocked (LAPACK `getrf`). The rank-1 update becomes a matrix-matrix product (BLAS-3): same flop count, 10–50× faster.

### Cholesky

For SPD $A$: $A = LL^\top$.

$$
l_{jj} = \sqrt{a_{jj} - \textstyle\sum_{k<j} l_{jk}^2}, \qquad
l_{ij} = \frac{a_{ij} - \sum_{k<j} l_{ik} l_{jk}}{l_{jj}} \quad (i > j)
$$

- Half the cost and half the storage of LU, and no pivoting needed.
- A non-positive radicand means $A$ is not SPD. This is the cheapest SPD test.
- **Free extras:**
  - $\log\det A = 2\sum_i \log l_{ii}$ (Gaussian likelihoods, GPs).
  - $x = \mu + Lz$ with $z \sim \mathcal N(0, I)$ samples $\mathcal N(\mu, A)$.

### Householder QR

$A = QR$ with orthogonal $Q$, built from reflectors $H = I - 2vv^\top / v^\top v$.

- **Solve:** $Rx = Q^\top b$. Apply the reflectors to $b$ instead of forming $Q$.
- **Why:** orthogonal transforms don't amplify errors, so QR is the safe choice when $\kappa(A)$ is large.
- The standard for least squares (see [least squares](/ref/solve/least-squares)).
- Column pivoting (`geqp3`) reveals the numerical rank.

### Banded and tridiagonal

- **Banded LU:** fill-in stays inside the band, so the cost is $O(np^2)$ instead of $O(n^3)$.
- **Thomas algorithm** (tridiagonal, e.g. 1D diffusion, cubic splines): LU without pivoting. With sub-, main- and super-diagonals $a_i, b_i, c_i$ and right-hand side $d_i$:
  1. Forward sweep: $m_i = b_i - a_i c'_{i-1}$, then $c'_i = c_i / m_i$ and $d'_i = (d_i - a_i d'_{i-1}) / m_i$.
  2. Back substitution: $x_n = d'_n$, then $x_i = d'_i - c'_i x_{i+1}$.
- **Periodic (cyclic) tridiagonal:** Sherman–Morrison on top of Thomas.

### Sparse direct

The same factorisations, but cost and memory are set by **fill-in**: zeros that turn non-zero during elimination.

1. **Ordering:** pick a fill-reducing permutation first (AMD, nested dissection; RCM if you only want a small bandwidth).
2. **Symbolic phase:** compute the sparsity pattern of the factors. Done once per pattern.
3. **Numeric phase:** compute the values.

Implicit ODE solvers and Newton loops refactor matrices with the same pattern, so reuse the symbolic phase. Details are on the [sparse matrices](/ref/solve/sparse-matrices) page.

### Iterative (pointer)

When factorising is too expensive in memory, use Krylov methods:

- CG for SPD.
- MINRES for symmetric indefinite.
- GMRES / BiCGSTAB for general matrices.

Each iteration costs one matrix-vector product. CG needs roughly $\sqrt{\kappa}\,\log(1/\varepsilon)$ iterations, so the preconditioner is the whole game. See [Krylov methods & preconditioning](/ref/solve/krylov-preconditioning).

### Useful tricks

- **Many right-hand sides:** factor once and solve in a batch ($AX = B$ is BLAS-3).
- **Low-rank update** $A + UCV^\top$ with $U, V \in \mathbb R^{n\times k}$: Woodbury reuses the factorisation of $A$ at $O(n^2 k)$ cost instead of refactoring: $\displaystyle (A + UCV^\top)^{-1} = A^{-1} - A^{-1}U\,(C^{-1} + V^\top A^{-1} U)^{-1}\,V^\top A^{-1}$.
- **Mixed-precision iterative refinement:** gives Float64 accuracy if $\kappa \ll 10^8$.
  1. Factor in Float32 (2× faster, half the memory).
  2. Compute the residual $r = b - Ax$ in Float64.
  3. Solve $A\,\delta = r$ with the cheap factors, update $x \mathrel{+}= \delta$, and repeat from step 2.
- **Equilibrate** (scale rows/columns) before factoring badly scaled matrices.

## Other lenses

- **Elimination = Schur complement.** Every elimination step replaces the trailing block with $A_{22} - A_{21}A_{11}^{-1}A_{12}$. Block elimination, static condensation and conditioning a Gaussian on observed variables all use the same formula.
- **SPD solve = optimisation.** $x$ minimises $\tfrac12 x^\top A x - b^\top x$. CG is an optimiser that uses conjugate directions on this quadratic.
- **Probability.**
  - One Cholesky of a covariance gives sampling ($Lz$), whitening ($L^{-1}x$) and the log-determinant.
  - Precision matrices of Markov random fields are sparse, so their Cholesky factor is the sparse-direct story again.
- **Graphs.**
  - Symmetric sparse elimination is vertex elimination on the adjacency graph: eliminating a vertex connects all its neighbours (fill).
  - The minimum fill is governed by treewidth, and nested dissection is graph partitioning.
- **QR = Gram–Schmidt**, done stably. $Q$ spans the column space and $R$ holds the projection coefficients.

## Pitfalls & diagnostics

- **Condition number:**
  - Expect to lose about $\log_{10}\kappa(A)$ digits. Forward error ≲ $\kappa$ × backward error.
  - A small residual does **not** mean an accurate $x$ when $\kappa$ is large.
  - Estimate $\kappa$ cheaply from the factorisation (LAPACK `gecon`, $O(n^2)$) rather than via the SVD.
- **Normal equations** $A^\top A x = A^\top b$ square the condition number. Use QR unless $A$ is well-conditioned and very tall.
- **Cholesky failing on a "covariance"** usually means numerical semi-definiteness. Add jitter $\epsilon I$ (GPs) or use pivoted Cholesky.
- **Determinants** over- or underflow: use $\log\lvert\det\rvert$ from the factor diagonal.
- **Reuse the factorisation.** Re-solving from scratch inside a loop is the most common silent $n^3$.

## See also

- [Least squares](/ref/solve/least-squares)
- [Sparse matrices](/ref/solve/sparse-matrices)
- [Krylov methods & preconditioning](/ref/solve/krylov-preconditioning)
- [Conditioning & stability](/ref/reliable/conditioning-stability)
- [Eigenproblems](/ref/solve/eigenproblems)

**References:**

- Golub & Van Loan, _Matrix Computations_ (4th ed.), ch. 3–4.
- Trefethen & Bau, _Numerical Linear Algebra_, lectures 20–23.
- Davis, _Direct Methods for Sparse Linear Systems_.
