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
updated: '2026-11-15'
published: true
layout: ref
---

## TL;DR

Factor once, then solve each right-hand side in $O(n^2)$. **SPD → Cholesky. General → LU with partial pivoting. Ill-conditioned, or rectangular → QR.** Banded → banded LU (tridiagonal: Thomas, $O(n)$). Large and sparse → sparse direct while memory allows (2D problems), Krylov + preconditioner beyond that (3D, matrix-free). Never form $A^{-1}$.

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

**Dense memory check:** $n = 10^4$ in Float64 is 800 MB. Above that, dense is out unless you have a cluster.

## Variants

### LU with partial pivoting

Gaussian elimination written as a factorisation: $PA = LU$ with $L$ unit lower triangular and $U$ upper triangular. To solve, run $Ly = Pb$ (forward substitution) and then $Ux = y$ (back substitution), $n^2$ flops each. Partial pivoting (taking the largest entry in the column as pivot) keeps $|l_{ij}| \le 1$.

```julia
function lu_pp!(A)
    n = size(A, 1); p = collect(1:n)
    for k in 1:n-1
        i = k - 1 + argmax(abs.(A[k:n, k]))              # pivot row
        A[[k, i], :] = A[[i, k], :]; p[[k, i]] = p[[i, k]]
        A[k+1:n, k] ./= A[k, k]                          # multipliers → L
        A[k+1:n, k+1:n] .-= A[k+1:n, k] * A[k, k+1:n]'   # Schur complement update
    end
    return A, p        # unit-lower L and U packed into A
end
```

Real implementations are **blocked** (LAPACK `getrf`): the rank-1 update becomes a matrix-matrix product (BLAS-3), which is where the speed comes from. The flop count is the same, but it runs 10–50× faster.

### Cholesky

For SPD $A$, $A = LL^\top$ and no pivoting is needed. It costs half of LU and stores only half of the matrix.

$$
l_{jj} = \sqrt{a_{jj} - \textstyle\sum_{k<j} l_{jk}^2}, \qquad
l_{ij} = \frac{a_{ij} - \sum_{k<j} l_{ik} l_{jk}}{l_{jj}} \quad (i > j)
$$

```julia
function cholesky_lower(A)
    n = size(A, 1); L = zeros(n, n)
    for j in 1:n
        s = A[j, j] - sum(abs2, L[j, 1:j-1]; init=0.0)
        s > 0 || error("not SPD")
        L[j, j] = sqrt(s)
        for i in j+1:n
            L[i, j] = (A[i, j] - L[i, 1:j-1]' * L[j, 1:j-1]) / L[j, j]
        end
    end
    return L
end
```

Free extras: $\log\det A = 2\sum_i \log l_{ii}$ (Gaussian likelihoods, GPs), and $x = \mu + Lz$ with $z \sim \mathcal N(0, I)$ samples $\mathcal N(\mu, A)$.

### Householder QR

$A = QR$ with orthogonal $Q$, built from reflectors $H = I - 2vv^\top / v^\top v$. To solve $Ax = b$, compute $Rx = Q^\top b$. Apply the reflectors to $b$ instead of forming $Q$. Orthogonal transforms don't amplify errors, so QR is the safe choice when $\kappa(A)$ is large, and it is the standard for least squares (see the least-squares page). Column pivoting (`geqp3`) reveals the numerical rank.

### Banded and tridiagonal

Fill-in from LU stays inside the band, so the cost is $O(np^2)$ instead of $O(n^3)$. For tridiagonal systems (1D diffusion, cubic splines) the Thomas algorithm is LU without pivoting:

```julia
function thomas(a, b, c, d)    # sub-, main-, super-diagonal, rhs
    n = length(b); c′ = similar(c); d′ = similar(d)
    c′[1] = c[1] / b[1]; d′[1] = d[1] / b[1]
    for i in 2:n
        m = b[i] - a[i-1] * c′[i-1]
        i < n && (c′[i] = c[i] / m)
        d′[i] = (d[i] - a[i-1] * d′[i-1]) / m
    end
    x = similar(d); x[n] = d′[n]
    for i in n-1:-1:1
        x[i] = d′[i] - c′[i] * x[i+1]
    end
    return x
end
```

Periodic tridiagonal (cyclic): Sherman–Morrison on top of Thomas.

### Sparse direct

The same factorisations, but cost and memory are set by **fill-in**: zeros that turn non-zero during elimination. A fill-reducing ordering (AMD, nested dissection; RCM if you only want a small bandwidth) is chosen first. The work splits into a **symbolic** phase (sparsity pattern, done once) and a **numeric** phase (values). Implicit ODE solvers and Newton loops refactor matrices with the same pattern, so reuse the symbolic phase. Details are on the sparse-matrices page.

### Iterative (pointer)

When factorising is too expensive in memory, use Krylov methods: CG for SPD, MINRES for symmetric indefinite, GMRES/BiCGSTAB for general. Each iteration costs one matrix-vector product. CG needs roughly $\sqrt{\kappa}\,\log(1/\varepsilon)$ iterations, so the preconditioner is the whole game. See the Krylov-and-preconditioning page.

### Useful tricks

- **Many right-hand sides:** factor once and solve in a batch ($AX = B$ is BLAS-3).
- **Low-rank update** $A + UCV^\top$ with $U, V \in \mathbb R^{n\times k}$: Woodbury reuses the factorisation of $A$ at $O(n^2 k)$ cost instead of refactoring: $\displaystyle (A + UCV^\top)^{-1} = A^{-1} - A^{-1}U\,(C^{-1} + V^\top A^{-1} U)^{-1}\,V^\top A^{-1}$.
- **Mixed-precision iterative refinement:** factor in Float32 (2× faster, half the memory), then repeat $r = b - Ax$ in Float64, solve $A\,\delta = r$ with the cheap factors, $x \mathrel{+}= \delta$. You get Float64 accuracy if $\kappa \ll 10^8$.
- **Equilibrate** (scale rows/columns) before factoring badly scaled matrices.

## Other lenses

- **Elimination = Schur complement.** Every elimination step replaces the trailing block with $A_{22} - A_{21}A_{11}^{-1}A_{12}$. Block elimination, static condensation and the conditioning of Gaussians on observed variables are the same formula.
- **SPD solve = optimisation.** $x$ minimises $\tfrac12 x^\top A x - b^\top x$. CG is an optimiser that uses conjugate directions on this quadratic.
- **Probability.** Cholesky of a covariance gives sampling ($Lz$), whitening ($L^{-1}x$) and the log-determinant in one factorisation. Precision matrices of Markov random fields are sparse, and their Cholesky factor is the sparse-direct story again.
- **Graphs.** Symmetric sparse elimination is vertex elimination on the adjacency graph: eliminating a vertex connects all its neighbours (fill). The minimum fill is governed by treewidth, and nested dissection is graph partitioning.
- **QR = Gram–Schmidt**, done stably. $Q$ spans the column space and $R$ holds the projection coefficients.

## Pitfalls & diagnostics

- **Condition number:** expect to lose about $\log_{10}\kappa(A)$ digits. Forward error ≲ $\kappa$ × backward error. A small residual does **not** mean an accurate $x$ when $\kappa$ is large.
- Estimate $\kappa$ cheaply from the factorisation (LAPACK `gecon`, $O(n^2)$) rather than via the SVD.
- **Normal equations** $A^\top A x = A^\top b$ square the condition number. Use QR unless $A$ is well-conditioned and very tall.
- Cholesky failing on a "covariance" usually means numerical semi-definiteness. Add jitter $\epsilon I$ (GPs) or use pivoted Cholesky.
- Determinants over- or underflow: use $\log\lvert\det\rvert$ from the factor diagonal.
- Keep the factorisation object and reuse it. Re-solving from scratch inside a loop is the most common silent $n^3$.

## See also

Least squares · Sparse matrices · Krylov methods & preconditioning · Conditioning & stability · Eigenproblems

**References:** Golub & Van Loan, _Matrix Computations_ (4th ed.), ch. 3–4. Trefethen & Bau, _Numerical Linear Algebra_, lectures 20–23. Davis, _Direct Methods for Sparse Linear Systems_.
