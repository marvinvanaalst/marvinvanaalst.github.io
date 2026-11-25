---
title: 'Sparse recovery & compressed sensing'
description: 'L1 regularisation (LASSO, basis pursuit), greedy methods (OMP), recovery guarantees (RIP, coherence), proximal solvers (ISTA/FISTA) and the link to SINDy.'
aliases:
  [
    compressed sensing,
    sparse recovery,
    LASSO,
    basis pursuit,
    BPDN,
    L1 regularization,
    OMP,
    orthogonal matching pursuit,
    CoSaMP,
    iterative hard thresholding,
    RIP,
    restricted isometry property,
    mutual coherence,
    ISTA,
    FISTA,
    soft thresholding,
    elastic net,
    SINDy,
    sparse regression,
    STLSQ
  ]
tags: [sparsity, regularisation, optimisation, signal processing]
order: 8
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- Recover a sparse $`x`$ (few non-zeros, or sparse in some basis) from $`y = Ax + e`$ with fewer measurements than unknowns, by preferring sparsity.
- *Convex route:* LASSO $`\min\tfrac12\lVert Ax - y\rVert^2 + \lambda\lVert x\rVert_1`$ (or basis pursuit). Solve with coordinate descent (small/medium), FISTA or ADMM (large, matrix-free).
- *Greedy route:* OMP / CoSaMP / iterative hard thresholding. Faster, simpler, comparable guarantees.
- *Guarantees:* with random (Gaussian, partial Fourier) $`A`$, $`m \gtrsim C\,k\log(n/k)`$ measurements recover any $`k`$-sparse $`x`$ stably (RIP).
- *SINDy* discovers ODE right-hand sides as sparse combinations of library functions; it is sparse regression with sequential thresholded least squares.

## Decision table

| Method | Problem solved | Cost per iteration | Use when |
| --- | --- | --- | --- |
| LASSO, coordinate descent | $`\tfrac12\lVert Ax - y\rVert^2 + \lambda\lVert x\rVert_1`$ | $`O(\text{nnz}(A))`$ per sweep | regression, regularisation paths (glmnet) |
| LARS / homotopy | full LASSO path in $`\lambda`$ | $`O(mn\,k)`$ | small problems, whole path |
| ISTA / FISTA | LASSO and other composite problems | 1 $`A`$, 1 $`A^\top`$ | large, matrix-free (imaging, Fourier sensing) |
| ADMM | LASSO, total variation, constrained variants | linear solve | structured problems, distributed |
| Basis pursuit (LP) | $`\min\lVert x\rVert_1`$ s.t. $`Ax = y`$ | LP solver | noiseless recovery, small |
| OMP | greedy $`k`$-sparse fit | least squares per step | small $`k`$, fast and simple |
| CoSaMP / subspace pursuit | greedy with pruning | $`O(mn)`$ | guarantees like L1, faster |
| Iterative hard thresholding | $`\min\lVert Ax - y\rVert^2`$ s.t. $`\lVert x\rVert_0 \le k`$ | 1 $`A`$, 1 $`A^\top`$ | known sparsity, large scale |
| Elastic net | L1 + L2 | as LASSO | correlated features (LASSO picks one arbitrarily) |
| STLSQ (SINDy) | thresholded least squares, repeated | least squares | system identification from derivatives |

## Variants

### Why L1

- $`\lVert x\rVert_0`$ (count of non-zeros) is the real objective but NP-hard.
- $`\lVert x\rVert_1`$ is its convex envelope on the unit box; its level sets have corners on the axes, so solutions tend to be sparse.
- Bayesian view: LASSO is the MAP estimate under a Laplace prior (but the posterior mean is not sparse; horseshoe priors give better uncertainty).

### Proximal gradient: ISTA and FISTA

For $`\min_x g(x) + \lambda\lVert x\rVert_1`$ with smooth $`g`$ and Lipschitz constant $`L`$ of $`\nabla g`$:

1. Gradient step: $`z = x_k - \tfrac1L\nabla g(x_k)`$.
2. *Soft thresholding* (prox of L1): $`x_{k+1,i} = \operatorname{sign}(z_i)\max(\lvert z_i\rvert - \lambda/L, 0)`$.
3. *FISTA:* add Nesterov momentum, $`y_{k+1} = x_{k+1} + \tfrac{t_k - 1}{t_{k+1}}(x_{k+1} - x_k)`$, improving the rate from $`O(1/k)`$ to $`O(1/k^2)`$.

### Greedy: OMP

1. Residual $`r = y`$, support $`S = \emptyset`$.
2. Add the column most correlated with $`r`$: $`j = \arg\max_i\lvert a_i^\top r\rvert`$.
3. Least squares on the support: $`x_S = \arg\min\lVert A_Sx_S - y\rVert`$, update $`r = y - A_Sx_S`$.
4. Repeat until $`\lVert r\rVert`$ is small or $`\lvert S\rvert = k`$.

### Recovery guarantees

- *RIP of order $`2k`$* with constant $`\delta_{2k} \lt \sqrt2 - 1`$: L1 recovers every $`k`$-sparse vector exactly (noiseless) and stably (noisy, compressible signals).
- *Random matrices* satisfy RIP with $`m = O(k\log(n/k))`$; deterministic matrices rarely provably do.
- *Mutual coherence* $`\mu = \max_{i \ne j}\lvert a_i^\top a_j\rvert`$ (normalised columns): recovery if $`k \lt \tfrac12(1 + 1/\mu)`$; easy to compute but pessimistic.
- Sparsity can be in a transform domain: wavelets for images, total variation for piecewise-constant signals.

### SINDy

1. Measure states $`X`$ and estimate derivatives $`\dot X`$ (smoothed finite differences or total-variation regularised differentiation).
2. Build a library $`\Theta(X)`$ of candidate terms (polynomials, trig functions, rational terms).
3. Solve $`\dot X \approx \Theta(X)\Xi`$ with sequential thresholded least squares: least squares, zero coefficients below $`\lambda`$, refit on the remaining terms, repeat.
4. Validate the identified model by simulation on held-out data.

- Weak/integral formulations avoid noisy derivative estimates; implicit-SINDy handles rational kinetics.

## Other lenses

- *Optimisation.* LASSO is a convex composite problem; proximal operators, duality and ADMM are the general toolkit (see [convex duality & proximal methods](/ref/optimise/convex-duality)).
- *Statistics.* LASSO does variable selection and shrinkage; selection consistency needs irrepresentability conditions, and post-selection inference needs care (see [regression & GLMs](/ref/freq/regression-glms)).
- *Signal processing.* Compressed sensing MRI, single-pixel cameras and radar exploit sparsity to reduce measurements; the same mathematics as sparse regression.
- *Scientific ML.* SINDy and sparse symbolic regression produce interpretable models; UDEs + SINDy recover missing terms from a trained neural network (see [neural ODEs & UDEs](/ref/optimise/neural-odes-udes)).

## Pitfalls & diagnostics

- *Choosing $`\lambda`$:* cross-validation for prediction, but it over-selects; use stability selection or information criteria for model identification.
- *Correlated columns* (collinear library terms in SINDy, correlated features): L1 picks one arbitrarily; use elastic net, group LASSO or orthogonalise.
- *Scaling:* standardise columns, otherwise the penalty treats variables unequally.
- *Shrinkage bias:* LASSO shrinks retained coefficients; refit by least squares on the selected support ("relaxed LASSO").
- *Noisy derivatives* ruin SINDy; smooth carefully or use weak formulations.

## See also

- [Least squares](/ref/solve/least-squares)
- [Convex duality & proximal methods](/ref/optimise/convex-duality)
- [Neural ODEs & UDEs](/ref/optimise/neural-odes-udes)
- [Regularisation & generalisation](/ref/learn/regularisation-generalisation)

*References:*

- Foucart & Rauhut, _A Mathematical Introduction to Compressive Sensing_.
- Hastie, Tibshirani & Wainwright, _Statistical Learning with Sparsity_.
- Brunton, Proctor & Kutz, "Discovering governing equations from data by sparse identification of nonlinear dynamical systems", PNAS (2016).
