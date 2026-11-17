---
title: 'Sampling from distributions'
description: 'Inverse CDF, Box–Muller and Ziggurat for normals, rejection sampling, the alias method, Marsaglia–Tsang for gamma, multivariate normals via Cholesky, and the Gumbel-max trick.'
aliases:
  [
    random variate generation,
    inverse transform sampling,
    inverse CDF,
    Box-Muller,
    polar method,
    Ziggurat,
    rejection sampling,
    acceptance-rejection,
    alias method,
    Walker,
    Vose,
    Marsaglia-Tsang,
    gamma sampling,
    beta sampling,
    Dirichlet,
    multivariate normal,
    Cholesky sampling,
    Gumbel-max trick,
    Gumbel-softmax,
    reservoir sampling,
    Poisson sampling,
    binomial sampling
  ]
tags: [random numbers, sampling, simulation]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Inverse CDF:** $X = F^{-1}(U)$ whenever $F^{-1}$ is cheap (exponential, Weibull, Cauchy, discrete with few outcomes). Also the natural choice with QMC or for correlated/antithetic samples.
- **Normal:** Ziggurat (fastest) or Box–Muller/polar (simple). Library defaults are fine.
- **Gamma:** Marsaglia–Tsang. Beta and Dirichlet from gammas; chi-square, Student-t from normals and gammas.
- **Discrete with many outcomes, many draws:** **alias method**, $O(1)$ per sample after $O(n)$ setup.
- **Multivariate normal:** $\mu + Lz$ with Cholesky $\Sigma = LL^\top$.
- **Unnormalised density, no structure:** rejection sampling in low dimensions; MCMC otherwise.
- **Categorical from logits:** Gumbel-max trick (and Gumbel-softmax for differentiable relaxations).

## Decision table

| Distribution / task                | Method                                                                              | Cost per sample                    | Notes                                     |
| ---------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------- |
| Exponential                        | $-\ln(U)/\lambda$                                                                   | 1 log                              | also Ziggurat                             |
| Normal                             | Ziggurat; Box–Muller / polar                                                        | ~1 uniform (Ziggurat)              | Box–Muller gives two at once              |
| Gamma($\alpha$)                    | Marsaglia–Tsang (normal + uniform, rejection)                                       | ~1.03 iterations                   | $\alpha \lt 1$: boost with $U^{1/\alpha}$ |
| Beta($a, b$)                       | $G_a/(G_a + G_b)$                                                                   | 2 gammas                           | specialised methods for small parameters  |
| Dirichlet($\alpha$)                | normalised independent gammas                                                       | $k$ gammas                         |                                           |
| Poisson($\lambda$)                 | inversion (small $\lambda$), PTRS / rejection (large)                               | $O(1)$                             | normal approximation only for rough work  |
| Binomial($n, p$)                   | inversion (small $np$), BTRS                                                        | $O(1)$                             |                                           |
| Discrete, $n$ outcomes, many draws | alias method (Walker/Vose)                                                          | $O(1)$ (setup $O(n)$)              | static probabilities                      |
| Discrete, changing weights         | binary tree / Fenwick tree of cumulative sums                                       | $O(\log n)$                        | SSA, particle resampling                  |
| Categorical from logits            | Gumbel-max: $\arg\max_i(\ell_i + G_i)$                                              | $n$ uniforms                       | parallel, no normalisation                |
| Multivariate normal                | $\mu + Lz$                                                                          | $O(d^2)$ (after $O(d^3)$ Cholesky) | reuse $L$                                 |
| Uniform on sphere / ball           | normalise a normal vector (× $U^{1/d}$ for ball)                                    | $O(d)$                             | not angles                                |
| Truncated normal                   | inverse CDF in tails, rejection (Robert) / Botev                                    | $O(1)$                             | naive rejection fails in far tails        |
| Arbitrary 1D density               | rejection with good envelope; adaptive rejection (log-concave); numerical inversion | varies                             |                                           |

## Variants

### Inverse transform

- If $U \sim \mathcal U(0, 1)$ then $F^{-1}(U)$ has CDF $F$. For discrete distributions, search the cumulative table.
- Monotone in $U$: preserves stratification, antithetic pairs and QMC structure; enables common random numbers.
- Numerical inversion (e.g. Chebyshev approximations of $F^{-1}$) when no closed form exists.

### Normals

- **Box–Muller:** $Z_1 = \sqrt{-2\ln U_1}\cos(2\pi U_2)$, $Z_2 = \sqrt{-2\ln U_1}\sin(2\pi U_2)$.
- **Polar (Marsaglia):** rejection in the unit disc avoids trig functions.
- **Ziggurat:** precomputed rectangles covering the density; ~99% of samples need one uniform, a comparison and a multiplication.

### Rejection sampling

1. Choose a proposal $g$ and constant $M$ with $f(x) \le Mg(x)$.
2. Draw $X \sim g$, $U \sim \mathcal U(0, 1)$.
3. Accept $X$ if $U \le f(X)/(Mg(X))$, else repeat.

- Acceptance rate $1/M$; degrades exponentially with dimension. Works with unnormalised $f$.
- **Adaptive rejection sampling** for log-concave densities builds piecewise-exponential envelopes on the fly (used in Gibbs samplers).

### Marsaglia–Tsang gamma ($\alpha \ge 1$)

1. $d = \alpha - 1/3$, $c = 1/\sqrt{9d}$.
2. Draw $Z \sim \mathcal N(0, 1)$, $V = (1 + cZ)^3$; reject if $V \le 0$.
3. Draw $U$; accept $dV$ if $\ln U \lt \tfrac12Z^2 + d - dV + d\ln V$.

For $\alpha \lt 1$: sample with $\alpha + 1$ and multiply by $U^{1/\alpha}$.

### Alias method (Vose)

1. Scale probabilities to $np_i$; split into "small" ($\lt 1$) and "large" ($\ge 1$) lists.
2. Repeatedly pair a small $i$ with a large $j$: $\text{prob}[i] = np_i$, $\text{alias}[i] = j$, reduce $np_j$ by $1 - np_i$, reclassify $j$.
3. Sample: pick a column $i$ uniformly, return $i$ with probability $\text{prob}[i]$, else $\text{alias}[i]$.

### Multivariate normal and Gumbel-max

- $\Sigma = LL^\top$ (Cholesky, add jitter if near-singular), $x = \mu + Lz$. With a precision matrix $Q = LL^\top$, solve $L^\top x = z$ instead (sparse GMRFs).
- **Gumbel-max:** $G_i = -\ln(-\ln U_i)$; $\arg\max_i(\log\pi_i + G_i)$ is a sample from Categorical($\pi$). Top-$k$ gives sampling without replacement. Gumbel-softmax replaces argmax with a temperature softmax for gradients.

## Other lenses

- **Measure theory.** Every method is a transformation of uniforms whose pushforward is the target; change of variables gives densities of transformed samples (the basis of normalising flows, see [generative models](/ref/learn/generative-models)).
- **MCMC.** Rejection sampling produces independent exact samples but fails in high dimensions; MCMC produces correlated samples but scales (see [MCMC](/ref/bayes/mcmc)).
- **Differentiability.** Reparametrisation ($x = \mu + \sigma z$) makes samples differentiable in parameters (VAEs, VI); rejection-based samplers need implicit reparametrisation gradients (see [variational inference](/ref/bayes/variational-inference)).

## Pitfalls & diagnostics

- **$\ln(0)$:** uniforms must exclude 0 (use $(0, 1]$ or $1 - U$).
- **Sampling angles uniformly** does not give uniform points on a sphere.
- **Normal approximations** to Poisson/binomial produce negative or non-integer values; use exact samplers.
- **Correlation matrices that aren't PSD** (from mixing estimates) break Cholesky; project to the nearest PSD matrix.
- **Validate samplers** with moments, histograms against the density, and Kolmogorov–Smirnov tests, especially for hand-written ones.

## See also

- [Pseudo-random number generators](/ref/random/prngs)
- [Quasi-Monte Carlo](/ref/random/quasi-monte-carlo)
- [Variance reduction](/ref/random/variance-reduction)
- [MCMC](/ref/bayes/mcmc)

**References:**

- Devroye, _Non-Uniform Random Variate Generation_ (free online).
- Marsaglia & Tsang, "A simple method for generating gamma variables", ACM TOMS (2000).
- Vose, "A linear algorithm for generating random numbers with a given distribution", IEEE TSE (1991).
