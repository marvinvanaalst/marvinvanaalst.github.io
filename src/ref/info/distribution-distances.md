---
title: 'Distances between distributions'
description: 'Total variation, Hellinger, KL/JS, Wasserstein and optimal transport (Sinkhorn), MMD and kernel two-sample tests: which divergence for which job and how to compute it from samples.'
aliases:
  [
    statistical distance,
    divergence,
    total variation,
    Hellinger distance,
    Wasserstein distance,
    earth mover's distance,
    optimal transport,
    Sinkhorn,
    entropic regularization,
    sliced Wasserstein,
    MMD,
    maximum mean discrepancy,
    kernel two-sample test,
    energy distance,
    Kolmogorov-Smirnov,
    f-divergence,
    integral probability metric,
    Bhattacharyya
  ]
tags: [probability, information theory, optimal transport, statistics]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **f-divergences** (KL, JS, TV, Hellinger, χ²) compare densities pointwise: blind to geometry, infinite or saturated when supports don't overlap.
- **Wasserstein / optimal transport** distances measure how far mass must move: meaningful for non-overlapping supports, respect the geometry of the space. Compute with **Sinkhorn** (entropic OT) in $O(n^2)$ per iteration, exact in 1D by sorting.
- **MMD** compares kernel mean embeddings: simple unbiased estimators from samples, the basis of **kernel two-sample tests**.
- **Two samples, need a test:** MMD or energy-distance permutation test (multivariate), Kolmogorov–Smirnov (1D).
- **Learning generative models:** Wasserstein or MMD losses behave better than JS when supports are disjoint.

## Decision table

| Distance / divergence | Definition (schematic)                                                                       | From samples                    | Metric                       | Use when                                         |
| --------------------- | -------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------------- | ------------------------------------------------ |
| Total variation       | $\sup_A\lvert P(A) - Q(A)\rvert = \tfrac12\int\lvert p - q\rvert$                            | hard (needs densities)          | yes                          | worst-case event probability differences         |
| Hellinger             | $\big(\tfrac12\int(\sqrt p - \sqrt q)^2\big)^{1/2}$                                          | needs densities                 | yes                          | bounded, related to TV and Bhattacharyya         |
| KL                    | $\int p\log(p/q)$                                                                            | needs densities / estimators    | no                           | likelihood, VI, information gain                 |
| Jensen–Shannon        | symmetrised KL to the mixture                                                                | classifier-based estimates      | $\sqrt{\cdot}$ yes           | bounded comparison, original GANs                |
| Kolmogorov–Smirnov    | $\sup_x\lvert F_P(x) - F_Q(x)\rvert$                                                         | sort, $O(n\log n)$              | yes                          | 1D two-sample tests                              |
| Wasserstein-1 (EMD)   | $\inf_\gamma\mathbb E_\gamma\lVert x - y\rVert$                                              | 1D: sorting; else LP / Sinkhorn | yes                          | geometry-aware comparison, histograms, shapes    |
| Wasserstein-2         | $\big(\inf_\gamma\mathbb E\lVert x - y\rVert^2\big)^{1/2}$                                   | Gaussians: closed form          | yes                          | interpolation between distributions, flows       |
| Sinkhorn divergence   | entropic OT, debiased                                                                        | $O(n^2)$ per iteration, GPU     | ~                            | large point clouds, differentiable losses        |
| Sliced Wasserstein    | average of 1D Wasserstein over projections                                                   | $O(Ln\log n)$                   | yes                          | high dimensions, cheap                           |
| MMD                   | $\lVert\mu_P - \mu_Q\rVert_{\mathcal H}$ in an RKHS                                          | $O(n^2)$ unbiased U-statistic   | yes (characteristic kernels) | two-sample tests, generative model training      |
| Energy distance       | $2\mathbb E\lVert X - Y\rVert - \mathbb E\lVert X - X'\rVert - \mathbb E\lVert Y - Y'\rVert$ | $O(n^2)$                        | yes                          | multivariate two-sample tests (MMD special case) |

## Variants

### f-divergences

- $D_f(P\Vert Q) = \mathbb E_Q[f(p/q)]$ with convex $f$, $f(1) = 0$: KL ($t\log t$), reverse KL, TV ($\tfrac12\lvert t - 1\rvert$), Hellinger, χ² ($(t - 1)^2$), JS.
- Inequalities: $\mathrm{TV} \le \sqrt{\mathrm{KL}/2}$ (Pinsker); $H^2 \le \mathrm{TV} \le \sqrt2H$.
- Estimate from samples with density-ratio estimation or classifiers (variational representation, as in f-GANs).

### Optimal transport and Wasserstein

- **Kantorovich problem:** minimise $\sum_{ij}C_{ij}\gamma_{ij}$ over couplings $\gamma$ with given marginals: an LP, $O(n^3\log n)$ exactly.
- **1D:** $W_p$ between empirical distributions = matching sorted samples; $W_1 = \int\lvert F_P - F_Q\rvert$.
- **Gaussians:** $W_2^2 = \lVert\mu_1 - \mu_2\rVert^2 + \operatorname{tr}\big(\Sigma_1 + \Sigma_2 - 2(\Sigma_2^{1/2}\Sigma_1\Sigma_2^{1/2})^{1/2}\big)$ (as in FID scores).
- **Dual (Kantorovich–Rubinstein):** $W_1 = \sup_{\lVert f\rVert_{\text{Lip}} \le 1}\mathbb E_Pf - \mathbb E_Qf$ (WGAN critics).

### Sinkhorn

1. Kernel $K = e^{-C/\varepsilon}$ from the cost matrix.
2. Alternate scaling: $u \leftarrow a/(Kv)$, $v \leftarrow b/(K^\top u)$.
3. Coupling $\gamma = \operatorname{diag}(u)K\operatorname{diag}(v)$.

- Work in the log domain for small $\varepsilon$; debias with the Sinkhorn divergence $S_\varepsilon(P, Q) = OT_\varepsilon(P, Q) - \tfrac12OT_\varepsilon(P, P) - \tfrac12OT_\varepsilon(Q, Q)$.
- Differentiable, GPU-friendly; $\varepsilon$ trades accuracy for speed and smoothness.

### MMD and kernel two-sample tests

- $\mathrm{MMD}^2 = \mathbb E k(X, X') + \mathbb E k(Y, Y') - 2\mathbb E k(X, Y)$; unbiased U-statistic from samples.
- Characteristic kernels (Gaussian, Matérn) make MMD = 0 iff $P = Q$.
- **Test:** permutation null distribution (or asymptotic approximations); kernel bandwidth by the median heuristic or by optimising test power on held-out data.

## Other lenses

- **Optimisation.** OT is a linear program (and entropic OT a matrix-scaling problem); Wasserstein gradient flows describe diffusion and many PDEs as steepest descent of free energies (see [convex duality](/ref/optimise/convex-duality)).
- **Kernels.** MMD lives in the RKHS of the kernel used by GPs and SVMs; Stein discrepancies are kernel-based goodness-of-fit measures needing only the score of the model (see [Gaussian processes](/ref/approx/gaussian-processes)).
- **Biology.** OT maps cells between time points in single-cell trajectory inference (Waddington-OT); Wasserstein distances compare histograms, spectra and shapes respecting the axis geometry.

## Pitfalls & diagnostics

- **Disjoint supports:** KL infinite, JS saturated at $\log 2$, gradients vanish; use Wasserstein or MMD.
- **Curse of dimensionality:** empirical Wasserstein converges as $n^{-1/d}$; use sliced, entropic or MMD variants in high dimensions.
- **Kernel/bandwidth choice** dominates MMD test power; tune on separate data.
- **Scale and units** matter for Wasserstein (cost is in data units); standardise features or pick a meaningful cost.
- **Report a null reference:** distances between two samples from the same distribution give the noise floor.

## See also

- [Entropy, KL & mutual information](/ref/info/entropy-kl-mi)
- [Hypothesis tests](/ref/freq/hypothesis-tests)
- [Generative models](/ref/learn/generative-models)
- [Convex duality & proximal methods](/ref/optimise/convex-duality)

**References:**

- Peyré & Cuturi, "Computational optimal transport", Found. Trends Mach. Learn. (2019).
- Gretton et al., "A kernel two-sample test", JMLR (2012).
- Villani, _Optimal Transport: Old and New_.
