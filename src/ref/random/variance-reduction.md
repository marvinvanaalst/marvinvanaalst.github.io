---
title: 'Variance reduction'
description: 'Antithetic variates, control variates, importance sampling, stratification, common random numbers and multilevel Monte Carlo: getting more accuracy per sample.'
aliases:
  [
    Monte Carlo variance reduction,
    antithetic variates,
    control variates,
    importance sampling,
    effective sample size,
    self-normalized importance sampling,
    stratified sampling,
    common random numbers,
    multilevel Monte Carlo,
    MLMC,
    conditional Monte Carlo,
    Rao-Blackwellization,
    rare event simulation,
    cross-entropy method,
    splitting
  ]
tags: [Monte Carlo, simulation, statistics]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- Monte Carlo error is $`\sigma/\sqrt N`$. Halving the error costs 4× samples, or a variance reduction that halves $`\sigma`$.
- *Cheap wins:* common random numbers when comparing scenarios; antithetic pairs for monotone integrands; stratification over an important input; conditioning (Rao–Blackwell) where part of the expectation is analytic.
- *Control variates:* subtract a correlated quantity with known mean; variance drops by a factor $`1 - \rho^2`$.
- *Importance sampling:* sample where the integrand matters (rare events, tails); watch the effective sample size, it can blow up the variance.
- *Multilevel Monte Carlo:* for SDE/PDE simulations with discretisation error, combine many cheap coarse samples with few fine ones; cost drops from $`O(\varepsilon^{-3})`$ to $`O(\varepsilon^{-2})`$ (or close).

## Decision table

| Technique | Variance reduction factor | Requirements | Use when |
| --- | --- | --- | --- |
| Antithetic variates | up to large for monotone $`f`$; can increase otherwise | symmetric inputs ($`U`$ and $`1 - U`$, $`Z`$ and $`-Z`$) | monotone responses |
| Common random numbers | large for differences | same random inputs across scenarios | comparing systems, finite-difference sensitivities |
| Control variates | $`1 - \rho^2`$ | correlated variable with known mean | known approximations (linearised model, analytic case) |
| Stratified sampling | removes between-strata variance | stratify a key input | one dominant input |
| Conditional MC / Rao–Blackwell | removes variance of the integrated part | part of the expectation analytically known | mixtures, hierarchical simulations |
| Importance sampling | arbitrary (also arbitrarily worse) | good proposal, density ratios | rare events, tail probabilities, Bayesian evidence |
| Multilevel MC | cost $`O(\varepsilon^{-2})`$ vs $`O(\varepsilon^{-3})`$ | hierarchy of discretisations, coupled samples | SDEs, PDEs with random inputs |
| Splitting / subset simulation | exponential gains for rare events | intermediate levels | very rare events (probabilities $`\lt 10^{-6}`$) |
| QMC | rate improvement | smooth integrand | see [quasi-Monte Carlo](/ref/random/quasi-monte-carlo) |

## Variants

### Antithetic variates

- Pair each sample $`U`$ with $`1 - U`$ (or $`Z`$ with $`-Z`$); average $`\tfrac12(f(U) + f(1 - U))`$.
- Variance of the pair average: $`\tfrac12\sigma^2(1 + \rho)`$ where $`\rho = \operatorname{corr}(f(U), f(1 - U))`$; helps when $`\rho \lt 0`$ (monotone $`f`$).

### Control variates

- Estimator $`\hat\mu = \overline{f} - \beta(\overline{g} - \mathbb E[g])`$ with known $`\mathbb E[g]`$.
- Optimal $`\beta = \operatorname{Cov}(f, g)/\operatorname{Var}(g)`$, estimated by regression on the same samples; residual variance $`\sigma_f^2(1 - \rho_{fg}^2)`$.
- Multiple controls = linear regression on several $`g_j`$. Good controls: a simplified model with known solution, a linear approximation, or the same simulation with an analytically solvable parameter set.

### Importance sampling

- $`\mathbb E_p[f] = \mathbb E_q[f\,p/q]`$: sample from $`q`$, weight by $`w = p/q`$.
- Zero-variance ideal: $`q \propto \lvert f\rvert p`$. Practical: shift/scale towards the important region (exponential tilting for rare events, defensive mixtures with $`p`$).
- *Self-normalised IS* with unnormalised densities: $`\sum w_if_i/\sum w_i`$ (biased but consistent).
- *Effective sample size:* $`\text{ESS} = (\sum w_i)^2/\sum w_i^2`$. Small ESS means a few samples dominate; heavy-tailed weights mean the estimate is unreliable (Pareto-$`\hat k`$ diagnostic, see [model comparison](/ref/bayes/model-comparison)).

### Stratified sampling

- Partition the input space into strata with known probabilities $`p_h`$; sample $`N_h`$ points in each.
- Neyman allocation $`N_h \propto p_h\sigma_h`$ minimises variance; proportional allocation never hurts.

### Multilevel Monte Carlo

- Levels $`l = 0..L`$ with step sizes $`h_l = h_02^{-l}`$; telescoping sum $`\mathbb E[P_L] = \mathbb E[P_0] + \sum_{l=1}^L\mathbb E[P_l - P_{l-1}]`$.
- Estimate each term independently, with the fine and coarse paths in $`P_l - P_{l-1}`$ driven by the *same* random inputs (e.g. Brownian increments summed pairwise for the coarse path).
- Variance of corrections decays with level (rate = strong convergence order), so most samples go to cheap coarse levels. Optimal $`N_l \propto\sqrt{V_l/C_l}`$.

## Other lenses

- *Statistics.* Control variates are regression adjustment; stratification and conditioning are survey-sampling ideas; Rao–Blackwell says conditioning never increases variance.
- *Bayesian computation.* Importance sampling underlies SMC, evidence estimation and PSIS-LOO; tempering and bridge sampling are importance sampling with intermediate distributions (see [sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)).
- *Gradients.* Score-function gradient estimators use baselines (control variates) to be usable; the reparametrisation trick is a variance reduction in disguise (see [variational inference](/ref/bayes/variational-inference)).

## Pitfalls & diagnostics

- *Always estimate variance* of the improved estimator; a bad antithetic or IS scheme can make things worse.
- *IS in high dimensions:* weights degenerate exponentially; it only works with very good proposals.
- *Control variate means must be exact;* an approximately known mean biases the estimator.
- *MLMC requires coupling:* independent fine and coarse samples give no variance reduction.
- *Compare cost-adjusted variance* (variance × cost per sample), not variance alone.

## See also

- [Quasi-Monte Carlo](/ref/random/quasi-monte-carlo)
- [SDEs](/ref/integrate/sdes)
- [Sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)
- [Resampling](/ref/freq/resampling)

*References:*

- Owen, _Monte Carlo Theory, Methods and Examples_ (online book), ch. 8–10.
- Glasserman, _Monte Carlo Methods in Financial Engineering_.
- Giles, "Multilevel Monte Carlo methods", Acta Numerica (2015).
