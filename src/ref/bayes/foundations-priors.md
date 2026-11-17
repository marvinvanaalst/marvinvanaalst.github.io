---
title: 'Bayesian foundations & priors'
description: "Bayes' rule in practice, conjugate models, choosing priors (weakly informative, reference, hierarchical), non-centred parametrisation and the Bayesian workflow."
aliases:
  [
    Bayes theorem,
    posterior,
    prior,
    likelihood,
    evidence,
    conjugate prior,
    Beta-binomial,
    normal-normal,
    gamma-Poisson,
    weakly informative prior,
    Jeffreys prior,
    reference prior,
    improper prior,
    hierarchical model,
    partial pooling,
    non-centered parametrization,
    prior predictive check,
    posterior predictive check,
    Bayesian workflow,
    MAP,
    credible interval
  ]
tags: [Bayesian, priors, hierarchical models]
order: 1
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- $p(\theta\mid y) \propto p(y\mid\theta)\,p(\theta)$: posterior ∝ likelihood × prior. The normalising constant $p(y)$ (evidence) is only needed for model comparison.
- **Conjugate pairs** give closed-form posteriors: use them for quick models, sanity checks and Gibbs updates.
- **Priors:** prefer **weakly informative** priors on a sensible scale (e.g. log-normal on rate constants spanning the plausible orders of magnitude). Check them with **prior predictive simulations**.
- **Hierarchical models** share information across groups (partial pooling). Use the **non-centred parametrisation** when groups have little data, otherwise samplers struggle (funnels).
- Report posterior summaries (medians, credible intervals) and **posterior predictive checks**, not just point estimates.

## Decision table

| Likelihood                        | Conjugate prior           | Posterior                                              | Typical use                   |
| --------------------------------- | ------------------------- | ------------------------------------------------------ | ----------------------------- |
| Bernoulli / binomial              | Beta($a, b$)              | Beta($a + k$, $b + n - k$)                             | proportions, success rates    |
| Poisson                           | Gamma($\alpha, \beta$)    | Gamma($\alpha + \sum y$, $\beta + n$)                  | counts, rates                 |
| Normal, known variance            | Normal($\mu_0, \tau_0^2$) | Normal, precision-weighted mean                        | measurements with known noise |
| Normal, unknown mean and variance | Normal-inverse-gamma      | Normal-inverse-gamma (Student-t marginal for the mean) | small-sample means            |
| Multinomial                       | Dirichlet($\alpha$)       | Dirichlet($\alpha + n$)                                | categorical frequencies       |
| Linear regression                 | Normal (+ inverse-gamma)  | Normal (ridge-like)                                    | Bayesian linear models        |
| Exponential                       | Gamma                     | Gamma                                                  | waiting times                 |

| Prior type                                   | Idea                                            | Use when                                            |
| -------------------------------------------- | ----------------------------------------------- | --------------------------------------------------- |
| Informative                                  | encodes real previous knowledge                 | literature values, previous experiments             |
| Weakly informative                           | rules out absurd values, otherwise broad        | **default** for scientific models                   |
| Flat / improper                              | constant density                                | rarely: can give improper posteriors, not invariant |
| Jeffreys / reference                         | invariant, "objective"                          | simple models, reporting where objectivity matters  |
| Hierarchical                                 | parameters drawn from a population distribution | repeated groups (patients, cell lines, replicates)  |
| Shrinkage (horseshoe, regularised horseshoe) | most effects ~0, a few large                    | sparse regression                                   |

## Variants

### Conjugacy example: Beta-binomial

- Prior Beta($a, b$) on a success probability, $k$ successes in $n$ trials: posterior Beta($a + k$, $b + n - k$).
- Posterior mean $(a + k)/(a + b + n)$: a weighted average of prior mean and data proportion, with the prior acting like $a + b$ pseudo-observations.

### Choosing priors

- **Work on interpretable scales:** log-parameters for positive quantities; standardised predictors in regression.
- **Prior predictive check:** sample $\theta \sim p(\theta)$, simulate data $y \sim p(y\mid\theta)$; the simulated data should look plausible (no concentrations of $10^{20}$ M, no negative times).
- Typical weakly informative choices: Normal(0, 1–2.5) on standardised regression coefficients; half-Normal/half-Student-t on scale parameters; LKJ(2) on correlation matrices; log-normal on rate constants.
- **Sensitivity:** refit with wider/narrower priors; conclusions that flip are prior-driven.

### Hierarchical models

- Group parameters $\theta_j \sim \mathcal N(\mu, \tau^2)$, hyperpriors on $\mu, \tau$.
- **Partial pooling:** groups with little data shrink towards $\mu$, groups with lots of data stay close to their own estimate. Avoids both overfitting (no pooling) and ignoring differences (complete pooling).
- **Centred** $\theta_j \sim \mathcal N(\mu, \tau^2)$ vs **non-centred** $\theta_j = \mu + \tau z_j$, $z_j \sim \mathcal N(0, 1)$. With weak data per group, the centred form has a funnel geometry (small $\tau$ forces $\theta_j$ together) that HMC can't explore; non-centred removes it. With strong data, centred is better.

### The Bayesian workflow

1. Build the model; prior predictive checks.
2. Fit (MCMC / VI); check computation (R-hat, ESS, divergences; see [MCMC](/ref/bayes/mcmc), [HMC](/ref/bayes/hmc-nuts)).
3. Posterior predictive checks: simulate replicated data from the posterior, compare with observed data (test statistics, plots).
4. Compare / expand models (LOO, see [model comparison](/ref/bayes/model-comparison)); iterate.
5. Simulation-based calibration to validate the inference pipeline on synthetic data.

## Other lenses

- **Optimisation.** The MAP estimate maximises log posterior = log likelihood + log prior: regularised MLE (Gaussian prior = ridge, Laplace = LASSO). MAP ignores posterior volume and is not invariant to reparametrisation.
- **Frequentist statistics.** With lots of data and regular models, posteriors concentrate around the MLE with covariance $\approx$ inverse Fisher information (Bernstein–von Mises), so credible and confidence intervals agree; they differ with little data, boundaries or non-identifiability (see [estimation & MLE](/ref/freq/estimation-mle)).
- **Information theory.** Jeffreys prior $\propto\sqrt{\det I(\theta)}$ is uniform in the Fisher metric; the posterior update is the minimum-information change consistent with the data (see [Fisher information](/ref/info/fisher-information-geometry)).

## Pitfalls & diagnostics

- **Improper or very flat priors** can produce improper posteriors or absurd prior predictive data (and huge Bayes factor sensitivity).
- **Uniform priors on positive parameters** over many orders of magnitude put almost all mass on the largest values; use log-uniform or log-normal.
- **Non-identifiable parameters:** the posterior equals the prior along those directions; that's informative, not a failure, but report it.
- **Funnels and divergences** in hierarchical models: reparametrise (non-centred) before tuning samplers.
- **Interpret posteriors, not MAP points,** especially for multimodal or skewed posteriors.

## See also

- [MCMC](/ref/bayes/mcmc)
- [HMC & NUTS](/ref/bayes/hmc-nuts)
- [Model comparison](/ref/bayes/model-comparison)
- [Estimation & MLE](/ref/freq/estimation-mle)
- [Probabilistic programming](/ref/programs/probabilistic-programming)

**References:**

- Gelman et al., _Bayesian Data Analysis_ (3rd ed.).
- McElreath, _Statistical Rethinking_ (2nd ed.).
- Gelman et al., "Bayesian workflow" (2020).
