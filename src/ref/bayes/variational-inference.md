---
title: 'Variational inference & Laplace approximations'
description: 'ELBO, mean-field and full-rank Gaussian families, ADVI and stochastic VI, the reparametrisation trick, normalising flows, and Laplace/INLA as cheap Gaussian approximations.'
aliases:
  [
    variational inference,
    VI,
    ELBO,
    evidence lower bound,
    KL divergence,
    mean-field,
    ADVI,
    stochastic variational inference,
    SVI,
    black-box VI,
    reparameterization trick,
    score function estimator,
    normalizing flows,
    amortized inference,
    VAE,
    Laplace approximation,
    INLA,
    expectation propagation,
    Pathfinder
  ]
tags: [Bayesian, approximate inference, optimisation]
order: 5
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **VI** turns inference into optimisation: pick a family $q_\phi(\theta)$ and maximise the **ELBO** $= \mathbb E_q[\log p(y, \theta)] - \mathbb E_q[\log q(\theta)]$, equivalent to minimising $\mathrm{KL}(q\Vert p(\cdot\mid y))$.
- **Fast** (orders of magnitude cheaper than MCMC for large data and models) but **biased**: reverse-KL VI underestimates posterior variance and picks one mode.
- **ADVI:** Gaussian $q$ in unconstrained space (mean-field or full-rank), stochastic gradients via the **reparametrisation trick**. Flows give richer $q$.
- **Laplace approximation:** Gaussian at the MAP with covariance = inverse Hessian. Nearly free, good for large data and unimodal posteriors. **INLA** makes this exact-ish for latent Gaussian models.
- Use for: large datasets, amortised inference (VAEs), initialisation of MCMC, quick model iteration. Validate against MCMC on a subset when the decision depends on uncertainty.

## Decision table

| Method                         | Approximating family                      | Cost                             | Captures                                        | Use when                                    |
| ------------------------------ | ----------------------------------------- | -------------------------------- | ----------------------------------------------- | ------------------------------------------- |
| Laplace                        | Gaussian at MAP                           | optimisation + Hessian           | local curvature                                 | large data, unimodal, quick uncertainty     |
| INLA                           | nested Laplace for latent Gaussian models | fast, deterministic              | accurate marginals                              | GLMMs, spatial models, GMRFs                |
| Mean-field ADVI                | diagonal Gaussian                         | cheap per iteration              | means, underestimates variance, no correlations | very large models, rough answers            |
| Full-rank ADVI                 | full Gaussian                             | $O(d^2)$ params                  | correlations                                    | moderate $d$                                |
| Normalising-flow VI            | flow-transformed base density             | more expensive                   | non-Gaussian shapes                             | complex posteriors, GPU                     |
| Stochastic VI (minibatches)    | any                                       | per-minibatch gradients          | —                                               | huge datasets                               |
| Amortised VI (encoder network) | $q_\phi(\theta\mid y)$                    | training once, instant inference | —                                               | many datasets of the same model (VAEs, SBI) |
| Expectation propagation        | exponential family, moment matching       | moderate                         | better variances (inclusive KL)                 | GP classification, specific models          |
| Pathfinder                     | Gaussians along an L-BFGS path            | cheap                            | rough posterior                                 | MCMC initialisation                         |

## Variants

### ELBO and KL

- $\log p(y) = \text{ELBO}(q) + \mathrm{KL}(q\Vert p(\theta\mid y))$, so maximising the ELBO minimises the KL and lower-bounds the evidence.
- **Reverse KL** $\mathrm{KL}(q\Vert p)$ is mode-seeking (zero-forcing): $q$ avoids regions where $p$ is small, underestimating spread and ignoring other modes.
- **Forward KL** $\mathrm{KL}(p\Vert q)$ is mass-covering (used by EP and by neural posterior estimation in SBI).

### Gradient estimators

- **Reparametrisation:** write $\theta = T(\epsilon; \phi)$ with $\epsilon \sim \mathcal N(0, I)$ (e.g. $\theta = \mu + L\epsilon$); then $\nabla_\phi\text{ELBO} = \mathbb E_\epsilon[\nabla_\phi(\log p(y, T(\epsilon; \phi)) - \log q(T(\epsilon; \phi)))]$. Low variance; needs differentiable $\log p$.
- **Score function (REINFORCE):** $\mathbb E_q[(\log p - \log q)\nabla_\phi\log q]$. Works for discrete variables and non-differentiable models; high variance, needs control variates.

### ADVI recipe

1. Transform constrained parameters to $\mathbb R^d$ (log, logit), adding log-Jacobian terms.
2. Choose $q = \mathcal N(\mu, \operatorname{diag}(\sigma^2))$ or $\mathcal N(\mu, LL^\top)$.
3. Optimise the ELBO with Adam using reparametrised Monte Carlo gradients (a few samples per step).
4. Monitor the ELBO (noisy); check convergence by running longer / multiple seeds.

### Normalising flows

- $q$ = pushforward of a simple density through invertible maps (affine coupling, autoregressive, spline flows) with tractable Jacobian determinants.
- Much more flexible than Gaussians; can be trained by reverse KL (VI) or forward KL (from samples).

### Laplace approximation and INLA

- Find the MAP $\hat\theta$; approximate $p(\theta\mid y) \approx \mathcal N(\hat\theta, H^{-1})$ with $H = -\nabla^2\log p(\hat\theta\mid y)$.
- Evidence approximation: $\log p(y) \approx \log p(y, \hat\theta) + \tfrac d2\log 2\pi - \tfrac12\log\det H$ (basis of BIC).
- Parametrisation matters (log scale for positive parameters makes posteriors more Gaussian).
- **INLA:** for latent Gaussian models with few hyperparameters, nested Laplace approximations over the latent field plus numerical integration over hyperparameters; accurate marginals in seconds.

## Other lenses

- **Optimisation.** VI is stochastic optimisation of distribution parameters; natural-gradient VI uses the Fisher metric of $q$ (see [Fisher information](/ref/info/fisher-information-geometry)).
- **Deep learning.** VAEs are amortised VI with neural encoders and decoders; diffusion and flow models share the change-of-variables machinery (see [generative models](/ref/learn/generative-models)).
- **Information theory.** ELBO = expected log-likelihood − KL to the prior: a trade-off between fit and complexity (bits-back coding, MDL).

## Pitfalls & diagnostics

- **Underestimated uncertainty** from mean-field VI; don't use its intervals for decisions without checks.
- **Diagnostics:** Pareto-$\hat k$ of importance weights $p/q$ (PSIS); $\hat k \gt 0.7$ means the approximation is poor. Importance-resampling corrections can fix moderate mismatches.
- **ELBO plateaus** at local optima; try several initialisations, increase Monte Carlo samples for gradients, lower learning rates.
- **Laplace at a boundary or for skewed posteriors** is poor; transform first or use VI/MCMC.
- **Comparing ELBOs across models** is only meaningful as evidence bounds, and loose bounds can mislead.

## See also

- [MCMC](/ref/bayes/mcmc)
- [Entropy, KL & mutual information](/ref/info/entropy-kl-mi)
- [Generative models](/ref/learn/generative-models)
- [Likelihood-free inference](/ref/bayes/likelihood-free)
- [Model comparison](/ref/bayes/model-comparison)

**References:**

- Blei, Kucukelbir & McAuliffe, "Variational inference: a review for statisticians", JASA (2017).
- Kucukelbir et al., "Automatic differentiation variational inference", JMLR (2017).
- Rue, Martino & Chopin, "Approximate Bayesian inference for latent Gaussian models using INLA", JRSS B (2009).
