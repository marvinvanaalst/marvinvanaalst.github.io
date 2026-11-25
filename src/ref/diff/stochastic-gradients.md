---
title: 'Stochastic gradient estimators'
description: 'Derivatives of expectations: pathwise (reparameterisation), score-function (likelihood-ratio), measure-valued and coupled finite-difference estimators, their bias and variance, and control variates.'
aliases:
  [
    gradient of an expectation,
    pathwise derivative,
    reparameterization trick,
    reparameterisation trick,
    infinitesimal perturbation analysis,
    IPA,
    score function estimator,
    likelihood ratio method,
    REINFORCE,
    measure-valued derivative,
    weak derivative,
    common random numbers,
    Gumbel-softmax,
    control variates,
    Monte Carlo gradient
  ]
tags: [differentiation, Monte Carlo, stochastic optimisation]
order: 5
updated: '2026-11-20'
published: true
template: ref
---

## TL;DR

To differentiate $`\nabla_\theta\,\mathbb E_{x\sim p_\theta}[f(x)]`$ (or $`\mathbb E[f(x, \theta)]`$) by Monte Carlo:

- *Pathwise / reparameterisation*: write $`x = g(\varepsilon, \theta)`$ with $`\varepsilon`$ independent of $`\theta`$, then $`\nabla_\theta\mathbb E[f] = \mathbb E[\nabla_x f\,\nabla_\theta g]`$. Lowest variance; needs $`f`$ and the sampler to be differentiable.
- *Score function / likelihood ratio (REINFORCE)*: $`\nabla_\theta\mathbb E[f] = \mathbb E[f(x)\,\nabla_\theta\log p_\theta(x)]`$. Works for discrete variables and black-box $`f`$; high variance, needs control variates.
- *Measure-valued*: write $`\nabla_\theta p_\theta`$ as a weighted difference of two distributions and sample both. Low variance, unbiased, but needs a decomposition per distribution and more samples per parameter.
- *Finite differences with common random numbers*: biased, simple, works for anything you can re-run with the same seed.

## Decision table

| Estimator | Bias | Variance | Needs | Use when |
| --- | --- | --- | --- | --- |
| Pathwise (reparameterisation, IPA) | none | low | differentiable $`f`$ and sampler $`g`$ | continuous latents, VAEs, smooth simulators |
| Score function (likelihood ratio) | none | high, grows with dimension and horizon | $`\nabla_\theta\log p_\theta`$, any $`f`$ | discrete variables, black-box $`f`$, RL |
| Measure-valued (weak derivative) | none | low | known decomposition of $`\nabla_\theta p`$ | few parameters, standard distributions |
| Finite differences + CRN | $`O(h)`$ or $`O(h^2)`$ | $`O(1/h)`$–$`O(1)`$ with good coupling | re-runnable simulator with fixed seeds | black-box simulators, gradient checks |
| Relaxations (Gumbel-softmax, concrete) | yes (temperature) | low | continuous relaxation of discrete variables | discrete latents where bias is acceptable |

## Variants

### Pathwise (reparameterisation)

- Sample $`\varepsilon\sim q`$ (fixed), set $`x = g(\varepsilon, \theta)`$, e.g. $`x = \mu + \sigma\varepsilon`$ for a Gaussian. Then differentiate inside the expectation with ordinary AD.
- Called *infinitesimal perturbation analysis (IPA)* in simulation. Invalid when $`f\circ g`$ is discontinuous in $`\theta`$ (thresholds, discrete events): the estimate is then biased, often silently.
- Implicit reparameterisation handles distributions without a simple $`g`$ (Gamma, Beta) by differentiating the CDF: $`\nabla_\theta x = -\nabla_\theta F(x;\theta)/p_\theta(x)`$.

### Score function (likelihood ratio)

- From $`\nabla_\theta p_\theta = p_\theta\nabla_\theta\log p_\theta`$: $`\nabla_\theta\mathbb E[f] = \mathbb E[f(x)\,\nabla_\theta\log p_\theta(x)]`$.
- For a path of a Markov process the score is a sum over steps, so variance grows with the horizon. For jump processes and SDEs this is the *Girsanov* / likelihood-ratio sensitivity (see [ODE sensitivities](/ref/diff/ode-sensitivities)).
- Always use a *baseline*: $`\mathbb E[(f - b)\nabla\log p]`$ is unbiased for any $`b`$ independent of $`x`$; $`b \approx \mathbb E f`$ often cuts variance by orders of magnitude.

### Measure-valued derivatives

- Decompose $`\partial_{\theta_i}p_\theta = c_i\,(p_i^+ - p_i^-)`$ with two probability distributions (for a Gaussian mean: $`x = \mu \pm \sigma W`$ with $`W \sim \mathrm{Weibull}(\sqrt2, 2)`$ and $`c = 1/(\sigma\sqrt{2\pi})`$). Then $`\partial_{\theta_i}\mathbb E f = c_i(\mathbb E_{p^+}f - \mathbb E_{p^-}f)`$, estimated with coupled samples.
- Unbiased, low variance, $`f`$ need not be differentiable. Costs two evaluations per parameter, so it scales like finite differences in the number of parameters.

### Finite differences with common random numbers

- Estimate $`[\hat F(\theta + h) - \hat F(\theta - h)]/2h`$ with the *same random stream* for both evaluations. Without coupling the variance is $`O(1/h^2)`$; with good coupling $`O(1/h)`$ or $`O(1)`$ (see [variance reduction](/ref/random/variance-reduction)).
- For stochastic reaction networks, couple per-reaction Poisson streams (common reaction paths, Anderson's split coupling) rather than one global seed.
- SPSA is the randomised-direction version when there are many parameters (see [finite differences](/ref/diff/finite-differences)).

### Control variates and Rao–Blackwellisation

- Subtract a correlated quantity with known expectation: $`\hat g - c(h - \mathbb E h)`$. Baselines are the simplest case. Learned control variates (REBAR, RELAX) combine score-function and relaxed pathwise estimators to keep unbiasedness at low variance.
- Integrate out what you can analytically (Rao–Blackwellisation) before estimating.

## Other lenses

- *Variational inference.* The ELBO gradient is the textbook use: reparameterisation for continuous latents, score function for discrete ones (see [variational inference](/ref/bayes/variational-inference)).
- *Statistics.* The score $`\nabla_\theta\log p_\theta`$ is the same object as in maximum likelihood; its covariance is the Fisher information (see [MLE](/ref/freq/estimation-mle)).
- *Reinforcement learning.* The policy gradient theorem is the score-function estimator over trajectories; actor–critic methods are baselines.

## Pitfalls & diagnostics

- *Silent bias of pathwise gradients* when the simulator has discontinuities in $`\theta`$ (event thresholds, `if` on random draws). Compare against a score-function or finite-difference estimate on a small case.
- *Variance, not bias, usually dominates.* Report the standard error of the gradient estimate; if it exceeds the gradient, the optimiser is doing a random walk.
- *Fixed seeds* make stochastic objectives deterministic for finite differences and line searches, but the optimum then depends on the seed; re-check at the end with fresh samples.

## See also

- [Finite differences](/ref/diff/finite-differences)
- [Automatic differentiation](/ref/diff/automatic-differentiation)
- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Variance reduction](/ref/random/variance-reduction)
- [Stochastic simulation](/ref/integrate/stochastic-simulation)

*References:*

- Mohamed, Rosca, Figurnov & Mnih, "Monte Carlo gradient estimation in machine learning", JMLR (2020).
- Glasserman, _Monte Carlo Methods in Financial Engineering_, ch. 7 (2003).
- Figurnov, Mohamed & Mnih, "Implicit reparameterization gradients", NeurIPS (2018).
- Grathwohl et al., "Backpropagation through the void: optimizing control variates for black-box gradient estimation", ICLR (2018).
