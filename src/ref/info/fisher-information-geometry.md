---
title: 'Fisher information & information geometry'
description: 'Fisher information as curvature and as a metric, the natural gradient, Jeffreys prior, K-FAC, sloppiness, and exponential families in dual coordinates.'
aliases:
  [
    Fisher information,
    Fisher information matrix,
    FIM,
    observed information,
    information geometry,
    Fisher-Rao metric,
    natural gradient,
    K-FAC,
    Jeffreys prior,
    sloppiness,
    exponential family,
    natural parameters,
    mean parameters,
    dual coordinates,
    Cramer-Rao,
    Gauss-Newton matrix,
    empirical Fisher,
    mirror descent
  ]
tags: [information theory, statistics, geometry, optimisation]
order: 2
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Fisher information* $`I(\theta) = \mathbb E[\nabla\log p\,\nabla\log p^\top] = -\mathbb E[\nabla^2\log p]`$: curvature of the expected log-likelihood; inverse = lower bound on estimator covariance (Cramér–Rao).
- It is a *Riemannian metric* on the parameter space: the local KL divergence between nearby models, $`\mathrm{KL} \approx \tfrac12d\theta^\top I\,d\theta`$.
- *Natural gradient* $`I^{-1}\nabla L`$: steepest descent in distribution space, invariant to reparametrisation. Approximations (K-FAC, Shampoo-like, diagonal) make it practical for neural nets.
- *Jeffreys prior* $`\propto\sqrt{\det I}`$: the reparametrisation-invariant "uniform" prior.
- *Sloppiness:* FIM eigenvalues of mechanistic models span many decades; most parameter combinations barely matter.

## Decision table

| Quantity / method | Formula | Use for |
| --- | --- | --- |
| Expected Fisher information | $`\mathbb E_{y\sim p_\theta}[ss^\top]`$ | design, Jeffreys prior, theory |
| Observed information | $`-\nabla^2\ell(\hat\theta)`$ | standard errors after fitting |
| Gaussian model FIM | $`S^\top\Sigma^{-1}S`$ (sensitivities $`S`$) | ODE models, experimental design, identifiability |
| Gauss–Newton matrix | $`J^\top J`$ | = FIM for Gaussian least squares |
| Empirical Fisher | $`\frac1n\sum_is_is_i^\top`$ at data | cheap, but biased approximation (not the FIM) |
| Natural gradient | $`\Delta\theta = -\eta I^{-1}\nabla L`$ | invariant optimisation, VI, policy gradients |
| K-FAC | Kronecker-factored layer-wise FIM | approximate natural gradient for neural nets |
| Mirror descent | gradient steps in dual (mean) coordinates | exponential families, simplex constraints |
| Jeffreys prior | $`\sqrt{\det I(\theta)}`$ | default prior for simple models |

## Variants

### Fisher information as curvature and metric

- Score $`s = \nabla_\theta\log p(y\mid\theta)`$ has mean zero; its covariance is $`I(\theta)`$.
- Under regularity, $`I = -\mathbb E[\nabla^2\log p]`$: average curvature of the log-likelihood.
- *Information is additive* over independent observations; for experiment design, information matrices of experiments add (see [experimental design](/ref/freq/experimental-design)).
- *Metric:* the Fisher–Rao metric is the unique (up to scale) metric invariant under sufficient statistics (Čencov).

### Natural gradient

- Solve $`\min_{d\theta}L(\theta + d\theta)`$ subject to $`\mathrm{KL}(p_\theta\Vert p_{\theta + d\theta}) \le \epsilon`$ → $`d\theta \propto -I^{-1}\nabla L`$.
- Same step regardless of parametrisation (unlike plain gradient descent).
- For Gaussian least squares, natural gradient = Gauss–Newton; for exponential families, natural gradient in natural parameters = plain gradient in mean parameters (mirror descent duality).
- *K-FAC:* approximate each layer's FIM block by $`A\otimes G`$ (activation and gradient covariances), invert the small factors. *Natural-gradient VI* updates variational parameters with the Fisher of $`q`$.

### Exponential families and dual coordinates

- $`p(y\mid\eta) = h(y)\exp(\eta^\top T(y) - A(\eta))`$; natural parameters $`\eta`$, mean parameters $`\mu = \nabla A(\eta)`$.
- $`I(\eta) = \nabla^2A(\eta) = \operatorname{Cov}(T)`$; the two coordinate systems are dual (Legendre transform), flat in different senses (e- and m-flatness).
- MLE: match sufficient statistics, $`\mu(\hat\eta) = \bar T`$.

### Sloppiness and FIM spectra

- Eigenvalues of the FIM (in log-parameters) for many mechanistic models are roughly evenly spaced in log scale over $`10^6`$–$`10^{12}`$.
- Stiff eigendirections are well determined, sloppy ones barely affect predictions. Implications: individual parameters are poorly identifiable, yet predictions can be precise; model reduction by removing sloppy combinations (manifold boundary approximation).

## Other lenses

- *Statistics.* Cramér–Rao bound, asymptotic normality of the MLE and Wald intervals all use $`I^{-1}`$ (see [estimation & MLE](/ref/freq/estimation-mle)).
- *Optimisation.* Adam's second-moment estimate resembles a diagonal empirical Fisher; trust-region methods with KL constraints (TRPO) are natural-gradient methods (see [stochastic gradient](/ref/optimise/stochastic-gradient)).
- *Differential geometry.* Geodesics, curvature and volumes in parameter space: the Fisher volume counts distinguishable models (MDL, Jeffreys prior normalisation) (see [manifold optimisation](/ref/optimise/manifold-optimisation)).

## Pitfalls & diagnostics

- *Empirical Fisher ≠ Fisher:* using observed gradients' outer products as a curvature estimate can be badly wrong away from the optimum.
- *Local quantity:* the FIM at one point says nothing about nonlinear, global uncertainty; use profile likelihood or sampling for that (see [identifiability](/ref/freq/identifiability)).
- *Singular FIM* means non-identifiable directions; natural gradient needs damping ($`I + \lambda I`$).
- *Parametrisation for eigen-analysis:* use log-parameters so eigenvalues compare relative changes.
- *Jeffreys prior* can be improper or behave oddly in multiparameter models; reference priors refine it.

## See also

- [Estimation & MLE](/ref/freq/estimation-mle)
- [Identifiability](/ref/freq/identifiability)
- [Experimental design](/ref/freq/experimental-design)
- [Entropy, KL & mutual information](/ref/info/entropy-kl-mi)
- [Stochastic gradient methods](/ref/optimise/stochastic-gradient)

*References:*

- Amari, _Information Geometry and Its Applications_.
- Martens, "New insights and perspectives on the natural gradient method", JMLR (2020).
- Transtrum, Machta, Brown, Daniels, Myers & Sethna, "Perspective: sloppiness and emergent theories in physics, biology, and beyond", J. Chem. Phys. (2015).
