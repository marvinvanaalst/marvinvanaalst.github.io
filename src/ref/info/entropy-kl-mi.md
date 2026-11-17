---
title: 'Entropy, KL divergence & mutual information'
description: 'Shannon and differential entropy, cross-entropy and KL divergence (and its asymmetry), Jensen–Shannon, mutual information and how to estimate it from samples.'
aliases:
  [
    entropy,
    Shannon entropy,
    differential entropy,
    cross-entropy,
    KL divergence,
    Kullback-Leibler,
    relative entropy,
    forward KL,
    reverse KL,
    Jensen-Shannon divergence,
    mutual information,
    conditional entropy,
    information gain,
    KSG estimator,
    MINE,
    InfoNCE,
    perplexity,
    bits,
    nats,
    data processing inequality,
    channel capacity
  ]
tags: [information theory, probability, machine learning]
order: 1
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Entropy** $H(X) = -\sum p\log p$: uncertainty / average code length. Bits with $\log_2$, nats with $\ln$.
- **Cross-entropy** $H(p, q) = -\mathbb E_p[\log q]$ = entropy + KL. Minimising cross-entropy loss = maximum likelihood.
- **KL divergence** $\mathrm{KL}(p\Vert q) = \mathbb E_p[\log p/q] \ge 0$, asymmetric. **Forward** KL (mass-covering) vs **reverse** KL (mode-seeking) matters for approximate inference.
- **Mutual information** $I(X; Y) = \mathrm{KL}(p_{XY}\Vert p_Xp_Y)$: how much knowing one reduces uncertainty about the other; detects any dependence, not only linear.
- **Estimating MI from samples** is hard: binning is biased; use **KSG** (k-nearest neighbours) for low dimensions, variational bounds (InfoNCE, MINE) for high dimensions.

## Decision table

| Quantity                    | Definition                                                                            | Properties                                                     | Typical use                                                       |
| --------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------- |
| Entropy $H(X)$              | $-\sum_xp(x)\log p(x)$                                                                | $\ge 0$, max for uniform                                       | uncertainty, diversity, coding                                    |
| Differential entropy $h(X)$ | $-\int p\log p$                                                                       | can be negative, not invariant                                 | continuous variables (Gaussian: $\tfrac12\log\det(2\pi e\Sigma)$) |
| Cross-entropy $H(p, q)$     | $-\mathbb E_p\log q$                                                                  | $= H(p) + \mathrm{KL}(p\Vert q)$                               | classification loss                                               |
| KL divergence               | $\mathbb E_p\log(p/q)$                                                                | $\ge 0$, asymmetric, infinite if $q = 0$ where $p \gt 0$       | VI, model comparison, information gain                            |
| Jensen–Shannon              | $\tfrac12\mathrm{KL}(p\Vert m) + \tfrac12\mathrm{KL}(q\Vert m)$, $m = \tfrac{p + q}2$ | symmetric, bounded by $\log 2$, $\sqrt{\text{JS}}$ is a metric | comparing distributions, GANs                                     |
| Mutual information          | $H(X) - H(X\mid Y)$                                                                   | $\ge 0$, 0 iff independent, symmetric                          | dependence, feature selection, design                             |
| Conditional MI              | $I(X; Y\mid Z)$                                                                       | can increase or decrease with conditioning                     | conditional independence, causal discovery                        |
| Perplexity                  | $e^{H}$ (or $2^H$)                                                                    | effective number of choices                                    | language models, t-SNE                                            |

## Variants

### Entropy

- Discrete: maximum $\log K$ for $K$ equally likely outcomes. Chain rule $H(X, Y) = H(X) + H(Y\mid X)$.
- Differential entropy depends on units/parametrisation; differences (KL, MI) are invariant.
- **Maximum entropy principle:** the least committal distribution given constraints (mean → exponential, mean + variance → Gaussian); gives exponential families.

### KL divergence

- **Forward** $\mathrm{KL}(p\Vert q)$ with $p$ the data/truth: penalises $q$ missing mass of $p$ → $q$ covers all modes (maximum likelihood, NPE in SBI, EP).
- **Reverse** $\mathrm{KL}(q\Vert p)$: penalises $q$ putting mass where $p$ is small → $q$ locks onto a mode (variational inference, see [variational inference](/ref/bayes/variational-inference)).
- **Gaussians:** $\mathrm{KL}(\mathcal N_0\Vert\mathcal N_1) = \tfrac12\big[\operatorname{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^\top\Sigma_1^{-1}(\mu_1 - \mu_0) - d + \log\frac{\det\Sigma_1}{\det\Sigma_0}\big]$.
- Second-order expansion: $\mathrm{KL}(p_\theta\Vert p_{\theta + d\theta}) \approx \tfrac12d\theta^\top I(\theta)\,d\theta$ (Fisher information; see [Fisher information](/ref/info/fisher-information-geometry)).

### Mutual information

- $I(X; Y) = H(X) + H(Y) - H(X, Y)$. For jointly Gaussian variables: $-\tfrac12\log(1 - \rho^2)$.
- **Data processing inequality:** for $X \to Y \to Z$, $I(X; Z) \le I(X; Y)$; processing can't create information.
- Invariant under invertible transformations of each variable separately.

### Estimating from samples

- **Plug-in with histograms:** biased (downwards for entropy, upwards for MI) and sensitive to bin width; Miller–Madow and other corrections help a bit.
- **KSG estimator** (Kraskov–Stögbauer–Grassberger): uses distances to the $k$-th nearest neighbour in joint and marginal spaces; good for continuous variables up to ~10 dimensions.
- **Kernel density** estimates: smooth but bandwidth-dependent.
- **Variational lower bounds:** MINE (Donsker–Varadhan), InfoNCE (contrastive; bounded by $\log$ batch size), NWJ; train a critic network. Used for high-dimensional representation learning.
- **Gaussian copula MI:** rank-transform to Gaussian marginals, compute the Gaussian MI; fast lower bound, robust.

## Other lenses

- **Statistics.** Log-likelihood ratios are pointwise KL contributions; AIC estimates expected KL to the truth; likelihood-ratio tests and KL are tied together (Neyman–Pearson, Stein's lemma) (see [model selection](/ref/freq/model-selection)).
- **Bayesian.** Expected information gain from an experiment is the MI between parameters and data (see [experimental design](/ref/freq/experimental-design)); KL from prior to posterior measures what was learned.
- **ML.** Cross-entropy loss, VAEs (ELBO with a KL term), contrastive learning (InfoNCE), information bottleneck and knowledge distillation all use these quantities.
- **Physics.** Entropy is the thermodynamic entropy up to $k_B$; KL is free-energy difference / entropy production in non-equilibrium thermodynamics.

## Pitfalls & diagnostics

- **KL is not a distance:** asymmetric, no triangle inequality; infinite when supports differ. Use JS, Wasserstein or Hellinger when a metric is needed (see [distribution distances](/ref/info/distribution-distances)).
- **Zero probabilities:** add pseudocounts or smoothing before computing KL between empirical histograms.
- **MI estimates without error bars** are meaningless; bootstrap or compare against shuffled data (null distribution).
- **High MI ≠ causation;** conditional MI still can't resolve confounding without causal assumptions.
- **Units:** state bits vs nats; mixing them is a factor $\ln 2$.

## See also

- [Fisher information & geometry](/ref/info/fisher-information-geometry)
- [Distribution distances](/ref/info/distribution-distances)
- [Variational inference](/ref/bayes/variational-inference)
- [Experimental design](/ref/freq/experimental-design)

**References:**

- Cover & Thomas, _Elements of Information Theory_ (2nd ed.).
- MacKay, _Information Theory, Inference, and Learning Algorithms_ (free online).
- Kraskov, Stögbauer & Grassberger, "Estimating mutual information", Phys. Rev. E (2004).
