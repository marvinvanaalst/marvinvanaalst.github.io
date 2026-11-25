---
title: 'Feature attribution & model explanation'
description: 'Permutation importance, partial dependence and ICE, SHAP (Kernel, Tree, Deep), LIME, integrated gradients and saliency: what each explanation method measures and how it misleads.'
aliases:
  [
    explainable AI,
    XAI,
    interpretability,
    feature importance,
    permutation importance,
    partial dependence plot,
    PDP,
    ICE plots,
    accumulated local effects,
    ALE,
    SHAP,
    Shapley values,
    KernelSHAP,
    TreeSHAP,
    DeepSHAP,
    LIME,
    integrated gradients,
    saliency maps,
    gradient times input,
    SmoothGrad,
    attention maps,
    counterfactual explanations,
    global vs local explanations
  ]
tags: [explainability, machine learning, interpretability]
order: 1
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Global, model-agnostic importance:* *permutation importance* on held-out data (drop in performance when a feature is shuffled).
- *How does the prediction depend on a feature?* *ALE* plots (robust to correlated features) or PDP + ICE curves.
- *Local attribution of one prediction:* *SHAP* (Shapley values; exact and fast for trees with TreeSHAP), integrated gradients for differentiable models.
- *LIME* and plain saliency maps are fragile; use as rough hints only.
- Attributions explain the *model*, not the world: they become causal statements only with causal assumptions (see [causal graphs](/ref/explain/causal-graphs-scms)).
- Correlated features are the main source of misleading explanations for every method.

## Decision table

| Method | Scope | Model access | Cost | Measures | Pitfalls |
| --- | --- | --- | --- | --- | --- |
| Permutation importance | global | predictions | $`O(dr)`$ predictions ($`r`$ repeats) | performance drop when feature is broken | correlated features share/hide importance; extrapolation |
| Impurity importance (trees) | global | tree internals | free | split gain | biased to high-cardinality features |
| PDP | global | predictions | grid × $`n`$ predictions | average marginal effect | extrapolates into impossible feature combinations |
| ICE | local curves | predictions | as PDP | per-instance effect curves | same extrapolation issue; heterogeneity visible |
| ALE | global | predictions | similar | local effects accumulated | harder to read, but robust to correlation |
| KernelSHAP | local (→ global) | predictions | many evaluations per instance | Shapley values with a background distribution | slow; depends on background; assumes independence (interventional) |
| TreeSHAP | local (→ global) | tree ensembles | polynomial, fast | exact Shapley values for trees | path-dependent vs interventional variants differ |
| DeepSHAP / GradientSHAP | local | gradients / network | cheap | approximate Shapley values | approximation quality |
| Integrated gradients | local | gradients | \~50–300 gradients | path integral of gradients from a baseline | baseline choice |
| Saliency, gradient × input | local | gradients | 1 gradient | local sensitivity | noisy, fails sanity checks |
| LIME | local | predictions | sampling + linear fit | local linear surrogate | unstable, neighbourhood definition |
| Global surrogate model | global | predictions | train a simple model | interpretable approximation | fidelity must be checked |

## Variants

### Permutation importance

1. Compute the model's score on validation data.
2. For each feature $`j`$: shuffle column $`j`$, recompute the score; importance = score drop (repeat and average).
3. Report with spread across repeats.

- Measures reliance of this model on the feature; correlated features can substitute for each other (each looks unimportant). Use grouped permutation or conditional permutation for correlated groups.

### PDP, ICE and ALE

- *PDP:* $`\hat f_S(x_S) = \frac1n\sum_i\hat f(x_S, x_{C}^{(i)})`$: average prediction with feature $`S`$ set to a value.
- *ICE:* the individual curves before averaging; reveal interactions (diverging curves).
- *ALE:* average the local changes of predictions within small intervals of the feature, using only observed neighbours, then accumulate; avoids evaluating the model at unrealistic combinations.

### Shapley values and SHAP

- Shapley value of feature $`j`$: average marginal contribution over all orderings, $`\phi_j = \sum_{S\subseteq F\setminus j}\frac{\lvert S\rvert!(d - \lvert S\rvert - 1)!}{d!}[v(S\cup j) - v(S)]`$.
- Axioms: efficiency (attributions sum to prediction − baseline), symmetry, dummy, additivity.
- *Value function* choice matters: interventional (break correlations, "what does the model do") vs observational/conditional (respect correlations, "what does the data say").
- *KernelSHAP:* weighted linear regression approximation; *TreeSHAP:* exact polynomial-time algorithm for trees; global importance = mean $`\lvert\phi_j\rvert`$.

### Gradient-based methods

- *Integrated gradients:* $`\mathrm{IG}_j = (x_j - x'_j)\int_0^1\partial_jf(x' + \alpha(x - x'))\,d\alpha`$ (Riemann sum with \~50–300 steps); satisfies completeness; baseline $`x'`$ (zeros, blurred input, mean) must be meaningful.
- *SmoothGrad:* average gradients over noisy copies to reduce noise.
- Sanity checks: attributions should change when model weights are randomised; many saliency methods fail this.

## Other lenses

- *Sensitivity analysis.* Local derivatives and Sobol indices in mechanistic modelling are the same idea; Shapley effects unify variance-based sensitivity with Shapley attribution (see [global sensitivity](/ref/dynamics/global-sensitivity)).
- *Game theory.* Shapley values are the unique fair allocation under the axioms; the "players" are features, the "payout" the prediction.
- *Causality.* Interventional SHAP describes the model under feature interventions; causal Shapley values or SCM-based explanations are needed to explain the real system (see [counterfactuals](/ref/explain/counterfactuals)).

## Pitfalls & diagnostics

- *Explaining a bad model:* check performance first; attributions of an overfitted model explain noise.
- *Correlated features:* all methods split, shift or hide importance; group features or use conditional methods.
- *Baseline/background dependence* of SHAP and IG; report the choice and test sensitivity.
- *Instability:* compare explanations across seeds, retrained models and bootstrap samples.
- *Human interpretation:* attributions are not causal effects; avoid "feature X causes Y" language.

## See also

- [Causal graphs & SCMs](/ref/explain/causal-graphs-scms)
- [Counterfactuals](/ref/explain/counterfactuals)
- [Global sensitivity analysis](/ref/dynamics/global-sensitivity)
- [Classical supervised learning](/ref/learn/classical-supervised)

*References:*

- Molnar, _Interpretable Machine Learning_ (free online).
- Lundberg & Lee, "A unified approach to interpreting model predictions", NeurIPS (2017).
- Sundararajan, Taly & Yan, "Axiomatic attribution for deep networks", ICML (2017).
