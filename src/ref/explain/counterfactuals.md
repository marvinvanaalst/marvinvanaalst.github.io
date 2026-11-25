---
title: 'Counterfactuals'
description: 'Counterfactual reasoning in SCMs (abduction, action, prediction), probabilities of necessity and sufficiency, and counterfactual explanations for ML models (recourse).'
aliases:
  [
    counterfactual,
    counterfactual inference,
    abduction action prediction,
    twin network,
    probability of necessity,
    probability of sufficiency,
    PNS,
    attribution,
    effect of treatment on the treated,
    counterfactual explanation,
    algorithmic recourse,
    DiCE,
    actionable recourse,
    contrastive explanation,
    what-if analysis,
    individual treatment effect
  ]
tags: [causality, explainability, SCM]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Counterfactual question:* "given what actually happened, what would have happened if $`X`$ had been different?" — rung 3 of the causal ladder; needs a full *SCM*, not just a DAG.
- *Three steps:* (1) *abduction*: infer the exogenous noise $`U`$ from the observed facts; (2) *action*: modify the equations with $`\operatorname{do}(X = x')`$; (3) *prediction*: compute outcomes with the same $`U`$.
- Many counterfactual quantities are not identifiable from data, even with experiments; bounds (e.g. on probability of necessity) are often the honest answer.
- *ML counterfactual explanations* ("what minimal change of inputs flips the prediction?") are a different, model-level concept: useful for recourse, but they ignore causal relations between features unless you add them.

## Decision table

| Concept | Definition | Needs | Use for |
| --- | --- | --- | --- |
| Interventional effect | $`p(y\mid\operatorname{do}(x))`$ | DAG + data | population-level decisions |
| Counterfactual $`Y_{x'}`$ given evidence | $`p(y_{x'}\mid x, y)`$ | SCM (functional forms + noise) | individual "what if" questions |
| Effect of treatment on the treated | $`\mathbb E[Y_1 - Y_0\mid T = 1]`$ | weaker (often identifiable) | evaluating programs |
| Probability of necessity (PN) | $`P(Y_{x'} = 0\mid X = x, Y = 1)`$ | SCM or bounds from data | attribution, legal "but-for" causation |
| Probability of sufficiency (PS) | $`P(Y_x = 1\mid X = x', Y = 0)`$ | SCM or bounds | "would $`X`$ have caused $`Y`$" |
| PNS | $`P(Y_x = 1, Y_{x'} = 0)`$ | bounds from experimental + observational data | combined necessity and sufficiency |
| Counterfactual explanation (ML) | nearest input $`x'`$ with $`f(x') \ne f(x)`$ | model + distance + constraints | recourse, contestability |
| Causal (actionable) recourse | minimal intervention in an SCM of the features | SCM over features | realistic recourse respecting causal effects |

## Variants

### Abduction–action–prediction

1. *Abduction:* given evidence $`e`$ (observed values), compute the posterior over exogenous variables $`p(u\mid e)`$ (deterministic for invertible additive-noise models: $`u_i = x_i - f_i(\text{pa}_i)`$).
2. *Action:* replace the equation of $`X`$ by $`X := x'`$.
3. *Prediction:* propagate $`p(u\mid e)`$ through the modified model to get the counterfactual distribution of $`Y`$.

- *Twin network* representation: a factual and a counterfactual copy of the graph sharing the exogenous nodes; d-separation in the twin network answers counterfactual independence questions.

### Example: linear SCM

- $`Y = \beta X + U_Y`$. Observed $`(x, y)`$ ⇒ $`u_Y = y - \beta x`$ ⇒ counterfactual $`Y_{x'} = \beta x' + u_Y = y + \beta(x' - x)`$.
- In nonlinear models, the counterfactual depends on the functional form and noise model, not only on the interventional distribution.

### Probabilities of causation

- With experimental and observational data, PN and PS are bounded (Tian–Pearl bounds); under monotonicity (no unit harmed by treatment) they become identifiable, e.g. $`\mathrm{PN} = (P(y) - P(y_{x'}))/P(x, y)`$.

### Counterfactual explanations for ML models

- Optimise $`\min_{x'}\ d(x, x') + \lambda\,\ell(f(x'), y_{\text{target}})`$ subject to plausibility (stay on the data manifold) and actionability constraints (immutable features, monotone changes).
- Generate diverse sets (DiCE) so users see different options.
- Without a causal model of features, recommended changes may be infeasible or have side effects (changing education changes income too); causal recourse uses an SCM over features.

## Other lenses

- *Mechanistic modelling.* Simulating an ODE model with the same initial state and noise but an altered parameter or input is a counterfactual computation; mechanistic models are SCMs with explicit functional forms.
- *Bayesian inference.* Abduction is posterior inference over latent noise; counterfactuals are posterior predictive simulations in a modified model (see [probabilistic programming](/ref/programs/probabilistic-programming)).
- *Explainability.* Counterfactual explanations complement attribution methods: "what would change the outcome" rather than "what contributed to it" (see [feature attribution](/ref/explain/feature-attribution)).

## Pitfalls & diagnostics

- *Non-identifiability:* different SCMs with the same observational and interventional distributions give different counterfactuals; state the modelling assumptions.
- *Treating ML counterfactual explanations as causal claims* about the real world.
- *Off-manifold counterfactuals:* unrealistic feature combinations that the model never saw; enforce plausibility.
- *Individual-level claims* need much stronger assumptions than population-level ones; prefer bounds or population estimands when possible.

## See also

- [Causal graphs & SCMs](/ref/explain/causal-graphs-scms)
- [Causal effect estimation](/ref/explain/causal-effect-estimation)
- [Feature attribution](/ref/explain/feature-attribution)
- [Probabilistic programming](/ref/programs/probabilistic-programming)

*References:*

- Pearl, _Causality_ (2nd ed.), ch. 7, 9.
- Wachter, Mittelstadt & Russell, "Counterfactual explanations without opening the black box", Harv. J. Law Technol. (2018).
- Karimi, Schölkopf & Valera, "Algorithmic recourse: from counterfactual explanations to interventions", FAccT (2021).
