---
title: 'Causal graphs & structural causal models'
description: 'DAGs, d-separation, structural causal models, the do-operator, backdoor and frontdoor criteria, instrumental variables and the ladder of causation.'
aliases:
  [
    causal inference,
    DAG,
    directed acyclic graph,
    d-separation,
    structural causal model,
    SCM,
    structural equation model,
    do-calculus,
    do-operator,
    intervention,
    confounding,
    backdoor criterion,
    frontdoor criterion,
    collider,
    mediator,
    instrumental variable,
    adjustment set,
    ladder of causation,
    Pearl,
    identifiability of causal effects,
    selection bias,
    Simpson's paradox
  ]
tags: [causality, graphs, statistics]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Correlation is not causation** because of confounders, colliders and selection. A causal graph (DAG) makes your assumptions explicit and tells you what to adjust for.
- **SCM:** each variable is a function of its parents plus independent noise, $X_i = f_i(\text{pa}_i, U_i)$. Interventions $\operatorname{do}(X = x)$ replace an equation.
- **Ladder of causation:** association $p(y\mid x)$ → intervention $p(y\mid\operatorname{do}(x))$ → counterfactuals $p(y_x\mid x', y')$. Each rung needs more assumptions.
- **Backdoor criterion:** adjust for a set $Z$ that blocks all backdoor paths from $X$ to $Y$ and contains no descendants of $X$. Then $p(y\mid\operatorname{do}(x)) = \sum_zp(y\mid x, z)p(z)$.
- **Never adjust for colliders or mediators** (when you want total effects); "control for everything" can create bias.
- When unobserved confounding blocks backdoor adjustment: frontdoor criterion, instrumental variables, or do-calculus.

## Decision table

| Situation                                                              | Identification strategy                                                  | Requirement                                      |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------ |
| Randomised experiment                                                  | compare groups directly                                                  | randomisation, compliance                        |
| Observed confounders block all backdoor paths                          | backdoor adjustment                                                      | correct DAG, all confounders measured            |
| Unobserved confounder, observed mediator fully transmitting the effect | frontdoor adjustment                                                     | mediator unconfounded with outcome given $X$     |
| Unobserved confounding, valid instrument                               | instrumental variables                                                   | relevance, exclusion, independence of instrument |
| Policy / threshold assignment                                          | regression discontinuity                                                 | continuity at the threshold                      |
| Before/after with control group                                        | difference-in-differences                                                | parallel trends                                  |
| General graph with latent variables                                    | do-calculus / ID algorithm                                               | graph known                                      |
| Graph unknown                                                          | causal discovery (see [causal discovery](/ref/explain/causal-discovery)) | assumptions (faithfulness etc.)                  |

| Path structure                  | Example                   | Conditioning on the middle node       |
| ------------------------------- | ------------------------- | ------------------------------------- |
| Chain $X \to M \to Y$           | mediation                 | blocks the path (removes the effect)  |
| Fork $X \leftarrow Z \to Y$     | confounding               | blocks the path (removes confounding) |
| Collider $X \to C \leftarrow Y$ | selection / common effect | **opens** the path (creates bias)     |

## Variants

### DAGs and d-separation

- Nodes = variables, edges = direct causal effects, no cycles.
- A path is **blocked** by a set $Z$ if it contains a chain or fork whose middle node is in $Z$, or a collider that is not in $Z$ and has no descendant in $Z$.
- **d-separation** ⇒ conditional independence in every distribution compatible with the graph (Markov condition). Testable implications let you falsify a DAG with data.

### Structural causal models

- $X_i := f_i(\text{pa}(X_i), U_i)$ with independent exogenous noise $U_i$.
- **Intervention** $\operatorname{do}(X = x)$: replace $X$'s equation by the constant $x$ (graph surgery: remove incoming edges).
- **Truncated factorisation:** $p(v\mid\operatorname{do}(x)) = \prod_{i: V_i \ne X}p(v_i\mid\text{pa}_i)$ evaluated at $X = x$.

### Backdoor and frontdoor

- **Backdoor:** $Z$ blocks every path from $X$ to $Y$ that starts with an arrow into $X$, and contains no descendant of $X$.
- **Frontdoor:** if $M$ intercepts all directed paths $X \to Y$, there's no unblocked backdoor from $X$ to $M$, and $X$ blocks backdoors from $M$ to $Y$: $p(y\mid\operatorname{do}(x)) = \sum_mp(m\mid x)\sum_{x'}p(y\mid m, x')p(x')$.
- Tools (DAGitty, causal-learn) list valid adjustment sets.

### do-calculus

- Three rules for inserting/deleting observations and interventions given d-separation in modified graphs; complete: if an effect is identifiable from the graph, do-calculus (or the ID algorithm) finds the formula.

### Instrumental variables

- $Z$ affects $X$ (relevance), affects $Y$ only through $X$ (exclusion), and shares no confounders with $Y$.
- Linear case: $\beta = \operatorname{Cov}(Z, Y)/\operatorname{Cov}(Z, X)$ (Wald estimator); two-stage least squares with several instruments. Weak instruments give biased, unstable estimates. Mendelian randomisation uses genetic variants as instruments.

## Other lenses

- **Dynamical systems.** Mechanistic ODE models are SCMs in continuous time: interventions = clamping a variable or changing a parameter; equilibrium SCMs arise as steady states of ODEs (with cycles allowed in the dynamics).
- **Probability.** A DAG is a factorisation of the joint distribution (Bayesian network); causal semantics add the interventional meaning.
- **Statistics.** Regression coefficients have causal meaning only when the adjustment set is valid; the DAG decides which covariates belong in the model (see [regression & GLMs](/ref/freq/regression-glms)).

## Pitfalls & diagnostics

- **Collider bias / selection bias:** conditioning on a common effect (e.g. only hospitalised patients) creates spurious associations.
- **Adjusting for mediators** removes part of the effect you want to estimate.
- **Table 2 fallacy:** coefficients of adjustment variables in the same model are not their causal effects.
- **Unmeasured confounding** cannot be detected from data alone; do sensitivity analysis (E-values, bias formulas).
- **Draw the DAG before analysing data,** and check its testable conditional independencies.

## See also

- [Causal effect estimation](/ref/explain/causal-effect-estimation)
- [Counterfactuals](/ref/explain/counterfactuals)
- [Causal discovery](/ref/explain/causal-discovery)
- [Regression & GLMs](/ref/freq/regression-glms)

**References:**

- Pearl, _Causality_ (2nd ed.).
- Pearl, Glymour & Jewell, _Causal Inference in Statistics: A Primer_.
- Hernán & Robins, _Causal Inference: What If_ (free online).
