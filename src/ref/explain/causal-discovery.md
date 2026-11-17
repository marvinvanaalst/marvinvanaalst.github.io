---
title: 'Causal discovery'
description: 'Learning causal graphs from data: constraint-based (PC, FCI), score-based (GES), functional (LiNGAM, ANM) and continuous-optimisation (NOTEARS) methods, with their assumptions.'
aliases:
  [
    causal discovery,
    structure learning,
    causal structure learning,
    PC algorithm,
    FCI,
    GES,
    greedy equivalence search,
    LiNGAM,
    additive noise models,
    ANM,
    NOTEARS,
    DAGMA,
    Markov equivalence class,
    CPDAG,
    PAG,
    faithfulness,
    causal sufficiency,
    conditional independence tests,
    Granger causality,
    interventional data,
    network inference
  ]
tags: [causality, graphs, structure learning]
order: 5
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- From observational data alone, conditional independences identify the graph only up to a **Markov equivalence class** (some edge directions remain unknown).
- **Constraint-based** (PC; FCI with latent confounders): conditional independence tests remove edges and orient v-structures.
- **Score-based** (GES): search over equivalence classes maximising a score (BIC).
- **Functional causal models** (LiNGAM for linear non-Gaussian, additive noise models for nonlinear): identify full edge directions from asymmetries in noise distributions.
- **Continuous optimisation** (NOTEARS and successors): smooth acyclicity constraints; easy to use, but sensitive to data scaling.
- **Interventional / time-series data** identify much more; use them whenever available. Treat discovered graphs as hypotheses.

## Decision table

| Method                                              | Assumptions                                      | Output                   | Latent confounders | Use when                                               |
| --------------------------------------------------- | ------------------------------------------------ | ------------------------ | ------------------ | ------------------------------------------------------ |
| PC                                                  | causal sufficiency, faithfulness, valid CI tests | CPDAG                    | no                 | moderate number of variables, sparse graphs            |
| FCI / RFCI                                          | faithfulness, CI tests                           | PAG (with latent info)   | yes                | hidden confounding plausible                           |
| GES                                                 | sufficiency, faithfulness, score consistency     | CPDAG                    | no                 | score-based search, BIC                                |
| LiNGAM                                              | linear, non-Gaussian noise, no latents           | full DAG                 | variants exist     | linear systems with non-Gaussian noise                 |
| Additive noise models                               | nonlinear $f$, independent additive noise        | direction for pairs/DAGs | no                 | bivariate direction questions                          |
| NOTEARS / DAGMA                                     | continuous score with acyclicity constraint      | weighted DAG             | no                 | quick exploration (standardise data, check robustness) |
| Granger / PCMCI                                     | time order, no instantaneous hidden confounding  | lagged causal graph      | partly             | time series                                            |
| Interventional methods (GIES, invariant prediction) | data from interventions/environments             | DAG / invariant parents  | partly             | perturbation experiments (knockouts, drugs)            |

## Variants

### PC algorithm

1. Start with a complete undirected graph.
2. For increasing conditioning set sizes, remove edge $X - Y$ if $X \perp Y\mid S$ for some subset $S$ of neighbours; record separating sets.
3. Orient v-structures $X \to Z \leftarrow Y$ when $Z \notin S_{XY}$.
4. Apply orientation rules (Meek) to propagate directions without creating new v-structures or cycles.

- CI tests: partial correlation (Gaussian), kernel tests (KCI), conditional mutual information. Errors in early tests propagate; order-independent variants (stable PC) help.

### Score-based search

- Score each DAG/equivalence class (BIC, BDeu); GES greedily adds then removes edges in the space of equivalence classes; consistent under its assumptions.
- Exact search (dynamic programming, integer programming) feasible for ≲ 30 variables.

### Functional causal models

- **LiNGAM:** $x = Bx + e$ with non-Gaussian independent $e$; ICA recovers $B$ and the causal order.
- **Additive noise models:** $Y = f(X) + N$ with $N \perp X$; the reverse direction generally doesn't admit an independent additive noise representation. Test with HSIC on residuals.

### Continuous optimisation

- NOTEARS: minimise a least-squares loss over weighted adjacency $W$ subject to $h(W) = \operatorname{tr}(e^{W\odot W}) - d = 0$ (zero iff acyclic); DAGMA uses log-determinant constraints.
- Known issue: results can exploit varying marginal variances ("varsortability"); always standardise data and compare against simple baselines.

### Using interventions and time

- Perturbation experiments break symmetries of equivalence classes; invariant causal prediction finds parents whose relation to the target is stable across environments.
- Time series: Granger causality (predictive), PCMCI (conditional independence with lags), transfer entropy.

## Other lenses

- **Statistics.** Conditional independence testing is the core primitive; its power and multiple-testing behaviour bound what PC can do (see [hypothesis tests](/ref/freq/hypothesis-tests)).
- **Information theory.** Conditional mutual information is a nonparametric CI measure; transfer entropy generalises Granger causality (see [entropy & MI](/ref/info/entropy-kl-mi)).
- **Systems biology.** Network inference from expression data (gene regulatory networks) is causal discovery with few samples and strong confounding; perturbation data (knockouts) are essential.

## Pitfalls & diagnostics

- **Assumption violations** (hidden confounders, selection bias, feedback loops, measurement error) produce confident wrong graphs.
- **Sample size:** CI tests with many conditioning variables need lots of data; results on small datasets are unstable.
- **Stability:** bootstrap the discovery procedure and report edge frequencies.
- **Benchmarks on simulated data** often favour methods whose assumptions match the simulator; test on data with known interventions.
- **Use background knowledge** (time order, impossible edges) as constraints; it helps more than algorithm choice.

## See also

- [Causal graphs & SCMs](/ref/explain/causal-graphs-scms)
- [Entropy, KL & mutual information](/ref/info/entropy-kl-mi)
- [Time-series models](/ref/signals/time-series-models)
- [Graph algorithms](/ref/graphs/graph-algorithms)

**References:**

- Spirtes, Glymour & Scheines, _Causation, Prediction, and Search_ (2nd ed.).
- Peters, Janzing & Schölkopf, _Elements of Causal Inference_ (free online).
- Glymour, Zhang & Spirtes, "Review of causal discovery methods based on graphical models", Front. Genet. (2019).
