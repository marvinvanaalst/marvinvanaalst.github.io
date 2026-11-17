---
title: 'Multiple testing'
description: 'Family-wise error rate (Bonferroni, Holm, Šidák) vs false discovery rate (Benjamini–Hochberg, BY, q-values), dependence, and practical choices for omics-scale testing.'
aliases:
  [
    multiple comparisons,
    multiple testing correction,
    family-wise error rate,
    FWER,
    Bonferroni,
    Holm,
    Sidak,
    Hochberg,
    false discovery rate,
    FDR,
    Benjamini-Hochberg,
    Benjamini-Yekutieli,
    q-value,
    Storey,
    local FDR,
    Tukey HSD,
    Dunnett,
    post-hoc tests,
    p-value adjustment,
    omics
  ]
tags: [statistics, testing, omics]
order: 5
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Testing $m$ hypotheses at level $\alpha$ each produces about $\alpha m$ false positives when all nulls are true.
- **FWER** (probability of any false positive): **Holm** (uniformly better than Bonferroni, no assumptions). Use for confirmatory claims and small $m$.
- **FDR** (expected fraction of false positives among discoveries): **Benjamini–Hochberg**. Use for screening and large $m$ (genes, metabolites, features).
- **Arbitrary dependence:** Benjamini–Yekutieli (conservative); BH is valid under positive dependence, which covers most practical cases.
- **Post-hoc after ANOVA:** Tukey HSD (all pairs), Dunnett (vs control).

## Decision table

| Procedure                       | Controls                            | Assumptions                        | Power                | Use when                                  |
| ------------------------------- | ----------------------------------- | ---------------------------------- | -------------------- | ----------------------------------------- |
| Bonferroni ($p_i \le \alpha/m$) | FWER                                | none                               | lowest               | simple statements, very few tests         |
| Šidák                           | FWER                                | independence                       | ≈ Bonferroni         | rarely worth it                           |
| Holm (step-down)                | FWER                                | none                               | ≥ Bonferroni         | **default for FWER**                      |
| Hochberg (step-up)              | FWER                                | independence / positive dependence | ≥ Holm               | independent tests                         |
| Tukey HSD / Dunnett             | FWER (pairwise / vs control)        | ANOVA model                        | tailored             | post-hoc group comparisons                |
| Benjamini–Hochberg (BH)         | FDR                                 | independence / PRDS                | high                 | **default for FDR** (omics)               |
| Benjamini–Yekutieli (BY)        | FDR                                 | any dependence                     | lower                | strong arbitrary dependence               |
| Storey q-values                 | FDR (estimates $\pi_0$)             | as BH                              | ≥ BH                 | many true effects expected                |
| Local FDR (empirical Bayes)     | posterior null probability per test | mixture model                      | —                    | ranking, very large $m$                   |
| Permutation-based max-T / minP  | FWER                                | exchangeability                    | good with dependence | correlated tests (neuroimaging, genomics) |

## Variants

### Holm

1. Sort p-values $p_{(1)} \le \dots \le p_{(m)}$.
2. Find the first $k$ with $p_{(k)} \gt \alpha/(m - k + 1)$.
3. Reject hypotheses $(1), \dots, (k - 1)$.

### Benjamini–Hochberg

1. Sort p-values $p_{(1)} \le \dots \le p_{(m)}$.
2. Find the largest $k$ with $p_{(k)} \le \frac km\alpha$.
3. Reject hypotheses $(1), \dots, (k)$.

- Adjusted p-values: $\tilde p_{(i)} = \min_{j \ge i}\min\big(1, \frac mjp_{(j)}\big)$; report these and threshold at the target FDR (e.g. 0.05 or 0.1).
- Controls FDR at $\pi_0\alpha \le \alpha$, where $\pi_0$ is the fraction of true nulls; Storey's method estimates $\pi_0$ from the p-value histogram to regain power.

### Dependence

- BH remains valid under positive regression dependence (PRDS), e.g. one-sided tests on positively correlated statistics.
- Strong or unknown dependence: BY ($\alpha$ divided by $\sum_{i=1}^m 1/i \approx \ln m$), or resampling-based procedures that use the actual correlation structure.

### Practical omics workflow

1. Fit per-feature models (often with variance shrinkage, e.g. limma/DESeq2-style empirical Bayes).
2. Check the p-value histogram: uniform background + peak near 0 is healthy; deviations signal model problems.
3. Apply BH; report adjusted p-values and effect sizes (fold changes with intervals).
4. Independent filtering (remove low-information features before testing, using a criterion independent of the test statistic) increases power.

## Other lenses

- **Bayesian.** Hierarchical models with shrinkage handle multiplicity naturally (partial pooling); local FDR is an empirical-Bayes posterior probability of being null.
- **Decision theory.** FWER suits "any error is costly" (confirmatory claims); FDR suits "a list of candidates for follow-up", where some false leads are acceptable.
- **Selective inference.** Testing effects chosen because they look large (winner's curse) inflates estimates; post-selection inference corrects intervals for the selection step.

## Pitfalls & diagnostics

- **Defining the family:** correct over all tests performed, including those not reported.
- **Correcting non-independent repeated analyses** of the same data still leaves garden-of-forking-paths problems.
- **FDR ≠ per-hypothesis error:** a q-value of 0.04 means 4% of discoveries at that threshold are expected false, not that this one has 4% chance.
- **Discrete tests** (Fisher exact with small counts) have conservative p-values; specialised procedures help.
- **Ignoring effect size:** a long list of significant tiny effects may be scientifically irrelevant.

## See also

- [Hypothesis tests](/ref/freq/hypothesis-tests)
- [Regression & GLMs](/ref/freq/regression-glms)
- [Resampling](/ref/freq/resampling)

**References:**

- Benjamini & Hochberg, "Controlling the false discovery rate", JRSS B (1995).
- Holm, "A simple sequentially rejective multiple test procedure", Scand. J. Stat. (1979).
- Storey & Tibshirani, "Statistical significance for genomewide studies", PNAS (2003).
