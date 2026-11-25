---
title: 'Hypothesis tests'
description: 't-tests, chi-square, likelihood-ratio / Wald / score tests, nonparametric and permutation tests, power and sample size, and how to read p-values correctly.'
aliases:
  [
    hypothesis testing,
    p-value,
    significance,
    t-test,
    Welch t-test,
    paired t-test,
    ANOVA,
    chi-square test,
    Fisher exact test,
    likelihood ratio test,
    LRT,
    Wald test,
    score test,
    Mann-Whitney,
    Wilcoxon,
    Kruskal-Wallis,
    permutation test,
    Kolmogorov-Smirnov,
    power analysis,
    sample size,
    effect size,
    type I error,
    type II error,
    equivalence test,
    TOST
  ]
tags: [statistics, testing, inference]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- A p-value is $`P(\text{data at least this extreme}\mid H_0)`$, not the probability that $`H_0`$ is true, and not an effect size.
- *Two group means:* Welch's t-test (don't assume equal variances). *Paired data:* paired t-test or Wilcoxon signed-rank. *More groups:* ANOVA / Welch ANOVA, or a regression model.
- *Nested parametric models:* likelihood-ratio test ($`2\Delta\ell \sim\chi^2_{\Delta k}`$). Wald and score tests are cheaper approximations.
- *No distributional assumptions you trust:* *permutation tests* (exact under exchangeability, any test statistic).
- *Report effect sizes with confidence intervals;* plan *power* before collecting data; correct for multiple testing (see [multiple testing](/ref/freq/multiple-testing)).
- Want to show "no difference"? Use an equivalence test (TOST), not a non-significant p-value.

## Decision table

| Question | Test | Assumptions |
| --- | --- | --- |
| Two independent means | Welch t-test | approx. normal means (CLT), independence |
| Two paired means | paired t-test | normal differences |
| Two independent groups, robust | Mann–Whitney U (Wilcoxon rank-sum) | tests stochastic dominance, not means |
| Paired, robust | Wilcoxon signed-rank | symmetric differences |
| $`k`$ group means | one-way ANOVA / Welch ANOVA | normality, (equal variances for classic) |
| $`k`$ groups, robust | Kruskal–Wallis | ranks |
| Contingency table | chi-square; Fisher exact (small counts) | expected counts ≳ 5 for chi-square |
| Proportion vs value | exact binomial | — |
| Nested models (GLMs, ODE models) | likelihood-ratio test | regularity, nested, $`H_0`$ not on a boundary |
| Single coefficient | Wald test (z / t) | approx. normal estimator |
| Distribution equality | Kolmogorov–Smirnov, Anderson–Darling | continuous distributions |
| Any statistic, exchangeable units | permutation test | exchangeability under $`H_0`$ |
| Correlation | Pearson (linear), Spearman/Kendall (monotone) | — |
| Normality | Shapiro–Wilk (or just Q-Q plots) | low power for small $`n`$, oversensitive for large $`n`$ |

## Variants

### The logic

- *$`H_0`$, test statistic $`T`$, null distribution.* Reject if $`p \le \alpha`$.
- *Type I error* $`\alpha`$: rejecting a true $`H_0`$. *Type II error* $`\beta`$: missing a true effect; *power* $`= 1 - \beta`$.
- A CI excluding the null value ⇔ rejection at the corresponding level (for consistent test/interval pairs).

### t-tests

- Welch: $`t = (\bar x_1 - \bar x_2)/\sqrt{s_1^2/n_1 + s_2^2/n_2}`$ with Welch–Satterthwaite degrees of freedom.
- Robust to moderate non-normality for moderate $`n`$ (CLT), sensitive to outliers and strong skew; log-transform skewed positive data.

### Likelihood-ratio, Wald and score tests

- *LRT:* $`\Lambda = 2[\ell(\hat\theta_1) - \ell(\hat\theta_0)] \sim\chi^2_{\Delta k}`$ for nested models. Needs both fits; best small-sample behaviour of the three.
- *Wald:* $`(\hat\theta - \theta_0)^2/\mathrm{SE}^2 \sim\chi^2_1`$; only the full fit; not invariant to reparametrisation.
- *Score (Lagrange multiplier):* uses the gradient at the null fit; only the null fit.
- *Boundary cases* (testing a variance component = 0): the null distribution is a mixture, e.g. $`\tfrac12\chi^2_0 + \tfrac12\chi^2_1`$.

### Permutation tests

1. Compute the test statistic $`T_{\text{obs}}`$.
2. Randomly permute group labels (or signs for paired data) many times, recomputing $`T`$.
3. $`p = (1 + \#\{T_{\text{perm}} \ge T_{\text{obs}}\})/(1 + N_{\text{perm}})`$.

- Exact under exchangeability; works with any statistic (difference of medians, model-based statistics). For regression, permute residuals (Freedman–Lane).

### Power and sample size

- Power depends on effect size, noise, $`n`$ and $`\alpha`$. Two-sample t-test: $`n \approx 2(z_{1-\alpha/2} + z_{1-\beta})^2\sigma^2/\delta^2`$ per group (≈ 16 $`\sigma^2/\delta^2`$ for 80% power at α = 0.05).
- For complex models: simulate data under the assumed effect and count rejections.

### Equivalence tests

- TOST: two one-sided tests against equivalence margins $`\pm\Delta`$; conclude equivalence if both reject (equivalently, if the 90% CI lies within $`\pm\Delta`$).

## Other lenses

- *Model comparison.* Testing nested models is a special case of model selection; AIC/BIC/LOO compare non-nested models without p-values (see [model selection](/ref/freq/model-selection)).
- *Bayesian.* Bayes factors quantify evidence for $`H_0`$ as well as against it; posterior probabilities of effects answer the question people usually think p-values answer (see [Bayesian model comparison](/ref/bayes/model-comparison)).
- *Decision theory.* Neyman–Pearson: tests are decision rules with controlled error rates; the likelihood ratio is the most powerful statistic for simple hypotheses.

## Pitfalls & diagnostics

- *"Not significant" ≠ "no effect";* check power or use equivalence tests.
- *p-hacking:* choosing tests, outlier rules or subsets after seeing data invalidates p-values; preregister or correct.
- *Pseudo-replication:* technical replicates or repeated measurements are not independent samples; use mixed models or aggregate per unit.
- *Many tests:* control FWER or FDR (see [multiple testing](/ref/freq/multiple-testing)).
- *Large $`n`$:* tiny, irrelevant effects become significant; focus on effect sizes and intervals.

## See also

- [Confidence intervals](/ref/freq/confidence-intervals)
- [Multiple testing](/ref/freq/multiple-testing)
- [Regression & GLMs](/ref/freq/regression-glms)
- [Resampling](/ref/freq/resampling)

*References:*

- Wasserman, _All of Statistics_.
- Good, _Permutation, Parametric, and Bootstrap Tests of Hypotheses_.
- Wasserstein & Lazar, "The ASA statement on p-values", Am. Stat. (2016).
