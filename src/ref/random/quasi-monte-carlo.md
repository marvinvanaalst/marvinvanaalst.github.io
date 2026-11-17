---
title: 'Quasi-Monte Carlo & space-filling designs'
description: 'Sobol, Halton and lattice rules, scrambling and randomised QMC for error estimates, Latin hypercube sampling, discrepancy and convergence rates.'
aliases:
  [
    QMC,
    quasi-Monte Carlo,
    low-discrepancy sequences,
    Sobol sequence,
    Halton sequence,
    lattice rules,
    rank-1 lattice,
    Owen scrambling,
    randomized QMC,
    RQMC,
    digital nets,
    star discrepancy,
    Koksma-Hlawka,
    Latin hypercube sampling,
    LHS,
    maximin design,
    space-filling design,
    effective dimension
  ]
tags: [random numbers, integration, design of experiments]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- QMC replaces random points with **deterministic low-discrepancy points** that fill space evenly. For smooth integrands with low effective dimension, the error drops nearly as $N^{-1}$ instead of Monte Carlo's $N^{-1/2}$.
- **Default: scrambled Sobol** with $N$ a power of 2. Randomised (scrambled) QMC gives unbiased estimates and error bars from a few independent scramblings.
- **Lattice rules** are great for periodic (or periodised) smooth integrands; **Halton** is easy but degrades in higher dimensions unless scrambled.
- **LHS:** good marginal stratification for small designs (surrogate training, sensitivity analysis), but only $O(N^{-1/2})$ in general.
- QMC needs **inverse-CDF transforms** to non-uniform distributions; rejection sampling breaks its structure.

## Decision table

| Point set                 | Error (smooth $f$)                               | Extensible              | Error estimate          | Use when                                        |
| ------------------------- | ------------------------------------------------ | ----------------------- | ----------------------- | ----------------------------------------------- |
| Monte Carlo (i.i.d.)      | $O(N^{-1/2})$                                    | yes                     | sample variance         | rough integrands, very high effective dimension |
| Latin hypercube           | $O(N^{-1/2})$, smaller constant for additive $f$ | no (fixed $N$)          | replicate designs       | small designs, surrogates, screening            |
| Halton (scrambled)        | $O(N^{-1}\log^d N)$                              | yes                     | randomise               | low–moderate dimension, simple                  |
| Sobol (Owen-scrambled)    | $O(N^{-1}\log^d N)$; RQMC up to $O(N^{-3/2})$    | yes (powers of 2)       | independent scramblings | **default** QMC                                 |
| Rank-1 lattice (shifted)  | $O(N^{-\alpha})$ for periodic, smooth $f$        | with embedded sequences | random shifts           | periodic integrands, very smooth problems       |
| Higher-order digital nets | $O(N^{-\alpha})$                                 | yes                     | randomise               | very smooth integrands                          |
| Maximin / optimised LHS   | design criterion, not integration                | no                      | —                       | computer experiments with few runs              |

## Variants

### Discrepancy and Koksma–Hlawka

- **Star discrepancy** $D^*_N$: maximum difference between the fraction of points in an anchored box and its volume.
- **Koksma–Hlawka:** $\big\lvert\frac1N\sum f(x_i) - \int f\big\rvert \le V_{HK}(f)\,D^*_N$. Low-discrepancy sequences achieve $D^*_N = O(\log^d N/N)$.
- The bound is pessimistic; practical performance is governed by effective dimension and smoothness.

### Sobol sequences

- Digital $(t, s)$-sequences in base 2 built from direction numbers (Joe–Kuo are standard).
- Best properties at $N = 2^m$; balance: every elementary interval of the right volume contains the right number of points.
- **Owen scrambling** (nested uniform scrambling) or the cheaper linear matrix scrambling + digital shift randomises while keeping the net structure; Owen-scrambled nets give RQMC variance $O(N^{-3}\log^{d-1}N)$ for smooth $f$.
- Skip the first point (0) or use scrambling; don't take arbitrary $N$ (lose balance).

### Halton

- Coordinate $j$ uses the radical inverse in the $j$-th prime base. Simple and extensible.
- Higher dimensions (large primes) show strong correlations between coordinates; scrambling or leaping fixes this.

### Lattice rules

- $x_i = \{iz/N\}$ (fractional part) for a generating vector $z$, found by component-by-component construction for weighted function spaces.
- Excellent for periodic integrands; periodise others with a tent transform or variable substitution. Random shift for error estimates.

### Latin hypercube sampling

1. Split each dimension into $N$ equal-probability strata.
2. Place one sample per stratum in each dimension, permuting independently across dimensions.
3. Optionally optimise for maximin distance or low correlation between columns.

- Variance reduction for additive functions; no gain for interactions. Not extensible (adding points breaks the design).

### Randomised QMC in practice

1. Generate $R$ (≈ 10–30) independent scramblings of an $N$-point Sobol set.
2. Compute the estimate for each; report the mean and the standard error across replicates.
3. Increase $N$ (power of 2) rather than $R$ for accuracy.

## Other lenses

- **Number theory.** Lattice rules and digital nets come from number theory (Diophantine approximation, finite fields); figure-of-merit searches pick good generators.
- **Approximation.** Effective dimension (ANOVA decomposition) explains when QMC works in hundreds of dimensions, e.g. finance and SDE path integrals with Brownian bridge ordering (see [high-dimensional numerics](/ref/approx/high-dimensional-numerics)).
- **Design of experiments.** Space-filling designs for computer experiments (surrogate training, global sensitivity analysis) use the same point sets (see [surrogates](/ref/approx/surrogates), [global sensitivity](/ref/dynamics/global-sensitivity)).

## Pitfalls & diagnostics

- **Non-smooth integrands** (indicators, kinks) reduce QMC to near-MC rates; smooth them (conditioning, preintegration) if possible.
- **Variable ordering matters:** put the most important variables in the first coordinates (Brownian bridge or PCA construction for paths).
- **Rejection or acceptance steps** destroy the low-discrepancy structure; use inverse CDF transforms.
- **Unscrambled sequences give no error estimate**, and their leading points can be degenerate (e.g. many points on the diagonal in Halton).
- **Mixing QMC and pseudo-random numbers** in the same simulation loses most of the benefit unless done carefully (e.g. QMC for the important dimensions only).

## See also

- [Sampling from distributions](/ref/random/sampling-distributions)
- [Variance reduction](/ref/random/variance-reduction)
- [Quadrature](/ref/integrate/quadrature)
- [High-dimensional numerics](/ref/approx/high-dimensional-numerics)

**References:**

- Dick, Kuo & Sloan, "High-dimensional integration: the quasi-Monte Carlo way", Acta Numerica (2013).
- Owen, _Practical Quasi-Monte Carlo Integration_ (online book).
- Joe & Kuo, "Constructing Sobol sequences with better two-dimensional projections", SIAM J. Sci. Comput. (2008).
