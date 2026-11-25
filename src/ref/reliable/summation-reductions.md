---
title: 'Summation & reductions'
description: 'Naive, pairwise, Kahan and Neumaier summation, compensated dot products and reproducible parallel reductions: error bounds and costs.'
aliases:
  [
    Kahan summation,
    Neumaier,
    compensated summation,
    pairwise summation,
    TwoSum,
    error-free transformation,
    dot product,
    reproducible reduction,
    ReproBLAS
  ]
tags: [floating point, summation, reproducibility]
order: 2
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- Error of summing $`n`$ numbers: naive $`\sim n u \sum\lvert x_i\rvert`$ (worst case), pairwise $`\sim u \log_2 n \sum\lvert x_i\rvert`$, compensated (Kahan/Neumaier) $`\sim 2u\sum\lvert x_i\rvert`$ independent of $`n`$.
- *Pairwise* is the free lunch: same cost as naive and what NumPy/Julia `sum` already do (blocked).
- *Neumaier* when sums cancel heavily or $`n`$ is huge and accuracy matters; \~4× the flops, but usually memory-bound anyway.
- Parallel or SIMD reductions change the summation order, so bits change with thread count. If you need identical bits, fix the reduction tree or use a reproducible algorithm.

## Decision table

| Method | Error bound (relative to $`\sum\lvert x_i\rvert`$) | Flops per element | Use when | Pitfalls |
| --- | --- | --- | --- | --- |
| Naive loop | $`(n-1)u`$ | 1 | short sums, integers | error grows with $`n`$ |
| Pairwise / blocked | $`u \lceil\log_2 n\rceil`$ | 1 | default | needs recursion or blocking |
| Kahan | $`2u + O(nu^2)`$ | 4 | long sums of same-sign terms | fails when a term exceeds the running sum |
| Neumaier (Kahan–Babuška) | $`2u + O(nu^2)`$ | 4–5 (branch) | general compensated summation | fast-math reassociation kills it |
| Sorted (increasing magnitude) | improves naive | $`n\log n`$ sort | offline, all same sign | useless for mixed signs |
| Double-double / long accumulator | \~exact, correctly rounded | 10–20 / more | reference results, exact dot products | slow |
| Reproducible (binned, ReproBLAS) | \~compensated, order-independent | 3–8 | bitwise identical results across thread counts | throughput cost |

## Variants

### Pairwise summation

- Split the array in halves recursively, sum each half, add the results. Each element passes through \~$`\log_2 n`$ additions.
- In practice: sum blocks of \~128–1024 elements naively (SIMD-friendly), combine block sums pairwise.

### Error-free transformation: TwoSum

The rounding error of a single addition is itself a float and can be computed exactly:

1. $`s = \mathrm{fl}(a + b)`$
2. $`b' = s - a`$, $`a' = s - b'`$
3. $`e = (a - a') + (b - b')`$

Then $`a + b = s + e`$ exactly. Fast2Sum (3 flops) does the same if $`\lvert a\rvert \ge \lvert b\rvert`$. The product analogue uses FMA: $`e = \mathrm{fma}(a, b, -ab)`$.

### Kahan and Neumaier

Carry a running compensation $`c`$ of lost low-order bits:

1. Start with $`s = 0`$, $`c = 0`$.
2. For each $`x_i`$: $`t = s + x_i`$.
3. *Neumaier:* if $`\lvert s\rvert \ge \lvert x_i\rvert`$ then $`c \mathrel{+}= (s - t) + x_i`$, else $`c \mathrel{+}= (x_i - t) + s`$. Set $`s = t`$.
4. Return $`s + c`$.

- Kahan's original subtracts the compensation before adding and assumes $`\lvert s\rvert \ge \lvert x_i\rvert`$. Neumaier handles terms larger than the running sum (e.g. $`[1, 10^{100}, 1, -10^{100}]`$).
- Compensated dot product (Ogita–Rump–Oishi "Dot2"): TwoProduct for each $`a_i b_i`$ plus TwoSum accumulation. As accurate as computing in twice the precision.

### Reproducible reductions

- *Source of non-reproducibility:* different thread counts, SIMD widths or GPU schedules produce different summation trees.
- *Options:*
  - Fix the tree (fixed block size, fixed combination order) for a given target. Cheap, reproducible per architecture.
  - Pre-rounding / binned summation (ReproBLAS): split each number into fixed exponent bins whose sums are exact, so the result is order-independent.
  - Exact long accumulators (Kulisch): huge fixed-point register covering the full exponent range.

## Other lenses

- *Algebra.* Summation is a monoid fold; parallel reduction requires associativity, which floats lack. Compensated or binned arithmetic restores associativity of the _result_. See [algebraic structures](/ref/programs/algebraic-structures).
- *Statistics.* Rounding errors in long sums behave roughly like a random walk: observed error $`\sim\sqrt{n}\,u`$ for naive, $`\sim\sqrt{\log n}\,u`$ for pairwise.
- *Conditioning.* The condition number of a sum is $`\sum\lvert x_i\rvert / \lvert\sum x_i\rvert`$. No algorithm fixes a badly conditioned sum in working precision except doing it more precisely (compensation is effectively double precision).

## Pitfalls & diagnostics

- *Fast-math flags* let the compiler simplify $`(s - t) + x`$ to 0, deleting the compensation. Compile compensated code without reassociation.
- *Variance and means:* don't sum $`x^2`$ and subtract. Use Welford's online update or two passes.
- *Accumulating in low precision:* Float16/bfloat16 sums stagnate once the sum's ulp exceeds the increments. Accumulate in Float32 (what tensor cores do).
- *Testing:* compare against a high-precision reference (BigFloat, double-double, or sorting by magnitude with Neumaier).

## See also

- [Floating-point arithmetic](/ref/reliable/floating-point)
- [Conditioning & stability](/ref/reliable/conditioning-stability)
- [Algebraic structures](/ref/programs/algebraic-structures)

*References:*

- Higham, _Accuracy and Stability of Numerical Algorithms_, ch. 4.
- Ogita, Rump & Oishi, "Accurate sum and dot product", SIAM J. Sci. Comput. (2005).
- Demmel & Nguyen, "Parallel reproducible summation", IEEE Trans. Comput. (2015).
