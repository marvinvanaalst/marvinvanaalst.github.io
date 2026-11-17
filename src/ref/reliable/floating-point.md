---
title: 'Floating-point arithmetic'
description: 'IEEE 754 formats, machine epsilon and ulps, rounding, cancellation, FMA, subnormals and NaN semantics: what every numerical result is built on.'
aliases:
  [
    IEEE 754,
    machine epsilon,
    ulp,
    rounding,
    cancellation,
    FMA,
    subnormal,
    denormal,
    NaN,
    Float32,
    Float64,
    bfloat16,
    half precision
  ]
tags: [floating point, rounding error, precision]
order: 1
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- A float is $\pm m \times 2^e$ with a fixed number of mantissa bits. Every basic operation is **correctly rounded**: $\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta)$ with $\lvert\delta\rvert \le u$.
- Default to Float64 ($u \approx 1.1 \times 10^{-16}$, ~16 digits). Use Float32 or lower only when bandwidth or memory dominates and you know the conditioning.
- The big error source is not a single rounding but **cancellation**: subtracting nearly equal numbers exposes earlier errors.
- Floating-point addition is **not associative**, so summation order, vectorisation and parallel reductions change results.

## Decision table

| Format                   | Bits (sign/exp/mantissa) | Unit roundoff $u$                | Range (max)         | Use when                                                          |
| ------------------------ | ------------------------ | -------------------------------- | ------------------- | ----------------------------------------------------------------- |
| Float64                  | 1 / 11 / 52              | $2^{-53} \approx 10^{-16}$       | $\approx 10^{308}$  | default for scientific computing                                  |
| Float32                  | 1 / 8 / 23               | $2^{-24} \approx 6\times10^{-8}$ | $\approx 10^{38}$   | GPU, ML, memory-bound kernels, well-conditioned problems          |
| bfloat16                 | 1 / 8 / 7                | $2^{-8} \approx 4\times10^{-3}$  | as Float32          | ML training: Float32 range, cheap; accumulate in Float32          |
| Float16                  | 1 / 5 / 10               | $2^{-11} \approx 5\times10^{-4}$ | 65504               | inference, storage; overflows easily                              |
| FP8 (E4M3/E5M2)          | 1 / 4 / 3, 1 / 5 / 2     | $2^{-4}$, $2^{-3}$               | 448, 57344          | ML with per-tensor scaling only                                   |
| Float128 / double-double | 1 / 15 / 112             | $\approx 10^{-34}$               | $\approx 10^{4932}$ | ill-conditioned subproblems, reference solutions (slow, software) |

## Variants

### The model

- **Representation:** $x = (-1)^s \cdot 1.f \cdot 2^{e - \text{bias}}$ (normal numbers, implicit leading 1).
- **Machine epsilon** $\varepsilon = 2^{1-p}$ is the gap between 1 and the next float ($p$ = precision in bits incl. hidden bit). Unit roundoff $u = \varepsilon/2$ under round-to-nearest.
- **ulp** (unit in the last place) of $x$: spacing of floats around $x$, $\approx \varepsilon\lvert x\rvert$. Library functions quote accuracy in ulps (correctly rounded = 0.5 ulp).
- **Standard model:** $\mathrm{fl}(a \circ b) = (a \circ b)(1+\delta)$, $\lvert\delta\rvert \le u$, for $+,-,\times,/,\sqrt{\ }$. Error analysis chains these $\delta$s.

### Rounding modes

- Round to nearest, ties to even (default). Unbiased on average.
- Toward $0$, $+\infty$, $-\infty$: used for interval arithmetic (round lower bounds down, upper bounds up).
- **Stochastic rounding** (not IEEE, but in hardware for ML): round up with probability proportional to the distance. Unbiased in expectation, so long low-precision accumulations don't stagnate.

### Cancellation

- Subtracting $a - b$ with $a \approx b$ is exact (Sterbenz lemma, if $b/2 \le a \le 2b$), but it **reveals** the relative errors already in $a$ and $b$, amplified by $\lvert a\rvert / \lvert a - b\rvert$.
- **Fixes by reformulation:**
  - $\sqrt{x+1} - \sqrt{x} = 1/(\sqrt{x+1} + \sqrt{x})$.
  - Quadratic roots: compute $q = -\tfrac12\big(b + \operatorname{sign}(b)\sqrt{b^2 - 4ac}\big)$, then $x_1 = q/a$, $x_2 = c/q$.
  - Use `expm1`, `log1p` for $e^x - 1$ and $\log(1+x)$ near 0.
  - Variance: Welford's update instead of $\overline{x^2} - \bar x^2$.
  - Log-sum-exp: $\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$ with $m = \max_i x_i$.

### FMA

- Fused multiply-add computes $a \times b + c$ with **one** rounding.
- Faster and more accurate for dot products, Horner, Newton updates.
- Gives exact products: $\mathrm{err} = \mathrm{fma}(a, b, -ab)$ is the exact rounding error of $ab$ (the basis of error-free transformations, compensated algorithms).
- Caveat: the compiler deciding where to contract changes bits between builds.

### Special values

- **Subnormals** fill the gap between 0 and the smallest normal number (gradual underflow). They can be 10–100× slower on some CPUs; flush-to-zero (FTZ/DAZ) trades accuracy for speed.
- **Signed zero:** $+0 = -0$ compares equal, but $1/(-0) = -\infty$. Matters for branch cuts.
- **Inf:** overflow, $x/0$ for $x \ne 0$.
- **NaN:** $0/0$, $\infty - \infty$, $\sqrt{-1}$. Propagates through arithmetic; every comparison with NaN is false, including `NaN == NaN`. Use `isnan`.

## Other lenses

- **Backward error.** The standard model says each operation computes the exact result for slightly perturbed inputs. Stable algorithms keep this backward error at $O(u)$ overall; accuracy then depends on the condition number (see [conditioning & stability](/ref/reliable/conditioning-stability)).
- **Algebra.** Floats form no field and no group: addition is commutative but not associative, there is no distributivity, and NaN breaks reflexivity of equality. Compiler "fast-math" flags pretend otherwise.
- **Probability.** Rounding errors often behave like independent random variables, so error growth is $\sim\sqrt{n}\,u$ in practice rather than the worst case $n u$.

## Pitfalls & diagnostics

- **Equality tests:** compare with a tolerance mixing absolute and relative scales, $\lvert a - b\rvert \le \mathrm{atol} + \mathrm{rtol}\cdot\max(\lvert a\rvert, \lvert b\rvert)$.
- **Decimal literals:** 0.1 is not representable. Money or exact decimals need decimal or integer arithmetic.
- **Integer conversion:** Float64 represents integers exactly only up to $2^{53}$. Float32 only up to $2^{24}$ (≈16.7 M), which bites counters and indices on GPUs.
- **Fast-math** (`-ffast-math`) allows reassociation and assumes no NaN/Inf: it can silently remove NaN checks and break compensated summation.
- **Overflow in intermediates:** $\sqrt{x^2 + y^2}$ overflows for large $x$; use `hypot`. Softmax: subtract the max first.
- **Diagnose precision problems** by rerunning in higher precision (or with randomised rounding) and comparing. Large differences point at cancellation or ill-conditioning.

## See also

- [Summation & reductions](/ref/reliable/summation-reductions)
- [Conditioning & stability](/ref/reliable/conditioning-stability)
- [Interval & affine arithmetic](/ref/reliable/interval-affine-arithmetic)
- [Training in practice](/ref/learn/training-practice) (mixed precision)

**References:**

- Goldberg, "What every computer scientist should know about floating-point arithmetic" (1991).
- Higham, _Accuracy and Stability of Numerical Algorithms_ (2nd ed.), ch. 1–2.
- Muller et al., _Handbook of Floating-Point Arithmetic_.
