---
title: 'Finite differences'
description: 'Forward, central and higher-order difference formulas, the optimal step size, complex-step and hyper-dual differentiation, Richardson extrapolation and randomised (SPSA) gradients.'
aliases:
  [
    finite difference,
    forward difference,
    central difference,
    step size,
    truncation error,
    complex step,
    Richardson extrapolation,
    numerical derivative,
    stencil,
    Fornberg weights,
    gradient check,
    hyper-dual numbers,
    multicomplex step,
    SPSA,
    simultaneous perturbation,
    Gaussian smoothing,
    zeroth-order gradient
  ]
tags: [differentiation, numerical derivative]
order: 1
updated: '2026-11-20'
published: true
template: ref
---

## TL;DR

- *Forward difference:* $`n`$ extra evaluations for a gradient, error $`O(h)`$, best $`h \approx \sqrt{\varepsilon}\,\lvert x\rvert`$ giving \~8 digits in Float64.
- *Central difference:* $`2n`$ evaluations, error $`O(h^2)`$, best $`h \approx \varepsilon^{1/3}\lvert x\rvert`$ giving \~10–11 digits.
- *Complex step:* if $`f`$ is real-analytic and implemented with complex-safe operations, $`f'(x) = \operatorname{Im} f(x + ih)/h`$ with $`h = 10^{-20}`$ gives derivatives *to machine precision*, 1 complex evaluation per direction.
- *Hyper-dual / multicomplex step* extend this to exact second (and higher) derivatives.
- *SPSA* estimates a full gradient from 2 evaluations regardless of dimension, at the cost of variance; for noisy or very expensive black boxes.
- For anything beyond gradient checks or black-box code, prefer [automatic differentiation](/ref/diff/automatic-differentiation): exact, and for gradients far cheaper.

## Decision table

| Formula | Error | Evaluations per derivative | Optimal $`h`$ (relative) | Achievable accuracy (Float64) | Use when |
| --- | --- | --- | --- | --- | --- |
| Forward $`[f(x+h) - f(x)]/h`$ | $`O(h)`$ | 1 (+ $`f(x)`$ shared) | $`\sqrt{\varepsilon} \approx 10^{-8}`$ | \~$`10^{-8}`$ | cheap Jacobians, black-box code |
| Central $`[f(x+h) - f(x-h)]/2h`$ | $`O(h^2)`$ | 2 | $`\varepsilon^{1/3} \approx 6\times10^{-6}`$ | \~$`10^{-11}`$ | more accuracy, gradient checks |
| 5-point central | $`O(h^4)`$ | 4 | $`\varepsilon^{1/5} \approx 10^{-3}`$ | \~$`10^{-13}`$ | smooth $`f`$, high accuracy |
| Complex step | $`O(h^2)`$, no cancellation | 1 complex | anything tiny ($`10^{-20}`$) | machine precision | analytic $`f`$, complex-safe code |
| Hyper-dual step | exact (no truncation) | 1 hyper-dual (\~4× real) | any (e.g. 1) | machine precision, 2nd order | exact Hessian entries, analytic $`f`$ |
| SPSA (2-point) | $`O(h^2)`$ + variance | 2 for the whole gradient | decaying with iterations | noisy estimate | stochastic optimisation, huge $`n`$ |
| Richardson extrapolation | removes leading terms | several $`h`$ | adaptive | high | derivative estimates with error control |
| Second derivative $`[f(x+h) - 2f(x) + f(x-h)]/h^2`$ | $`O(h^2)`$ | 3 | $`\varepsilon^{1/4} \approx 10^{-4}`$ | \~$`10^{-8}`$ | Hessian diagonals, stencils |

## Variants

### Step size trade-off

- Total error ≈ truncation + rounding: forward difference $`\tfrac12\lvert f''\rvert h + 2\varepsilon\lvert f\rvert/h`$.
- Minimising gives $`h^* \approx 2\sqrt{\varepsilon\lvert f/f''\rvert}`$. In practice use $`h = \sqrt{\varepsilon}\max(\lvert x\rvert, x_{\text{typ}})`$.
- Make $`h`$ exactly representable: $`h \leftarrow (x + h) - x`$ removes the error in the step itself.
- If $`f`$ is noisy (e.g. an ODE solved with tolerance $`10^{-6}`$), $`\varepsilon`$ is the noise level, not machine epsilon: $`h \approx \sqrt{\text{noise}}`$, and accuracy is limited accordingly.

### Gradients and Jacobians

- Gradient of $`f : \mathbb R^n \to \mathbb R`$: $`n`$ (forward) or $`2n`$ (central) evaluations.
- Jacobian of $`F : \mathbb R^n \to \mathbb R^m`$: $`n`$ evaluations of $`F`$ (one per column).
- *Sparse Jacobians:* perturb structurally orthogonal columns simultaneously (graph colouring, Curtis–Powell–Reid); cost = number of colours, e.g. 3 for a tridiagonal Jacobian regardless of $`n`$.
- *Directional derivative* $`Jv \approx [F(x + hv) - F(x)]/h`$: one evaluation, used in Jacobian-free Newton–Krylov.

### Complex step

- Taylor: $`f(x + ih) = f(x) + ihf'(x) - \tfrac{h^2}{2}f''(x) - \dots`$, so $`f'(x) = \operatorname{Im} f(x + ih)/h + O(h^2)`$ with *no subtraction*, hence no cancellation.
- Requirements: real-analytic $`f`$; code must not use `abs` (replace with a complex-safe version), comparisons on complex numbers, or conjugation.
- Equivalent to forward-mode AD with dual numbers, minus some edge cases; dual numbers are cleaner where available.

### Hyper-dual and multicomplex step

- *Hyper-dual numbers* $`a + b\epsilon_1 + c\epsilon_2 + d\epsilon_1\epsilon_2`$ with $`\epsilon_1^2 = \epsilon_2^2 = 0`$: evaluating $`f(x + h_1\epsilon_1 + h_2\epsilon_2)`$ gives $`f'`$ in the $`\epsilon_1`$ part and $`\partial^2 f`$ in the $`\epsilon_1\epsilon_2`$ part, *exactly* for any $`h`$ (Fike & Alonso 2011). Equivalent to forward-over-forward AD.
- *Multicomplex step:* nest complex steps with independent imaginary units $`i_1, i_2, \dots`$; order-$`k`$ derivatives without subtraction, cost $`2^k`$ real evaluations.
- Same requirements as the complex step: analytic code, overloaded arithmetic. Where an AD tool exists, nested dual numbers do the same thing.

### Randomised gradients: SPSA and Gaussian smoothing

- *SPSA* (simultaneous perturbation, Spall): draw $`\Delta`$ with i.i.d. $`\pm 1`$ entries, then $`\hat g_i = \dfrac{f(x + h\Delta) - f(x - h\Delta)}{2h\,\Delta_i}`$. Two evaluations for all $`n`$ components; unbiased to $`O(h^2)`$ but noisy. Converges in stochastic approximation with decaying $`h_k`$ and step sizes.
- *Gaussian smoothing* (Nesterov–Spokoiny): $`\hat g = \dfrac{f(x + hu) - f(x)}{h}\,u`$, $`u \sim \mathcal N(0, I)`$, is an unbiased gradient of the smoothed $`f_h(x) = \mathbb E f(x + hu)`$. Averaging several directions reduces variance.
- Use for noisy or very expensive black boxes with many inputs and when a rough descent direction is enough (zeroth-order optimisation, see [derivative-free optimisation](/ref/optimise/global-derivative-free)). Not a substitute for an accurate Jacobian.

### Higher-order stencils and Richardson

- *Fornberg's algorithm* gives weights for any derivative order on arbitrary (non-uniform) stencils.
- *Richardson extrapolation:* with $`D(h) = f' + ch^2 + O(h^4)`$, combine $`\big(4D(h/2) - D(h)\big)/3 = f' + O(h^4)`$. Repeat (Romberg-style tableau) and stop when successive estimates agree; this also gives an error estimate.

## Other lenses

- *Interpolation.* A finite-difference formula is the derivative of the interpolating polynomial through the stencil points. Spectral differentiation is the same idea with global polynomials (see [Chebyshev & spectral methods](/ref/approx/chebyshev-spectral)).
- *PDEs.* Discretising derivatives on a grid gives the stencils of the method of lines; the truncation error analysis is identical (see [method of lines](/ref/integrate/pdes-method-of-lines)).
- *Linear algebra.* A difference operator is a (sparse, banded) matrix; its eigenvalues (e.g. $`-4\sin^2(k h/2)/h^2`$ for the second difference) explain stiffness and CFL limits.

## Pitfalls & diagnostics

- *Noisy or adaptive functions* (ODE solvers, Monte Carlo, iterative solvers with tolerances): finite differences amplify the noise by $`1/h`$. Tighten inner tolerances or fix random seeds and step sequences.
- *Branches and kinks* (max, abs, if-statements) give wrong derivatives when $`x \pm h`$ straddles a switch.
- *Scaling:* choose $`h`$ per component relative to each variable's typical size; a single global $`h`$ fails for variables spanning orders of magnitude.
- *Gradient checks:* compare AD or analytic gradients against central differences with relative error $`\lvert g - g_{fd}\rvert / \max(\lvert g\rvert, \lvert g_{fd}\rvert)`$; $`10^{-6}`$–$`10^{-8}`$ is a pass for smooth functions.

## See also

- [Automatic differentiation](/ref/diff/automatic-differentiation)
- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Stochastic gradient estimators](/ref/diff/stochastic-gradients)
- [Method of lines](/ref/integrate/pdes-method-of-lines)
- [Floating-point arithmetic](/ref/reliable/floating-point)

*References:*

- Nocedal & Wright, _Numerical Optimization_ (2nd ed.), ch. 8.
- Martins, Sturdza & Alonso, "The complex-step derivative approximation", ACM TOMS (2003).
- Fike & Alonso, "The development of hyper-dual numbers for exact second-derivative calculations", AIAA (2011).
- Spall, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation", IEEE TAC (1992).
- Nesterov & Spokoiny, "Random gradient-free minimization of convex functions", Found. Comput. Math. (2017).
- Fornberg, "Generation of finite difference formulas on arbitrarily spaced grids", Math. Comp. (1988).
