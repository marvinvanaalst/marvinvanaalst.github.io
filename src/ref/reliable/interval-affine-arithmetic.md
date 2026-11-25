---
title: 'Interval & affine arithmetic'
description: 'Rigorous enclosures of results: interval arithmetic, the dependency problem, affine arithmetic, Taylor models and validated numerics.'
aliases:
  [
    interval arithmetic,
    affine arithmetic,
    Taylor models,
    validated numerics,
    verified computing,
    dependency problem,
    wrapping effect,
    interval Newton,
    directed rounding
  ]
tags: [verification, rounding error, enclosure]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- Compute with *sets* guaranteed to contain the true result, including all rounding errors.
- *Intervals* are cheap (2–4× float cost) but overestimate whenever a variable appears more than once (dependency problem).
- *Affine arithmetic* tracks linear correlations between quantities, so it overestimates much less for smooth functions; cost grows with the number of noise symbols.
- *Taylor models* (polynomial + interval remainder) are the tool for validated ODE integration and global optimisation.
- Use these for proofs, verified bounds, robust geometric predicates and branch-and-bound; not as a general replacement for floats.

## Decision table

| Method | Representation | Cost vs float | Overestimation | Use when |
| --- | --- | --- | --- | --- |
| Interval arithmetic | $`[\underline x, \overline x]`$ | 2–4× | grows with dependencies, wrapping | bounds of simple expressions, branch & bound |
| Centred / mean-value form | $`f(c) + F'(X)(X - c)`$ | needs derivative | quadratic in width | narrow boxes, interval Newton |
| Affine arithmetic | $`x_0 + \sum_i x_i\varepsilon_i`$, $`\varepsilon_i \in [-1, 1]`$ | \~$`k`$ noise terms | linear correlations exact | smooth expressions, longer computations |
| Taylor models | polynomial $`p(x)`$ + remainder interval | high | high-order in width | validated ODE solving, verified global optimisation |
| Ball arithmetic (Arb) | midpoint + radius | \~1–2× multiprecision | like intervals | high-precision verified computation |

## Variants

### Interval arithmetic

- *Operations* with outward (directed) rounding:
  - $`[a, b] + [c, d] = [a + c, b + d]`$
  - $`[a, b] \cdot [c, d] = [\min(ac, ad, bc, bd), \max(\dots)]`$
  - Division by an interval containing 0 gives the whole line (or a union with extended arithmetic).
- *Fundamental theorem:* evaluating any expression for $`f`$ with intervals gives an enclosure of the range of $`f`$ over the box.
- *Dependency problem:* $`X - X`$ for $`X = [0, 1]`$ gives $`[-1, 1]`$, not $`0`$. Each occurrence is treated as independent. Rewrite expressions to use each variable once where possible.
- *Wrapping effect:* enclosing a rotated box in an axis-aligned box at every step makes widths explode in iterated maps and ODEs.

### Interval Newton

For a root of $`f`$ in box $`X`$ with midpoint $`c`$:

1. $`N(X) = c - F'(X)^{-1} f(c)`$ (interval Jacobian $`F'(X)`$).
2. Replace $`X`$ by $`X \cap N(X)`$.
3. If $`N(X) \subseteq X`$: a unique root exists in $`X`$ (existence and uniqueness proof). If $`X \cap N(X) = \emptyset`$: no root.

Combined with bisection this finds *all* roots in a box with guarantees. Krawczyk's operator avoids inverting interval matrices.

### Affine arithmetic

- A quantity is $`\hat x = x_0 + \sum_i x_i\varepsilon_i`$ with shared noise symbols $`\varepsilon_i \in [-1, 1]`$.
- Linear operations are exact: $`\hat x - \hat x = 0`$.
- Nonlinear operations (product, $`\exp`$) get a best linear approximation plus a *new* noise symbol for the error.
- Converts to an interval by $`x_0 \pm \sum_i\lvert x_i\rvert`$.
- Zonotopes (sets of affine forms) are the geometric view, used in reachability analysis.

### Taylor models and validated ODEs

- Represent $`f`$ over a box as a degree-$`n`$ polynomial plus an interval remainder.
- Validated ODE solvers (e.g. Lohner's method, Taylor-model integrators) step forward with a guaranteed enclosure of all trajectories from a set of initial conditions, fighting the wrapping effect with local coordinate changes.

## Other lenses

- *Abstract interpretation.* Intervals and zonotopes are abstract domains: sound over-approximations of sets of program states. Directed rounding is the soundness condition. See [lattices & fixpoints](/ref/programs/lattices-fixpoints-dataflow).
- *Optimisation.* Interval bounds give lower bounds for branch-and-bound global optimisation; the dependency problem is why relaxations (McCormick, αBB) matter. See [mixed-integer & SAT/SMT](/ref/optimise/mixed-integer-sat-smt).
- *Uncertainty.* Intervals are worst-case (set-based) uncertainty propagation, in contrast to probabilistic propagation (Monte Carlo, polynomial chaos).

## Pitfalls & diagnostics

- *Compiler and library rounding:* soundness requires directed rounding for every operation and correctly bounded elementary functions. Generic math libraries don't provide that.
- *Exploding widths* are usually dependency or wrapping, not a bug. Subdivide the input box, use centred forms, or move to affine arithmetic / Taylor models.
- *Comparisons are three-valued:* $`[1, 3] \lt [2, 4]`$ is neither true nor false. Branching code must handle "undecided".
- *Proof ≠ precision:* a valid but enormous enclosure proves nothing useful; always look at widths.

## See also

- [Floating-point arithmetic](/ref/reliable/floating-point)
- [Nonlinear roots](/ref/solve/nonlinear-roots)
- [Geometry primitives](/ref/space/geometry-primitives) (robust predicates)
- [Lattices, fixpoints & dataflow](/ref/programs/lattices-fixpoints-dataflow)

*References:*

- Moore, Kearfott & Cloud, _Introduction to Interval Analysis_.
- Tucker, _Validated Numerics_.
- de Figueiredo & Stolfi, "Affine arithmetic: concepts and applications", Numer. Algorithms (2004).
