---
title: 'Chemical reaction network theory'
description: 'Deficiency zero and one theorems, complex balancing, multistationarity tests, toric steady states and symbolic steady-state solving with Gröbner bases.'
aliases:
  [
    CRNT,
    chemical reaction network theory,
    deficiency,
    deficiency zero theorem,
    deficiency one theorem,
    complex balanced,
    detailed balanced,
    weakly reversible,
    linkage classes,
    multistationarity,
    injectivity,
    toric steady states,
    Groebner basis,
    sum of squares,
    mass action,
    Horn-Jackson,
    Feinberg,
    absolute concentration robustness,
    ACR,
    persistence,
    global attractor conjecture
  ]
tags: [reaction networks, mass action, algebraic methods, metabolism]
order: 8
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- CRNT predicts qualitative behaviour of mass-action networks *from the reaction graph alone*, for all rate constants.
- *Deficiency* $`\delta = n - \ell - s`$ ($`n`$ complexes, $`\ell`$ linkage classes, $`s`$ = rank of the stoichiometric matrix).
- *Deficiency zero theorem:* weakly reversible and $`\delta = 0`$ ⇒ exactly one positive steady state in each stoichiometric compatibility class, locally asymptotically stable, no oscillations, for *any* rate constants. Not weakly reversible and $`\delta = 0`$ ⇒ no positive steady states at all.
- *Multistationarity:* injectivity criteria (determinant sign tests) rule it out; deficiency one algorithm and sign-condition methods detect it.
- *Steady-state varieties* of mass-action systems are algebraic; Gröbner bases, toric parametrisations and homotopy continuation compute them symbolically or completely.

## Decision table

| Question | Tool | Cost |
| --- | --- | --- |
| Unique stable steady state for all rates? | deficiency zero theorem (+ weak reversibility) | graph + rank: trivial |
| Can the network be multistationary? | injectivity test (sign of Jacobian determinant terms), deficiency one algorithm, CRNToolbox | small: easy; large: combinatorial |
| Which rates give multistationarity? | sign conditions, Kac–Rice / discriminants, sampling + homotopy | moderate–hard |
| Robust concentrations (ACR)? | Shinar–Feinberg theorem ($`\delta = 1`$ structure) | structural check |
| All steady states symbolically | Gröbner bases, toric parametrisation, elimination | exponential worst case |
| All steady states numerically | polyhedral homotopy continuation | grows with mixed volume |
| Global stability | complex balancing + pseudo-Helmholtz Lyapunov function | structural |
| Persistence (no species vanish) | siphon analysis | combinatorial |

## Variants

### Structure: complexes, linkage classes, deficiency

- *Complexes:* left- and right-hand sides of reactions (e.g. $`A + B`$, $`2C`$). The reaction graph has complexes as nodes.
- *Linkage classes:* connected components of the reaction graph. *Weakly reversible:* every component is strongly connected.
- *Deficiency* $`\delta = n - \ell - s \ge 0`$. Many textbook networks have $`\delta = 0`$ (e.g. reversible Michaelis–Menten mechanisms).

### Complex balancing

- A steady state is *complex balanced* if, at each complex, inflow equals outflow. Detailed balance (each reaction balanced) is a special case.
- Complex-balanced systems (Horn–Jackson): all positive steady states are complex balanced, exactly one per compatibility class, locally stable, with Lyapunov function $`G(x) = \sum_i x_i(\ln(x_i/x_i^*) - 1) + x_i^*`$.
- Deficiency zero + weak reversibility ⇒ complex balanced for all rate constants. With $`\delta \gt 0`$, complex balancing requires $`\delta`$ algebraic conditions on the rate constants.

### Multistationarity

- *Injectivity:* if the species formation rate function is injective on each compatibility class for all rate constants (checked by sign patterns of the Jacobian determinant, or via the network's species–reaction graph), multistationarity is impossible.
- *Deficiency one theorem/algorithm* (Feinberg): decides capacity for multiple steady states for many $`\delta = 1`$ networks.
- *Parametric search:* reduce steady states to a univariate polynomial (elimination), use sign changes of its coefficients (Descartes) or discriminants to find parameter regions with several positive roots.

### Algebraic steady-state computation

- Mass-action steady states solve polynomial equations $`N\,\operatorname{diag}(k)\,x^{Y} = 0`$ plus conservation laws.
- *Gröbner bases* (lex order) triangularise the system, eliminating variables; exact but expensive.
- *Toric steady states:* for many networks (including complex-balanced ones), positive steady states form a monomial parametrisation $`x = x^* \odot e^{v}`$, $`v \in \ker(\ldots)`$, giving explicit formulas.
- *Homotopy continuation* finds all complex solutions numerically; filter positive real ones (see [nonlinear equations](/ref/solve/nonlinear-roots)).
- *Linear elimination* for intermediates (Thomson–Gunawardena): steady-state expressions of intermediates via the matrix-tree theorem.

### Absolute concentration robustness

- A species has ACR if its steady-state value is the same in every compatibility class (independent of totals). Shinar–Feinberg: a structural condition on $`\delta = 1`$ networks (two non-terminal complexes differing in that species) guarantees it, as in bacterial two-component systems.

## Other lenses

- *Graph theory.* Most results are statements about the reaction graph and the species–reaction bipartite graph; the matrix-tree theorem gives steady states of linear subnetworks (King–Altman for enzyme kinetics).
- *Thermodynamics.* Complex balancing generalises detailed balance; the Lyapunov function is a free energy. Deficiency zero networks cannot sustain the non-equilibrium behaviours (oscillations, switches) that require energy-driven, higher-deficiency structure.
- *Algebraic geometry.* Steady-state varieties, toric ideals and Gröbner bases are the language; the "global attractor conjecture" (complex-balanced steady states are globally stable) is the famous open-turned-mostly-solved problem.
- *Constraint-based modelling.* Stoichiometric analysis uses $`N`$ without kinetics; CRNT adds mass-action kinetics but stays parameter-free (see [stoichiometric analysis](/ref/dynamics/stoichiometric-analysis)).

## Pitfalls & diagnostics

- *Mass-action only:* most theorems assume mass-action kinetics; Michaelis–Menten or Hill rate laws (from QSSA) change the picture. Analyse the elementary mechanism, not the reduced one.
- *Positive vs boundary steady states:* theorems concern positive steady states; boundary states (some species zero) need persistence/siphon arguments.
- *"Capacity for" multistationarity* means some rate constants allow it, not that your parameters do.
- *Symbolic methods blow up* with network size; use structure (linear elimination, toric parametrisations) before brute-force Gröbner bases.
- *Tools:* CRNToolbox, CoNtRol, Chemical Reaction Network Toolbox in Julia (Catalyst), Macaulay2 / Singular for algebra.

## See also

- [Stoichiometric analysis](/ref/dynamics/stoichiometric-analysis)
- [Steady states & stability](/ref/dynamics/steady-states-stability)
- [Bifurcation & continuation](/ref/dynamics/bifurcation-continuation)
- [Stochastic simulation](/ref/integrate/stochastic-simulation)

*References:*

- Feinberg, _Foundations of Chemical Reaction Network Theory_.
- Dickenstein, "Biochemical reaction networks: an invitation for algebraic geometers" (2016).
- Shinar & Feinberg, "Structural sources of robustness in biochemical reaction networks", Science (2010).
