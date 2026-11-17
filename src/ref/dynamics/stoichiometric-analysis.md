---
title: 'Stoichiometric network analysis'
description: 'Null spaces of the stoichiometric matrix, conservation laws, elementary flux modes, and flux balance analysis and FVA as linear programs.'
aliases:
  [
    stoichiometric matrix,
    null space,
    kernel,
    left null space,
    conservation laws,
    moiety conservation,
    elementary flux modes,
    EFM,
    extreme pathways,
    flux balance analysis,
    FBA,
    flux variability analysis,
    FVA,
    parsimonious FBA,
    constraint-based modelling,
    COBRA,
    gene knockouts,
    MOMA,
    thermodynamic constraints
  ]
tags: [metabolism, networks, linear programming, linear algebra]
order: 5
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- The stoichiometric matrix $N$ (species × reactions) gives $x' = Nv(x)$. Its structure alone answers a lot:
  - **Right null space** ($Nv = 0$): steady-state flux distributions.
  - **Left null space** ($LN = 0$): conservation laws $Lx = \text{const}$. Remove dependent species before simulating or solving for steady states.
- **Elementary flux modes:** minimal pathways (support-minimal steady-state fluxes respecting irreversibility). Complete but combinatorially explosive for genome-scale models.
- **Flux balance analysis:** $\max c^\top v$ s.t. $Nv = 0$, $l \le v \le u$. One LP, scales to genome-scale models. **FVA**: $2n$ LPs for flux ranges over the optimal face.
- FBA predictions depend entirely on the objective and bounds; the LP optimum is usually not unique.

## Decision table

| Analysis                    | Computation                                            | Cost                          | Answers                                      |
| --------------------------- | ------------------------------------------------------ | ----------------------------- | -------------------------------------------- |
| Rank, null spaces           | SVD / rational QR of $N$                               | $O(mn^2)$                     | degrees of freedom, conservation laws        |
| Conservation laws           | left null space, preferably non-negative integer basis | linear algebra / extreme rays | moiety pools, reduced models                 |
| Elementary flux modes       | double description / Nullspace algorithm               | exponential in network size   | all minimal pathways, yields                 |
| FBA                         | one LP                                                 | fast (simplex / IPM)          | optimal growth / yield under constraints     |
| Parsimonious FBA            | LP (+ minimise $\sum\lvert v\rvert$)                   | 2 LPs                         | a unique-ish, enzyme-efficient flux          |
| FVA                         | $2n$ LPs                                               | moderate, parallel            | flux ranges, essential and blocked reactions |
| Knockouts (FBA, MOMA, ROOM) | LP / QP / MILP per knockout                            | many solves                   | essential genes, mutant phenotypes           |
| Flux sampling               | MCMC (hit-and-run) on the polytope                     | expensive                     | distribution of feasible fluxes              |
| Thermodynamic FBA           | MILP with $\Delta G$ constraints                       | harder                        | loop-free, thermodynamically feasible fluxes |

## Variants

### Null spaces and conservation laws

- **Rank** $r$ of $N$ ($m$ species, $n$ reactions): $n - r$ independent steady-state flux directions, $m - r$ conservation laws.
- **Conservation laws:** rows of $L$ with $LN = 0$ give $Lx(t) = Lx(0)$. Examples: total enzyme, ATP + ADP + AMP, NAD + NADH.
- **Model reduction:** split $x$ into independent $x_i$ and dependent $x_d$; $x_d = c - L_0x_i$. Simulate only $x_i$; the reduced Jacobian is non-singular at steady state.
- Prefer sparse, non-negative integer bases (biochemically meaningful pools) over SVD bases.

### Elementary flux modes

- Steady-state flux cone: $\{v : Nv = 0,\ v_{\text{irr}} \ge 0\}$. EFMs are its support-minimal vectors (generalisation of extreme rays).
- Every feasible steady-state flux is a non-negative combination of EFMs.
- Computed by the double description method; numbers grow exponentially (millions for central metabolism). Minimal cut sets (dual concept) identify knockout strategies.

### Flux balance analysis

- LP: $\max_v c^\top v$ s.t. $Nv = 0$, $l \le v \le u$. Typical $c$ selects a biomass reaction; bounds encode uptake rates, irreversibility, knockouts.
- **Non-uniqueness:** the optimum is a face of the polytope, not a point. Report FVA ranges or use parsimonious FBA (minimise total flux at optimal objective).
- **Dual variables (shadow prices)** give the marginal value of each metabolite constraint.
- Enzyme-constrained variants (GECKO, resource balance analysis) add proteome capacity constraints.

### Flux variability analysis

1. Solve FBA, get optimal objective $z^*$.
2. Add constraint $c^\top v \ge \alpha z^*$ ($\alpha \le 1$ for sub-optimality).
3. For each reaction $j$: minimise and maximise $v_j$ (two LPs).

- Blocked reactions ($v_j = 0$ always) and essential ones (bounds exclude 0) fall out directly.

### Knockouts and loops

- **Single knockouts:** set bounds to zero and re-solve (warm starts make this fast).
- **MOMA:** QP minimising distance to the wild-type flux; **ROOM:** MILP minimising the number of large changes.
- **Thermodynamically infeasible loops** (internal cycles with net flux) appear in FBA solutions; remove with loopless FBA (MILP) or thermodynamic constraints.

## Other lenses

- **Linear programming.** FBA is an LP, FVA a family of LPs, knockout design a bilevel/MILP problem (OptKnock); dual LPs give shadow prices and reduced costs (see [constrained optimisation](/ref/optimise/constrained) and [mixed-integer](/ref/optimise/mixed-integer-sat-smt)).
- **Convex geometry.** The flux space is a polyhedral cone (or polytope with bounds); EFMs are its extreme rays, sampling explores its volume.
- **Dynamics.** Conservation laws define the stoichiometric compatibility classes on which dynamics live; reaction network theory makes statements about steady states within each class (see [reaction network theory](/ref/dynamics/reaction-network-theory)).

## Pitfalls & diagnostics

- **Numerical rank:** stoichiometric matrices have small integer entries; use exact or rational arithmetic for null spaces of small networks, SVD with a sensible threshold for large ones.
- **Reporting single FBA solutions** as "the" flux distribution ignores alternative optima; always check FVA.
- **Objective choice** (biomass, ATP, yield) drives the answer; justify it.
- **Unit and bound errors** (mmol/gDW/h vs per hour, missing exchange reactions) produce infeasible or unbounded LPs; check feasibility with a trivial objective first.
- **Simulating with redundant conservation laws** leads to singular Jacobians and drift; reduce first.

## See also

- [Constrained optimisation](/ref/optimise/constrained)
- [Mixed-integer & SAT/SMT](/ref/optimise/mixed-integer-sat-smt)
- [Reaction network theory](/ref/dynamics/reaction-network-theory)
- [Steady states & stability](/ref/dynamics/steady-states-stability)

**References:**

- Palsson, _Systems Biology: Constraint-based Reconstruction and Analysis_.
- Orth, Thiele & Palsson, "What is flux balance analysis?", Nat. Biotechnol. (2010).
- Klamt & Stelling, "Two approaches for metabolic pathway analysis?", Trends Biotechnol. (2003).
