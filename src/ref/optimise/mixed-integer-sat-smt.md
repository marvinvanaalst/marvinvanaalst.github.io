---
title: 'Mixed-integer programming, SAT & SMT'
description: 'Branch and bound, cutting planes, MILP modelling tricks (big-M, indicators), CDCL SAT solving and SMT: discrete decisions in optimisation, verification and type checking.'
aliases:
  [
    MILP,
    MIP,
    integer programming,
    branch and bound,
    branch and cut,
    cutting planes,
    Gomory cuts,
    LP relaxation,
    big-M,
    indicator constraints,
    SOS constraints,
    MINLP,
    SAT,
    CDCL,
    DPLL,
    unit propagation,
    clause learning,
    SMT,
    satisfiability modulo theories,
    Z3,
    constraint programming,
    combinatorial optimization,
    knapsack
  ]
tags: [optimisation, discrete, combinatorial, verification]
order: 10
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **MILP:** linear objective and constraints, some variables integer. Solved by **branch and cut** (LP relaxations + branching + cutting planes + heuristics). Modern solvers (Gurobi, CPLEX, HiGHS, SCIP) solve surprisingly large instances; the **formulation** matters more than the solver.
- **Modelling tricks:** binaries for on/off decisions, big-M or indicator constraints for logic, SOS constraints for piecewise linear functions. Tight formulations (small gap between LP relaxation and integer hull) solve orders of magnitude faster.
- **SAT:** Boolean satisfiability, NP-complete but CDCL solvers handle millions of clauses in practice.
- **SMT:** SAT plus theories (linear arithmetic, bit-vectors, arrays, uninterpreted functions). Z3/cvc5 power program verification, symbolic execution and refinement-type checking.
- **Constraint programming** for scheduling and highly combinatorial feasibility problems with global constraints.

## Decision table

| Problem                                              | Tool                                            | Notes                                          |
| ---------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------- |
| Linear + integer decisions                           | MILP (branch and cut)                           | most common; FBA knockouts, scheduling, design |
| Convex nonlinear + integer                           | MIQP / MISOCP, outer approximation              | solvers handle convex quadratic well           |
| Nonconvex nonlinear + integer                        | MINLP (spatial branch and bound, BARON, SCIP)   | hard; small-moderate size                      |
| Pure Boolean feasibility                             | SAT (CDCL)                                      | encode via Tseitin; huge instances feasible    |
| Logic + arithmetic, program properties               | SMT (Z3, cvc5)                                  | verification, symbolic execution, synthesis    |
| Scheduling, timetabling, routing (feasibility-heavy) | constraint programming (OR-Tools CP-SAT)        | global constraints, strong propagation         |
| Need any good solution fast                          | heuristics / metaheuristics (local search, LNS) | no optimality proof                            |
| Huge LP with many columns                            | column generation / branch and price            | cutting stock, vehicle routing                 |

## Variants

### Branch and bound / branch and cut

1. Solve the LP relaxation (drop integrality). Its value is a bound (lower bound for minimisation).
2. If the solution is integral: feasible incumbent. Otherwise pick a fractional variable $x_j = f$ and branch: $x_j \le \lfloor f\rfloor$ or $x_j \ge \lceil f\rceil$.
3. Prune nodes whose bound is worse than the incumbent.
4. **Cuts** (Gomory, cover, flow cover, clique) tighten relaxations; **presolve** removes and strengthens constraints; **primal heuristics** find incumbents early.
5. Stop when the **MIP gap** (incumbent vs best bound) is below tolerance.

- Dual simplex warm starts make re-solving child nodes fast.

### Modelling tricks

- **Big-M:** $x \le My$ with binary $y$ forces $x = 0$ when $y = 0$. Choose $M$ as small as valid; huge $M$ gives weak relaxations and numerical trouble.
- **Indicator constraints:** $y = 1 \Rightarrow a^\top x \le b$, handled natively by solvers (often better than big-M).
- **Logical constraints:** AND/OR/implication via linear inequalities on binaries; at most $k$ of a set via $\sum y_i \le k$.
- **Piecewise linear functions:** SOS2 constraints or incremental/convex-combination formulations.
- **Symmetry:** identical machines or items create equivalent solutions; add symmetry-breaking constraints or let the solver detect orbits.

### SAT: CDCL

- Input: CNF clauses. Core loop:
  1. **Decide:** assign a variable (VSIDS heuristic picks active variables).
  2. **Propagate:** unit propagation with two watched literals.
  3. **Conflict:** analyse the implication graph, learn a new clause (first UIP), backjump non-chronologically.
  4. **Restart** periodically, keeping learned clauses (with clause deletion).
- Encoding arithmetic and cardinality constraints into CNF efficiently (sorting networks, totalisers) matters.

### SMT

- **DPLL(T):** the SAT solver handles the Boolean structure; theory solvers check conjunctions of theory literals (simplex for linear arithmetic, congruence closure for equality, bit-blasting for bit-vectors) and return conflicts as clauses.
- Uses: bounded model checking, symbolic execution (KLEE), verification condition checking (Dafny, Why3, Liquid Haskell refinement types), program synthesis.
- Quantifiers make SMT undecidable in general; solvers use heuristics (E-matching).

## Other lenses

- **Polyhedral geometry.** Integer programming is LP over the convex hull of integer points; the art is describing that hull tightly with few inequalities (ideal formulations).
- **Logic and PL.** SAT is the propositional fragment; SMT adds first-order theories; type inference for advanced type systems and refinement checking reduce to constraint solving (see [type inference](/ref/programs/type-inference-unification)).
- **Biology.** Gene knockout design (OptKnock), thermodynamic loop removal and minimal cut sets in metabolic networks are MILPs (see [stoichiometric analysis](/ref/dynamics/stoichiometric-analysis)).
- **Global optimisation.** Spatial branch and bound for nonconvex continuous problems uses the same tree search with convex relaxations (McCormick envelopes) instead of integrality (see [global optimisation](/ref/optimise/global-derivative-free)).

## Pitfalls & diagnostics

- **Weak formulations** (large big-M, aggregated constraints) are the main cause of slow solves; compare the LP relaxation value to the best integer solution.
- **Numerics:** coefficients spanning many orders of magnitude cause wrong "optimal" answers; rescale and tighten bounds on all variables.
- **Report the MIP gap,** not just the solution; a time-limited run without a small gap is a heuristic result.
- **Encoding blow-up** in SAT/SMT: prefer native theories (arithmetic in SMT) over bit-blasting when possible.
- **Exponential worst case** is real; for huge instances use decomposition (Benders, Lagrangian relaxation, column generation) or heuristics.

## See also

- [Constrained optimisation](/ref/optimise/constrained)
- [Convex duality & proximal methods](/ref/optimise/convex-duality)
- [Stoichiometric analysis](/ref/dynamics/stoichiometric-analysis)
- [Type inference & unification](/ref/programs/type-inference-unification)

**References:**

- Wolsey, _Integer Programming_ (2nd ed.).
- Conforti, Cornuéjols & Zambelli, _Integer Programming_.
- Biere, Heule, van Maaren & Walsh (eds.), _Handbook of Satisfiability_ (2nd ed.).
- de Moura & Bjørner, "Satisfiability modulo theories: introduction and applications", CACM (2011).
