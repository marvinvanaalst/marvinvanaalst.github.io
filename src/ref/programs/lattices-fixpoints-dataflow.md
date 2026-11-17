---
title: 'Lattices, fixpoints & dataflow analysis'
description: 'Partial orders and lattices, Kleene and Tarski fixpoint theorems, worklist dataflow analysis, abstract interpretation with Galois connections, and widening for termination.'
aliases:
  [
    lattice,
    partial order,
    poset,
    complete lattice,
    join,
    meet,
    monotone function,
    fixpoint,
    least fixed point,
    Kleene fixpoint theorem,
    Knaster-Tarski,
    dataflow analysis,
    liveness,
    reaching definitions,
    available expressions,
    constant propagation,
    worklist algorithm,
    abstract interpretation,
    abstract domain,
    Galois connection,
    widening,
    narrowing,
    interval domain,
    octagons,
    soundness,
    static analysis
  ]
tags: [programming languages, static analysis, order theory]
order: 7
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Static analyses** compute facts about all executions by solving equations over a **lattice** of abstract facts; the answer is the **least fixpoint** of a monotone transfer function.
- **Kleene iteration** from bottom ($\bot, f(\bot), f^2(\bot), \dots$) reaches the least fixpoint for continuous $f$; on finite-height lattices it terminates.
- **Dataflow analysis:** per-program-point facts, forward (reaching definitions, constant propagation) or backward (liveness), may (union) or must (intersection); solved with a **worklist** in near-linear time for bit-vector problems.
- **Abstract interpretation:** relate concrete semantics to abstract domains via a **Galois connection**; soundness by construction. Domains: signs, intervals, octagons, polyhedra (increasing precision and cost).
- **Infinite-height domains** (intervals) need **widening** to terminate, then **narrowing** to recover precision.

## Decision table

| Analysis / domain              | Direction | Lattice / merge            | Cost                                   | Use                                                |
| ------------------------------ | --------- | -------------------------- | -------------------------------------- | -------------------------------------------------- |
| Reaching definitions           | forward   | sets, union (may)          | bit vectors, fast                      | use-def chains                                     |
| Available expressions          | forward   | sets, intersection (must)  | bit vectors                            | common subexpression elimination                   |
| Liveness                       | backward  | sets, union                | bit vectors                            | register allocation, dead code                     |
| Very busy expressions          | backward  | intersection               | bit vectors                            | code hoisting                                      |
| Constant propagation           | forward   | flat lattice per variable  | fast (sparse on SSA: SCCP)             | constant folding, branch pruning                   |
| Signs / parity                 | forward   | small finite lattices      | cheap                                  | simple checks                                      |
| Intervals                      | forward   | intervals, widening needed | $O(n)$ per operation                   | bounds checks, overflow detection                  |
| Octagons ($\pm x \pm y \le c$) | forward   | relational                 | $O(n^3)$                               | relations between variables (loop bounds)          |
| Convex polyhedra               | forward   | relational, very precise   | exponential worst case                 | precise numeric invariants, polyhedral compilation |
| Points-to / alias analysis     | forward   | sets of abstract locations | Andersen $O(n^3)$, Steensgaard ~linear | memory safety, optimisation                        |

## Variants

### Order theory basics

- **Poset:** reflexive, antisymmetric, transitive $\sqsubseteq$. **Lattice:** every pair has a join $\sqcup$ and meet $\sqcap$. **Complete lattice:** every subset has them (bottom $\bot$, top $\top$).
- **Monotone** $f$: $x \sqsubseteq y \Rightarrow f(x) \sqsubseteq f(y)$.
- **Knaster–Tarski:** a monotone function on a complete lattice has a complete lattice of fixpoints; the least is $\inf\{x : f(x) \sqsubseteq x\}$.
- **Kleene:** for Scott-continuous $f$, $\text{lfp}(f) = \bigsqcup_nf^n(\bot)$.

### Dataflow analysis

- For each basic block: transfer function $f_B$ (e.g. $\text{out} = \text{gen}_B \cup (\text{in} \setminus \text{kill}_B)$), merge at joins with $\sqcup$.
- **Worklist algorithm:**
  1. Initialise all facts to $\bot$ (or $\top$ for must-analyses); put all blocks on the worklist.
  2. Pop a block, recompute its output from its inputs.
  3. If the output changed, push its successors (forward) or predecessors (backward).
  4. Repeat until the worklist is empty.
- Process blocks in reverse postorder (forward) or postorder (backward) to converge in few passes; bit-vector problems converge in $O(\text{loop depth})$ passes.
- **Sparse analyses on SSA** attach facts to variables instead of program points (SCCP: sparse conditional constant propagation).

### Abstract interpretation

- Concrete domain: sets of states $\mathcal P(\Sigma)$; abstract domain $A$; **Galois connection** $\alpha : \mathcal P(\Sigma) \to A$, $\gamma : A \to \mathcal P(\Sigma)$ with $\alpha(S) \sqsubseteq a \iff S \subseteq\gamma(a)$.
- Best abstract transformer: $f^\sharp = \alpha\circ f\circ\gamma$; any sound over-approximation works.
- Soundness: the abstract fixpoint over-approximates all reachable concrete states. Precision vs cost = choice of domain.

### Widening and narrowing

- Intervals have infinite ascending chains ($[0, 0], [0, 1], [0, 2], \dots$); iteration may not terminate.
- **Widening** $\nabla$: extrapolate unstable bounds to $\pm\infty$ (e.g. $[0, 1]\nabla[0, 2] = [0, \infty]$), guaranteeing termination at the cost of precision. Apply at loop heads.
- **Narrowing** $\Delta$: iterate downward from the widened result to regain precision (e.g. recover the loop bound from the exit condition).
- Widening with thresholds (use constants from the program) improves precision.

## Other lenses

- **Category theory.** Galois connections are adjunctions between posets (see [category theory](/ref/programs/category-theory-core)).
- **Numerics.** Interval and affine arithmetic are abstract interpretations of floating-point programs (see [interval arithmetic](/ref/reliable/interval-affine-arithmetic)); polyhedral analysis feeds loop optimisation (see [polyhedral loop transforms](/ref/programs/polyhedral-loop-transforms)).
- **Algebra.** Dataflow frameworks are semiring/lattice computations on graphs; the meet-over-all-paths solution equals the fixpoint for distributive frameworks (see [algebraic structures](/ref/programs/algebraic-structures)).
- **Graphs.** Iteration order, SCCs (process loops together) and dominators shape analysis efficiency (see [SSA & IRs](/ref/programs/ssa-ir)).

## Pitfalls & diagnostics

- **Non-monotone transfer functions** break termination and soundness arguments.
- **Wrong initialisation** ($\bot$ vs $\top$) for must-analyses gives unsound results.
- **False positives** from imprecise domains; choose relational domains or path sensitivity where needed, at a cost.
- **Unsound shortcuts** (ignoring aliasing, reflection, concurrency) are common in practical analysers; document them.
- **Widening too early** destroys precision; delay it a few iterations.

## See also

- [SSA & intermediate representations](/ref/programs/ssa-ir)
- [Interval & affine arithmetic](/ref/reliable/interval-affine-arithmetic)
- [Algebraic structures](/ref/programs/algebraic-structures)
- [Polyhedral loop transforms](/ref/programs/polyhedral-loop-transforms)

**References:**

- Nielson, Nielson & Hankin, _Principles of Program Analysis_.
- Cousot & Cousot, "Abstract interpretation: a unified lattice model for static analysis of programs", POPL (1977).
- Møller & Schwartzbach, _Static Program Analysis_ (free online lecture notes).
