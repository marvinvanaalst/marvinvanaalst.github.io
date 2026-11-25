---
title: 'Term rewriting & e-graphs'
description: 'Rewrite systems, termination and confluence (Knuth–Bendix), rewriting strategies, congruence closure, and e-graphs with equality saturation for optimisation and simplification.'
aliases:
  [
    term rewriting,
    rewrite rules,
    rewriting strategies,
    normal form,
    termination,
    confluence,
    Church-Rosser,
    critical pairs,
    Knuth-Bendix completion,
    congruence closure,
    e-graph,
    equality saturation,
    egg,
    extraction,
    cost function,
    phase ordering,
    peephole optimization,
    algebraic simplification,
    computer algebra,
    pattern matching,
    unification
  ]
tags: [programming languages, compilers, symbolic computation]
order: 8
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Term rewriting:* repeatedly replace subterms matching a rule's left-hand side by its right-hand side. Simplifiers, optimisers, computer algebra and type-level computation are rewriting systems.
- Good systems are *terminating* (no infinite rewriting) and *confluent* (order doesn't matter), so every term has a unique normal form. Check confluence via *critical pairs*; *Knuth–Bendix completion* tries to add rules to achieve it.
- Destructive rewriting suffers from *phase ordering*: applying one rule can block a better one.
- *E-graphs + equality saturation* apply all rules non-destructively, storing all equivalent forms compactly; then *extract* the cheapest. Great for optimisers (tensor graphs, floating-point accuracy, DSP, CAD) where rules interact.
- E-graphs grow fast; limit by iteration count, node budgets or rule scheduling.

## Decision table

| Approach | Strength | Weakness | Use when |
| --- | --- | --- | --- |
| Confluent + terminating TRS | unique normal forms, simple implementation | hard to design for rich rule sets | canonical simplification (normalisation) |
| Strategic rewriting (innermost, outermost, custom strategies) | control over application order | phase ordering, local optima | compilers passes, program transformation |
| Peephole optimisation | cheap local improvements | myopic | late compiler stages |
| Knuth–Bendix completion | turns equations into a convergent TRS | may not terminate | word problems, algebraic theories |
| Congruence closure | decides ground equalities efficiently | ground terms only | SMT (theory of equality), e-graph core |
| Equality saturation (e-graphs) | explores all rewrites, no phase ordering | blow-up, extraction can be hard (ILP) | optimisation with many interacting rules |
| E-graphs + ILP extraction | optimal under DAG-sharing costs | expensive | small graphs where optimality matters |

## Variants

### Rewriting basics

- Rules $`l \to r`$ with variables; matching finds substitutions $`\sigma`$ with $`l\sigma`$ equal to the subterm.
- *Strategies:* innermost (call-by-value-like), outermost (lazy), bottom-up/top-down traversals, fixpoint iteration of rule sets.
- *Termination proofs:* find a well-founded order where every rule decreases terms (recursive path orders, polynomial interpretations).

### Confluence

- *Local confluence + termination ⇒ confluence* (Newman's lemma).
- *Critical pairs:* overlaps between left-hand sides; if all critical pairs are joinable, the system is locally confluent.
- *Knuth–Bendix:* orient unjoinable critical pairs into new rules using a reduction order; may succeed (convergent system), fail (unorientable equation) or run forever.

### Congruence closure

- Maintain equivalence classes of terms with union-find; when two terms are merged, their parents with equal function symbols and equivalent arguments are merged too (congruence).
- Decides $`E \vDash s = t`$ for ground equations in $`O(n\log n)`$; the backbone of SMT's equality theory and of e-graphs.

### E-graphs and equality saturation

- *E-graph:* e-classes (sets of equivalent e-nodes); e-nodes are operators whose children are e-classes. Represents exponentially many equivalent terms in polynomial space.
- *Equality saturation:*
  1. Insert the input term.
  2. Repeatedly: find all matches of all rules (e-matching), add right-hand sides, merge e-classes, restore congruence invariants (rebuilding, as in the egg library).
  3. Stop at saturation or when budgets are exhausted.
  4. *Extract* the best term by a cost function: greedy bottom-up for tree costs, ILP for costs with sharing.
- Extensions: e-class analyses (attach lattice facts like constant values or intervals), conditional rewrites, relational e-matching (Datalog-style, egglog).

### Applications

- Compiler optimisation (tensor graph rewrites, vectorisation, peephole superoptimisation), floating-point accuracy repair (Herbie), CAD simplification, theorem proving, symbolic algebra simplification.

## Other lenses

- *Lambda calculus.* β-reduction is a rewriting system; Church–Rosser is its confluence theorem (see [lambda calculus](/ref/programs/lambda-calculus-types)).
- *Algebra.* Rewriting decides word problems for algebraic theories (groups, rings) when a convergent system exists; Gröbner basis computation is Knuth–Bendix for polynomial ideals (see [reaction network theory](/ref/dynamics/reaction-network-theory)).
- *Logic programming.* Datalog evaluation and e-matching share relational join techniques; egglog unifies them.
- *Type inference.* Unification and congruence closure are close relatives, both using union-find (see [type inference](/ref/programs/type-inference-unification)).

## Pitfalls & diagnostics

- *Non-terminating rule sets* (commutativity, associativity both ways) in destructive rewriting; e-graphs handle them but blow up.
- *E-graph explosion:* associativity/commutativity with many terms grow combinatorially; use budgets, rule schedulers (backoff), or AC-specific handling.
- *Unsound rules* (e.g. floating-point reassociation assumed exact) propagate silently; tag rules with their side conditions.
- *Extraction cost model* must reflect reality (sharing, hardware costs), or the "optimal" term isn't.

## See also

- [Lambda calculus & types](/ref/programs/lambda-calculus-types)
- [Type inference & unification](/ref/programs/type-inference-unification)
- [SSA & intermediate representations](/ref/programs/ssa-ir)
- [Staging & partial evaluation](/ref/programs/staging-partial-evaluation)

*References:*

- Baader & Nipkow, _Term Rewriting and All That_.
- Willsey et al., "egg: fast and extensible equality saturation", POPL (2021).
- Tate, Stepp, Tatlock & Lerner, "Equality saturation: a new approach to optimization", POPL (2009).
