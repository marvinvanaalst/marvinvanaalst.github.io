---
title: 'Type inference & unification'
description: 'First-order unification, Hindley–Milner with algorithm W/J, let-generalisation and the value restriction, constraint-based inference, and bidirectional typing for richer type systems.'
aliases:
  [
    type inference,
    unification,
    most general unifier,
    occurs check,
    union-find,
    Hindley-Milner,
    algorithm W,
    algorithm J,
    let polymorphism,
    generalization,
    instantiation,
    value restriction,
    principal types,
    constraint-based type inference,
    HM(X),
    bidirectional type checking,
    local type inference,
    subtyping inference,
    row polymorphism,
    type classes,
    level-based generalization
  ]
tags: [type theory, programming languages, compilers]
order: 6
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Unification** solves equations between type terms with variables, finding the **most general unifier**; near-linear time with union-find. The occurs check prevents infinite types.
- **Hindley–Milner (HM):** complete, decidable inference with principal types for let-polymorphism. **Algorithm J** (mutable union-find, level-based generalisation) is what real compilers use; **algorithm W** is the substitution-passing textbook version.
- **Generalise at `let`**, instantiate at use sites; with side effects, apply the **value restriction**.
- **Constraint-based inference** (generate constraints, then solve) separates concerns, gives better error messages and extends to type classes, records, subtyping (HM(X)).
- **Beyond HM** (rank-N polymorphism, subtyping, dependent types, GADTs): use **bidirectional typing**: check against known types, infer where easy, require annotations at the boundaries.

## Decision table

| Approach                                | Supports                                  | Decidable / complete | Use when                                    |
| --------------------------------------- | ----------------------------------------- | -------------------- | ------------------------------------------- |
| First-order unification                 | type equality constraints                 | yes, near-linear     | the core of everything below                |
| Algorithm W                             | HM let-polymorphism                       | yes, principal types | teaching, proofs                            |
| Algorithm J / level-based               | HM, efficient                             | yes                  | implementations (OCaml-style)               |
| Constraint generation + solving (HM(X)) | HM + type classes, records, units         | depends on X         | good errors, extensible systems             |
| Row polymorphism                        | extensible records / variants / effects   | yes (with HM)        | structural records, effect rows             |
| Bidirectional typing                    | rank-N, subtyping, dependent types, GADTs | with annotations     | rich type systems, predictable inference    |
| Local type inference (Pierce–Turner)    | subtyping + generics                      | partial              | OO-ish languages (Scala, TypeScript-like)   |
| Flow / occurrence typing                | narrowing by control flow                 | partial              | gradual typing, TypeScript, Python checkers |
| SMT-backed refinement inference         | refinement predicates                     | depends on logic     | Liquid types                                |

## Variants

### Unification

1. Constraints $\tau_1 = \tau_2$ over type constructors and variables.
2. Same constructor on both sides: unify arguments pairwise; different constructors: type error.
3. Variable $\alpha = \tau$: if $\alpha$ occurs in $\tau$ (and $\tau \ne \alpha$), fail (**occurs check**: no infinite types); otherwise bind $\alpha \mapsto \tau$.
4. With union-find (path compression + union by rank), solving is nearly linear in the size of the constraints.

### Hindley–Milner

- **Instantiation:** each use of a polymorphic variable gets fresh type variables for its $\forall$-bound variables.
- **Generalisation at `let`:** quantify over type variables not free in the environment: `let id = λx.x` gets $\forall\alpha.\,\alpha\to\alpha$.
- **Algorithm W:** recursive over the AST, returning a substitution and a type; composes substitutions.
- **Algorithm J / levels:** destructive unification on mutable type variables; each variable records the `let`-nesting level where it was created; generalise variables whose level is deeper than the current one (efficient generalisation without scanning the environment).
- Worst-case exponential (nested lets), linear-ish in practice.

### Value restriction

- With mutable references, generalising `let r = ref []` to $\forall\alpha.\,\alpha\text{ list ref}$ is unsound.
- **Value restriction:** generalise only syntactic values (lambdas, constants, constructors); relaxed variants generalise covariant positions.

### Constraint-based inference and errors

- Phase 1: walk the program generating constraints with source locations. Phase 2: solve (unification + extra theories: type class instances, record fields, subtyping).
- Better errors: report conflicting constraints with their origins, minimal unsatisfiable subsets, or heuristics on likely culprits.

### Bidirectional typing

- Two judgements: **check** $\Gamma\vdash e \Leftarrow A$ (type known from context) and **synthesise** $\Gamma\vdash e \Rightarrow A$.
- Lambdas are checked (parameter types from the expected type), applications synthesise the function then check arguments, annotations switch modes.
- Scales to features where full inference is impossible; annotations required only at a few predictable places (top-level definitions, higher-rank arguments).

## Other lenses

- **Logic and SAT/SMT.** Unification is the equational-logic core of Prolog resolution; richer type systems reduce inference to constraint solving with SMT (see [mixed-integer & SAT/SMT](/ref/optimise/mixed-integer-sat-smt)).
- **Graph algorithms.** Union-find (disjoint sets) is the data structure behind efficient unification and also connected components (see [graph algorithms](/ref/graphs/graph-algorithms)).
- **Term rewriting.** Unification and matching are the operations of rewriting systems; e-graphs use congruence closure, a close relative (see [term rewriting & e-graphs](/ref/programs/term-rewriting-egraphs)).

## Pitfalls & diagnostics

- **Error locality:** HM reports the error where unification fails, often far from the real mistake; constraint-based reporting or annotations help.
- **Polymorphic recursion** needs annotations (inference is undecidable).
- **Generalisation bugs** (generalising variables still in the environment) are a classic implementation error; levels make it systematic.
- **Overloading and defaulting** (numeric literals, type classes) cause ambiguity; specify defaulting rules.
- **Performance** of naive substitution-based W is quadratic or worse; use mutable union-find.

## See also

- [Lambda calculus & types](/ref/programs/lambda-calculus-types)
- [Term rewriting & e-graphs](/ref/programs/term-rewriting-egraphs)
- [Graph algorithms](/ref/graphs/graph-algorithms) (union-find)
- [Mixed-integer & SAT/SMT](/ref/optimise/mixed-integer-sat-smt)

**References:**

- Damas & Milner, "Principal type-schemes for functional programs", POPL (1982).
- Pottier & Rémy, "The essence of ML type inference", in _Advanced Topics in Types and Programming Languages_ (2005).
- Dunfield & Krishnaswami, "Bidirectional typing", ACM Comput. Surv. (2021).
