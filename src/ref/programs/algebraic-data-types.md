---
title: 'Algebraic data types & recursion schemes'
description: 'Sum and product types, recursive types as fixed points of functors, F-algebras and catamorphisms (folds), anamorphisms (unfolds), hylomorphisms, and data vs codata.'
aliases:
  [
    algebraic data types,
    ADT,
    sum types,
    product types,
    tagged unions,
    pattern matching,
    exhaustiveness,
    recursive types,
    fixed point of functor,
    F-algebra,
    initial algebra,
    catamorphism,
    fold,
    anamorphism,
    unfold,
    hylomorphism,
    paramorphism,
    recursion schemes,
    codata,
    coalgebra,
    streams,
    GADTs,
    type algebra,
    zippers,
    derivatives of types
  ]
tags: [type theory, programming languages, category theory]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **ADTs:** build types from products (records, tuples: "and"), sums (tagged unions: "or"), unit and empty types. Pattern matching with exhaustiveness checking makes illegal states unrepresentable.
- **Recursive types** are fixed points of functors: `List A = μX. 1 + A × X`.
- A **fold (catamorphism)** is defined once per type by an **F-algebra** `F B → B`; it replaces constructors with functions. Unfolds (anamorphisms) build structures from seeds; **hylomorphisms** fuse unfold-then-fold without building the intermediate structure.
- **Data** (finite, inductive, consumed by folds) vs **codata** (possibly infinite, coinductive, observed by destructors: streams, processes).
- **Type algebra:** types obey arithmetic laws ($A\times(B + C) \cong A\times B + A\times C$, $1 + A$ = Option), and even derivatives (zippers).

## Decision table

| Construct            | Categorical reading                    | Programming form                           | Use for                                         |
| -------------------- | -------------------------------------- | ------------------------------------------ | ----------------------------------------------- |
| Product $A \times B$ | product                                | struct, tuple, record                      | data that has all fields                        |
| Sum $A + B$          | coproduct                              | enum with payloads, tagged union, `Either` | alternatives, states, ASTs, results             |
| Unit / empty         | terminal / initial object              | `()` / `Never`                             | trivial and impossible values                   |
| Recursive type       | initial algebra $\mu F$                | `List`, `Tree`, AST                        | finite structured data                          |
| Codata               | final coalgebra $\nu F$                | streams, lazy infinite structures, objects | infinite / interactive behaviour                |
| Catamorphism (fold)  | unique algebra morphism from $\mu F$   | `foldr`, visitors, evaluators              | consume structures (evaluate, sum, print)       |
| Anamorphism (unfold) | unique coalgebra morphism into $\nu F$ | `unfold`, generators                       | build structures from seeds                     |
| Hylomorphism         | ana then cata, fused                   | divide-and-conquer recursion               | quicksort, mergesort, many recursive algorithms |
| Paramorphism         | fold with access to substructures      | primitive recursion                        | factorial-like recursion, `tails`               |
| GADTs                | indexed families                       | type-refined constructors                  | typed ASTs, invariants in types                 |

## Variants

### Sums, products and pattern matching

- Sum types + exhaustive matching: the compiler checks that every case is handled; adding a variant shows all places to update.
- **Make illegal states unrepresentable:** model "loading | loaded(data) | failed(error)" as a sum instead of several nullable fields.
- **Expression problem:** sums make adding functions easy and adding variants hard; objects/interfaces the reverse. Tagless final / visitors / open unions trade between them.

### Recursive types as fixed points

- Pattern functor for lists: `ListF A X = Nil | Cons A X`; `List A = Fix (ListF A)`.
- Separating the recursion from the shape lets one generic `cata` work for all types.

### F-algebras and catamorphisms

- An F-algebra is a carrier $B$ and `alg : F B → B` (e.g. `Nil ↦ 0`, `Cons a b ↦ a + b` gives sum).
- **Catamorphism:** `cata alg = alg ∘ map(cata alg) ∘ unfix`. Initiality: for each algebra there is exactly one structure-preserving map out of the initial algebra, which is why folds are well defined and fusion laws hold.
- **Fusion law:** `h ∘ cata f = cata g` whenever `h ∘ f = g ∘ map h`: the basis of deforestation optimisations (removing intermediate lists).

### Unfolds and hylomorphisms

- **Anamorphism:** `ana coalg = fix ∘ map(ana coalg) ∘ coalg` with `coalg : B → F B` (e.g. range generation).
- **Hylomorphism:** `hylo alg coalg = alg ∘ map(hylo alg coalg) ∘ coalg`: recursion that splits a problem (coalgebra) and combines results (algebra), without materialising the call tree as data. Mergesort: split lists (coalgebra into a tree functor), merge (algebra).

### Data vs codata

- Data: finite, built by constructors, consumed by structural recursion (terminating folds).
- Codata: defined by observations (head/tail of a stream), may be infinite, produced by corecursion (productive unfolds). Total languages (Agda, Idris) distinguish them to guarantee termination/productivity.

### Type arithmetic and zippers

- $\lvert A + B\rvert = \lvert A\rvert + \lvert B\rvert$, $\lvert A\times B\rvert = \lvert A\rvert\lvert B\rvert$, $\lvert A \to B\rvert = \lvert B\rvert^{\lvert A\rvert}$.
- The derivative of a type's pattern functor gives its one-hole context (zipper): efficient local edits in immutable structures.

## Other lenses

- **Category theory.** Initial algebras / final coalgebras, (co)products and exponentials are universal constructions (see [category theory](/ref/programs/category-theory-core)).
- **Algebra.** Folds into monoids (`foldMap`) parallelise because of associativity (see [algebraic structures](/ref/programs/algebraic-structures)).
- **Compilers.** ASTs are ADTs; interpreters, type checkers and optimisation passes are folds/rewrites over them (see [term rewriting](/ref/programs/term-rewriting-egraphs)).
- **Dynamics.** Codata and coalgebras model state machines and processes: a coalgebra $S \to F S$ is a transition system (automata, Mealy machines) (see [parsing & automata](/ref/programs/parsing-automata)).

## Pitfalls & diagnostics

- **Non-exhaustive matches** with wildcards hide missing cases when variants are added; avoid catch-alls on evolving types.
- **Stack depth:** naive structural recursion on long lists/deep trees overflows; use tail recursion, accumulators or iterative folds.
- **Laziness and space leaks** (Haskell) with lazy folds; use strict folds for accumulations.
- **Over-engineering:** recursion schemes clarify complex transformations; for simple functions, direct recursion is clearer.

## See also

- [Category theory core](/ref/programs/category-theory-core)
- [Monads & effects](/ref/programs/monads-effects)
- [Algebraic structures](/ref/programs/algebraic-structures)
- [Lambda calculus & types](/ref/programs/lambda-calculus-types)

**References:**

- Meijer, Fokkinga & Paterson, "Functional programming with bananas, lenses, envelopes and barbed wire", FPCA (1991).
- Bird & de Moor, _Algebra of Programming_.
- McBride, "The derivative of a regular type is its type of one-hole contexts" (2001).
