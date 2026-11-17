---
title: 'Lambda calculus & type systems'
description: 'Untyped and simply typed lambda calculus, evaluation strategies, System F and parametric polymorphism, subtyping and variance, Curry–Howard, and dependent types in brief.'
aliases:
  [
    lambda calculus,
    beta reduction,
    alpha conversion,
    eta conversion,
    Church encoding,
    Y combinator,
    evaluation strategy,
    call by value,
    call by name,
    call by need,
    normal form,
    confluence,
    Church-Rosser,
    simply typed lambda calculus,
    STLC,
    System F,
    parametric polymorphism,
    subtyping,
    variance,
    covariance,
    contravariance,
    Curry-Howard,
    propositions as types,
    dependent types,
    linear types,
    soundness,
    progress and preservation
  ]
tags: [type theory, programming languages, logic]
order: 5
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Untyped λ-calculus:** variables, abstraction $\lambda x.\,e$, application; computation = **β-reduction** $(\lambda x.\,e)\,v \to e[x := v]$. Turing complete (Church encodings, fixed-point combinators).
- **Evaluation strategy** is a language design choice: call-by-value (most languages), call-by-name / call-by-need (lazy, Haskell). Confluence (Church–Rosser) means normal forms are unique when they exist.
- **Simply typed λ-calculus (STLC):** types prevent stuck terms; every well-typed term terminates. **Soundness** = progress + preservation.
- **System F** adds type abstraction (generics); **parametricity** gives free theorems. Full type inference for System F is undecidable; Hindley–Milner restricts it to stay inferable.
- **Subtyping and variance:** function types are contravariant in arguments, covariant in results; mutable references are invariant.
- **Curry–Howard:** types are propositions, programs are proofs. Dependent types (types depending on values) make types into specifications.

## Decision table

| Calculus / feature                | Adds                                  | Key property                             | Found in                            |
| --------------------------------- | ------------------------------------- | ---------------------------------------- | ----------------------------------- |
| Untyped λ-calculus                | functions only                        | Turing complete, confluent               | foundations, Scheme core            |
| STLC                              | base types, $\to$                     | strong normalisation, decidable checking | core of typed functional languages  |
| STLC + fix                        | general recursion                     | non-termination allowed                  | real languages                      |
| Hindley–Milner (let-polymorphism) | polymorphic `let` with inference      | principal types, decidable inference     | ML, Haskell (core), OCaml, F#       |
| System F                          | $\forall$ over types                  | parametricity, undecidable inference     | Haskell with RankN, generics theory |
| System F$_\omega$                 | type operators (type-level functions) | higher-kinded types                      | Haskell, Scala                      |
| Subtyping (F$_{<:}$)              | $S <: T$ relation                     | variance, bounded quantification         | OO languages, TypeScript            |
| Dependent types (MLTT, CoC)       | types depending on terms              | specifications as types, proofs          | Agda, Coq/Rocq, Lean, Idris         |
| Linear / affine types             | use exactly / at most once            | resource safety, no aliasing             | Rust ownership, Linear Haskell      |
| Refinement types                  | predicates on types, SMT-checked      | lightweight verification                 | Liquid Haskell, F*                  |
| Gradual types                     | `?` dynamic type with casts           | mix static and dynamic                   | TypeScript, Python typing           |

## Variants

### Untyped λ-calculus

- Terms: $x$, $\lambda x.\,e$, $e_1\,e_2$. α-conversion (rename bound variables), β-reduction, η ($\lambda x.\,f\,x = f$).
- **Capture-avoiding substitution** is the subtle part; implementations use de Bruijn indices or locally nameless representations.
- **Church encodings:** booleans $\lambda t.\lambda f.\,t$; numerals $\lambda f.\lambda x.\,f^n x$. **Y combinator** $\lambda f.(\lambda x.f(xx))(\lambda x.f(xx))$ gives recursion (Z combinator for call-by-value).

### Evaluation strategies

- **Call-by-value:** evaluate arguments before the call; predictable cost; may diverge on unused arguments.
- **Call-by-name:** substitute unevaluated arguments; terminates whenever any strategy does (normal order).
- **Call-by-need:** call-by-name with sharing (lazy evaluation); efficient but space behaviour harder to predict.

### STLC and soundness

- Typing rules: variables from context; abstraction $\Gamma, x{:}A \vdash e : B \Rightarrow \Gamma\vdash\lambda x.\,e : A \to B$; application.
- **Progress:** a well-typed closed term is a value or can step. **Preservation:** stepping preserves types. Together: "well-typed programs don't go wrong".
- Strong normalisation: no general recursion; adding `fix` restores Turing completeness.

### Polymorphism

- **Parametric (System F):** $\Lambda\alpha.\,\lambda x{:}\alpha.\,x : \forall\alpha.\,\alpha\to\alpha$. Parametricity: a function of type $\forall\alpha.\,[\alpha]\to[\alpha]$ can only rearrange/duplicate/drop elements, so it commutes with `map`.
- **Ad hoc (type classes, traits, overloading):** behaviour depends on the type; dictionary passing or monomorphisation.
- **Implementation:** monomorphisation (C++, Rust; fast, code bloat) vs uniform representation/boxing (Java erasure, OCaml) vs runtime dictionaries (Haskell); Julia specialises on concrete types at JIT time.

### Subtyping and variance

- If $S <: T$: covariant constructors preserve it (`List S <: List T` for immutable lists), contravariant reverse it (`T → R <: S → R`), invariant require equality (mutable arrays/references).
- Unsound covariance of mutable arrays (Java) requires runtime checks.

### Curry–Howard and dependent types

- $A \to B$ ↔ implication, $A\times B$ ↔ conjunction, $A + B$ ↔ disjunction, empty type ↔ falsity, $\forall$ ↔ universal quantification (System F ↔ second-order logic).
- **Dependent types:** $\Pi$-types (functions whose result type depends on the argument, e.g. `Vec n → Vec n`) and $\Sigma$-types (dependent pairs). Type checking may require evaluation; proofs are programs checked by the type checker.

## Other lenses

- **Category theory.** STLC ↔ cartesian closed categories; linear logic ↔ symmetric monoidal closed categories (see [category theory](/ref/programs/category-theory-core)).
- **Logic.** Proof normalisation = program evaluation; cut elimination = β-reduction.
- **Compilers.** Closure conversion, CPS conversion and lambda lifting are λ-calculus transformations; IRs like ANF/CPS are restricted λ-calculi (see [SSA & IRs](/ref/programs/ssa-ir)).

## Pitfalls & diagnostics

- **Variable capture** bugs in substitution/inlining; use de Bruijn or fresh-name generation.
- **Unsound features in combination** (mutable references + polymorphism needs the value restriction; covariant mutable containers).
- **Undecidable inference:** adding rank-N types, subtyping or dependent types requires annotations (bidirectional typing, see [type inference](/ref/programs/type-inference-unification)).
- **Laziness surprises:** space leaks, unpredictable evaluation order of effects.

## See also

- [Type inference & unification](/ref/programs/type-inference-unification)
- [Category theory core](/ref/programs/category-theory-core)
- [Algebraic data types](/ref/programs/algebraic-data-types)
- [Staging & partial evaluation](/ref/programs/staging-partial-evaluation)

**References:**

- Pierce, _Types and Programming Languages_.
- Harper, _Practical Foundations for Programming Languages_ (2nd ed.).
- Wadler, "Propositions as types", CACM (2015).
