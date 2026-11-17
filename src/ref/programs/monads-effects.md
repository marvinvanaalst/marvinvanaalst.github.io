---
title: 'Monads, applicatives & effects'
description: 'Functor, applicative and monad hierarchies, Kleisli composition, common monads, free monads, monad transformers, and algebraic effects with handlers.'
aliases:
  [
    monad,
    applicative,
    functor,
    Kleisli,
    bind,
    return,
    pure,
    monad laws,
    Maybe,
    Option,
    Either,
    Result,
    List monad,
    State monad,
    Reader,
    Writer,
    IO,
    continuation monad,
    monad transformers,
    free monad,
    algebraic effects,
    effect handlers,
    delimited continuations,
    do-notation,
    async await,
    effect systems
  ]
tags: [category theory, programming languages, effects]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- A **monad** is a type constructor $M$ with `pure : A → M A` and `bind : M A → (A → M B) → M B`, satisfying identity and associativity laws. It sequences computations with an effect (failure, state, nondeterminism, I/O, async).
- **Applicatives** combine independent effectful computations (parallelism, static analysis); every monad is applicative, not vice versa. Use the weakest structure that suffices.
- **Kleisli composition** `(f >=> g)` composes effectful functions $A \to M B$, $B \to M C$; the monad laws are exactly the category laws of the Kleisli category.
- **Combining effects:** monad transformer stacks (compose effects in a fixed order) or **algebraic effects + handlers** (declare operations, interpret them separately; composable, the modern language-design choice).
- `async/await`, `?` on Result, list comprehensions, generators and exceptions are all monadic or effectful patterns with special syntax.

## Decision table

| Abstraction                  | Signature (core)                     | Laws                                             | Enables                                                             |
| ---------------------------- | ------------------------------------ | ------------------------------------------------ | ------------------------------------------------------------------- |
| Functor                      | `map : (A → B) → F A → F B`          | identity, composition                            | transform inside a context                                          |
| Applicative                  | `pure`, `ap : F (A → B) → F A → F B` | identity, composition, homomorphism, interchange | combine independent effects, parallelism, static structure          |
| Monad                        | `pure`, `bind`                       | left/right identity, associativity               | dependent sequencing (next step depends on result)                  |
| Monad transformer            | `T M A`                              | lift laws                                        | stack effects in a fixed order                                      |
| Free monad                   | data structure of operations         | free (no extra laws)                             | separate syntax from interpretation                                 |
| Algebraic effects + handlers | `perform op`, `handle e with h`      | handler equations                                | modular, composable effects; resumable control                      |
| Comonad                      | `extract`, `extend`                  | dual laws                                        | context-dependent computation (streams, zippers, cellular automata) |

| Monad           | Effect                       | `bind` behaviour                                                                                       |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| Maybe / Option  | failure without info         | short-circuit on `Nothing`                                                                             |
| Either / Result | failure with error           | short-circuit on `Left`/`Err`                                                                          |
| List            | nondeterminism               | flatMap over all choices                                                                               |
| State $S$       | mutable state                | thread $S \to (A, S)$                                                                                  |
| Reader $R$      | read-only environment        | pass $R$ to every step                                                                                 |
| Writer $W$      | logging / accumulation       | append to a monoid $W$                                                                                 |
| Continuation    | control flow (callcc, async) | CPS composition                                                                                        |
| IO / Task       | external effects / async     | sequence actions                                                                                       |
| Probability     | random choice                | weighted sums over outcomes (see [probabilistic programming](/ref/programs/probabilistic-programming)) |

## Variants

### Monad laws

- Left identity: `bind(pure(a), f) == f(a)`.
- Right identity: `bind(m, pure) == m`.
- Associativity: `bind(bind(m, f), g) == bind(m, x -> bind(f(x), g))`.
- In Kleisli form: `pure >=> f == f`, `f >=> pure == f`, `(f >=> g) >=> h == f >=> (g >=> h)`.
- **join** formulation: `join : M (M A) → M A` with `bind(m, f) = join(map(f, m))`; categorically, a monoid in the category of endofunctors.

### Applicatives vs monads

- Applicative: the structure of the computation is fixed before running (all effects known up front) → can be analysed, parallelised, batched (e.g. validation collecting all errors, parsers that can be optimised, build systems).
- Monad: later effects can depend on earlier results → more expressive, less analysable.

### Free monads

- Represent a program as a tree of operations (`Free F A`), interpret later with different handlers (production, test, logging).
- Costs: allocation and interpretation overhead; codensity / freer encodings improve performance.

### Algebraic effects and handlers

- Declare effect operations (`get`, `put`, `raise`, `yield`, `choose`); code performs them without knowing how they're implemented.
- A **handler** interprets operations, receiving the **continuation** of the computation: it can resume once (state, I/O), never (exceptions), or many times (nondeterminism, probabilistic inference).
- Compose by nesting handlers; order of handlers determines semantics (e.g. state + exceptions: rollback or not).
- Implemented via delimited continuations; in OCaml 5, Koka, Eff, Effekt, Unison; effect systems track which effects a function may perform in its type.

### Monad transformers

- `StateT S (Either E)` etc.: each transformer adds one effect; `lift` moves operations through layers.
- Order matters and stacks get unwieldy (n² lifting instances); one motivation for effect handlers.

## Other lenses

- **Category theory.** Monads arise from adjunctions (state from currying); Kleisli categories make effectful functions composable; applicatives are lax monoidal functors (see [category theory](/ref/programs/category-theory-core)).
- **Compilers.** CPS transformation makes control explicit (monadic style); SSA and CPS are closely related (see [SSA & IRs](/ref/programs/ssa-ir)).
- **Probability.** The distribution monad underlies probabilistic programming; inference algorithms are effect handlers for `sample` and `observe` (see [probabilistic programming](/ref/programs/probabilistic-programming)).
- **AD.** Reverse-mode AD can be implemented with effect handlers or continuations (the backward pass as the continuation).

## Pitfalls & diagnostics

- **Lawless instances** break refactorings that assume the laws (e.g. reassociating binds); property-test them.
- **Over-abstraction:** a monad stack where a function argument or plain exception would do.
- **Performance:** naive free monads and deep transformer stacks allocate heavily; prefer effect handlers or specialised implementations.
- **Effect order semantics:** be explicit about how effects interact (state rollback on exception, etc.).

## See also

- [Category theory core](/ref/programs/category-theory-core)
- [Algebraic data types & recursion schemes](/ref/programs/algebraic-data-types)
- [Probabilistic programming](/ref/programs/probabilistic-programming)
- [Lambda calculus & types](/ref/programs/lambda-calculus-types)

**References:**

- Wadler, "Monads for functional programming" (1995).
- McBride & Paterson, "Applicative programming with effects", J. Funct. Program. (2008).
- Plotkin & Pretnar, "Handling algebraic effects", LMCS (2013).
