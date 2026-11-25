---
title: 'Category theory: the core vocabulary'
description: 'Categories, functors, natural transformations, universal properties and adjunctions, read as types, type constructors and polymorphic functions.'
aliases:
  [
    category,
    functor,
    natural transformation,
    parametricity,
    product,
    coproduct,
    sum type,
    exponential,
    currying,
    universal property,
    adjunction,
    Galois connection,
    duality,
    CCC
  ]
tags: [category theory, type theory, programming languages]
order: 1
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Category:* objects plus composable arrows.
- *Functor:* maps one category into another, preserving identities and composition.
- *Natural transformation:* a map between two functors that is uniform in the object.
- *Universal property:* defines a construction (product, sum, function type) by how it maps, up to unique isomorphism.
- *Adjunction:* a pair of functors that are "best approximations" of each other. Every adjunction produces a monad.

In programming:

- types are objects, functions are arrows
- generic containers are functors
- polymorphic functions are natural transformations (for free, by parametricity)

## Decision table

| Concept | One-line definition | Programming reading | Laws to check |
| --- | --- | --- | --- |
| Category | objects, arrows $`f: A \to B`$, $`\circ`$, $`\mathrm{id}`$ | types and pure functions | associativity, identity |
| Functor $`F`$ | maps objects and arrows, $`F f : F A \to F B`$ | container with `map` (`Vector`, `Maybe`, trees) | $`F\,\mathrm{id} = \mathrm{id}`$, $`F(g \circ f) = F g \circ F f`$ |
| Natural transformation | $`\eta_A : F A \to G A`$ for every $`A`$, commuting with $`F f`$/$`G f`$ | polymorphic function `F{T} → G{T}` | naturality square |
| Product $`A \times B`$ | projections; unique pairing $`\langle f, g \rangle`$ | tuple / record | $`\pi_1 \circ \langle f, g\rangle = f`$ |
| Coproduct $`A + B`$ | injections; unique case analysis $`[f, g]`$ | tagged union / enum / `Either` | $`[f, g] \circ \iota_1 = f`$ |
| Terminal / initial | exactly one arrow into / out of every object | unit type / empty type (`Never`, `Void`) | uniqueness |
| Exponential $`B^A`$ | object of arrows, with $`\mathrm{eval}`$ and $`\mathrm{curry}`$ | function type `A → B` | $`\mathrm{eval} \circ (\mathrm{curry}\,f \times \mathrm{id}) = f`$ |
| Adjunction $`F \dashv G`$ | $`\mathrm{Hom}(F A, B) \cong \mathrm{Hom}(A, G B)`$, natural | currying, free/forgetful, Galois connections | triangle identities |

## Variants

### Categories

- *Laws:*
  - associativity $`h \circ (g \circ f) = (h \circ g) \circ f`$
  - identities $`\mathrm{id} \circ f = f = f \circ \mathrm{id}`$
- *Examples worth keeping in mind:*
  - *Set / Types:* sets (types) and total functions.
  - *Preorder:* one arrow $`a \to b`$ iff $`a \le b`$. Composition is transitivity, identity is reflexivity.
  - *Monoid:* one object, the arrows are the elements, composition is the operation. `String` under concatenation, matrices under multiplication.
  - *Vect / Mat:* vector spaces and linear maps, or concretely natural numbers as objects with $`m \times n`$ matrices as arrows.
  - *Kleisli category* of a monad: arrows $`A \to M B`$ composed with `bind`. This is how effectful functions compose (see [monads & effects](/ref/programs/monads-effects)).
- *Opposite category* $`\mathcal C^{\mathrm{op}}`$: same objects, arrows reversed. Every definition then has a *dual* for free.

### Functors

- *Laws:*
  - identity: `map(identity, xs) == xs`
  - composition: `map(g ∘ f, xs) == map(g, map(f, xs))`
- *The composition law is loop fusion:* two `map` passes equal one. Compilers may rewrite with it only if `map` really is a functor.
- *Contravariant functors* flip arrows: $`F f : F B \to F A`$.
  - The function type $`A \to R`$ is contravariant in $`A`$ and covariant in $`R`$.
  - This is where subtyping variance rules come from.
- *Bifunctors* map over two arguments: products, sums, `Dict{K,V}`-like types.

### Natural transformations

A family $`\eta_A : F A \to G A`$ such that, for every $`f : A \to B`$, the square commutes:

```math
\begin{CD}
F A @>{\eta_A}>> G A \\
@V{F f}VV @VV{G f}V \\
F B @>{\eta_B}>> G B
\end{CD}
\qquad\quad G f \circ \eta_A = \eta_B \circ F f
```

- *Example:* `safe_first`, which returns the first element or `nothing`, maps `Vector{T}` to `Union{Nothing,T}`. Naturality: taking the first element and then applying $`f`$ is the same as mapping $`f`$ over the vector and then taking the first element.
- *Parametricity* ("theorems for free"):
  - Any function polymorphic in `T` that doesn't inspect `T` is automatically natural.
  - `reverse`, `first` and `flatten` all commute with `map`.
  - This is the license for rewriting `map(f) ∘ reverse` to `reverse ∘ map(f)`.

### Universal properties

A universal property defines an object by a unique-arrow condition. Such objects are unique up to *unique* isomorphism, so "the" product is well defined and implementations (struct vs tuple) are interchangeable.

- *Product:* for any $`f : X \to A`$ and $`g : X \to B`$ there is a unique $`\langle f, g\rangle : X \to A \times B`$ with $`\pi_1 \circ \langle f, g\rangle = f`$ and $`\pi_2 \circ \langle f, g\rangle = g`$.
- *Coproduct* (the dual): a unique $`[f, g] : A + B \to X`$. This is exactly `match`/`case` on a tagged union.
- *Exponential:* currying as a natural isomorphism,

```math
\mathrm{Hom}(A \times B,\; C) \;\cong\; \mathrm{Hom}(A,\; C^B) \qquad (\text{curry} \leftrightarrow \text{uncurry})
```

- *Cartesian closed category (CCC):* finite products and exponentials.
  - CCCs are the models of the simply typed lambda calculus.
- *Algebra of types* follows from the universal properties:
  - $`A \times (B + C) \cong A\times B + A\times C`$
  - $`C^{A+B} \cong C^A \times C^B`$ (a function on a sum is a pair of functions)
  - $`1 + A`$ is `Maybe`

### Adjunctions

$`F : \mathcal C \to \mathcal D`$ is left adjoint to $`G : \mathcal D \to \mathcal C`$ ($`F \dashv G`$) when

```math
\mathrm{Hom}_{\mathcal D}(F A, B) \;\cong\; \mathrm{Hom}_{\mathcal C}(A, G B) \quad \text{naturally in } A, B.
```

| Left $`F`$ | Right $`G`$ | Reading |
| --- | --- | --- |
| $`- \times S`$ | $`S \to -`$ | currying. $`G \circ F`$ = $`S \to (- \times S)`$ = *state monad* |
| free monoid (`List`) | forgetful (underlying set) | a monoid homomorphism out of `List{A}` is just a function out of `A` (= `foldMap`) |
| abstraction $`\alpha`$ | concretisation $`\gamma`$ | Galois connection in abstract interpretation: $`\alpha(c) \le a \iff c \le \gamma(a)`$ |
| $`\exists`$ | weakening | logic: quantifiers as adjoints |

- *Sanity check* when guessing which side something is on:
  - left adjoints preserve colimits (sums, initial objects)
  - right adjoints preserve limits (products, terminal objects)
- *Monads:* every adjunction gives a monad $`G \circ F`$ and a comonad $`F \circ G`$.

## Other lenses

- *Linear algebra.*
  - Transpose is the original "adjoint": $`\langle Ax, y\rangle = \langle x, A^\top y\rangle`$ has the shape of the Hom-isomorphism.
  - Matrices form a category, and $`\otimes`$ makes it monoidal. String diagrams for tensor networks are the graphical calculus of monoidal categories.
- *Order theory.*
  - In a preorder, products are meets ($`\wedge`$), coproducts are joins ($`\vee`$), and adjunctions are Galois connections.
  - Most "category theory for static analysis" lives at this level (see [lattices & fixpoints](/ref/programs/lattices-fixpoints-dataflow)).
- *Logic (Curry–Howard–Lambek).*
  - Products are $`\wedge`$, coproducts $`\vee`$, exponentials $`\Rightarrow`$, terminal $`\top`$, initial $`\bot`$.
  - Proofs are programs are arrows in a CCC.
- *Automatic differentiation.*
  - Forward mode is a functor sending $`f`$ to $`(f, Df)`$, and the chain rule is the composition law.
  - Reverse mode uses the contravariant (transposed) derivative.

## Pitfalls & diagnostics

- *Laws are not enforced* by most languages. Property-test functor and monad laws for custom types, or rewrites like fusion become unsound.
- *"Functor" is overloaded:* function object in C++, parameterised module in ML. Neither is the categorical notion.
- *Real languages are only approximately categories:* non-termination, exceptions, mutation and `seq` break laws at the edges. Reason in the idealised category, then check the edges.
- *Variance is direction:* arguments are contravariant, results covariant. Getting it wrong is a classic type-system soundness bug (mutable arrays must be invariant).
- *Don't climb to the abstraction too early.* Wait until a concrete pattern repeats. The payoff is laws you can rely on, not vocabulary.

## See also

- [Monads & effects](/ref/programs/monads-effects)
- [Algebraic data types & recursion schemes](/ref/programs/algebraic-data-types)
- [Algebraic structures (monoids, semirings)](/ref/programs/algebraic-structures)
- [Lambda calculus & types](/ref/programs/lambda-calculus-types)
- [Lattices, fixpoints & dataflow](/ref/programs/lattices-fixpoints-dataflow)
- [Automatic differentiation](/ref/diff/automatic-differentiation)

*References:*

- Milewski, _Category Theory for Programmers_.
- Leinster, _Basic Category Theory_.
- Pierce, _Basic Category Theory for Computer Scientists_.
- Wadler, "Theorems for free!" (1989).
