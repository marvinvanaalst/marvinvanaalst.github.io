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
updated: '2026-11-15'
published: true
layout: ref
---

## TL;DR

A **category** is objects plus composable arrows. A **functor** maps one category into another and preserves identities and composition. A **natural transformation** is a map between two functors that is uniform in the object. **Universal properties** define constructions (products, sums, function types) by how they map, up to unique isomorphism. **Adjunctions** are pairs of functors that are "best approximations" of each other, and every adjunction produces a monad. In programming: types are objects, functions are arrows, generic containers are functors, and polymorphic functions are natural transformations (for free, by parametricity).

## Decision table

| Concept                 | One-line definition                                              | Programming reading                             | Laws to check                                                    |
| ----------------------- | ---------------------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| Category                | objects, arrows $f: A \to B$, $\circ$, $\mathrm{id}$             | types and pure functions                        | associativity, identity                                          |
| Functor $F$             | maps objects and arrows, $F f : F A \to F B$                     | container with `map` (`Vector`, `Maybe`, trees) | $F\,\mathrm{id} = \mathrm{id}$, $F(g \circ f) = F g \circ F f$   |
| Natural transformation  | $\eta_A : F A \to G A$ for every $A$, commuting with $F f$/$G f$ | polymorphic function `F{T} → G{T}`              | naturality square                                                |
| Product $A \times B$    | projections; unique pairing $\langle f, g \rangle$               | tuple / record                                  | $\pi_1 \circ \langle f, g\rangle = f$                            |
| Coproduct $A + B$       | injections; unique case analysis $[f, g]$                        | tagged union / enum / `Either`                  | $[f, g] \circ \iota_1 = f$                                       |
| Terminal / initial      | exactly one arrow into / out of every object                     | unit type / empty type (`Never`, `Void`)        | uniqueness                                                       |
| Exponential $B^A$       | object of arrows, with $\mathrm{eval}$ and $\mathrm{curry}$      | function type `A → B`                           | $\mathrm{eval} \circ (\mathrm{curry}\,f \times \mathrm{id}) = f$ |
| Adjunction $F \dashv G$ | $\mathrm{Hom}(F A, B) \cong \mathrm{Hom}(A, G B)$, natural       | currying, free/forgetful, Galois connections    | triangle identities                                              |

## Variants

### Categories

Associativity $h \circ (g \circ f) = (h \circ g) \circ f$ and identities $\mathrm{id} \circ f = f = f \circ \mathrm{id}$. Examples worth keeping in mind:

- **Set / Types:** sets (types) and total functions.
- **A preorder:** one arrow $a \to b$ iff $a \le b$. Composition is transitivity, identity is reflexivity.
- **A monoid:** one object, with the arrows being the elements and composition being the operation. `String` under concatenation, matrices under multiplication.
- **Vect / Mat:** vector spaces and linear maps, or concretely natural numbers as objects with $m \times n$ matrices as arrows.
- **Kleisli category** of a monad: arrows $A \to M B$ composed with `bind`. This is how effectful functions compose (see monads & effects).

Opposite category $\mathcal C^{\mathrm{op}}$: same objects, arrows reversed. Every definition then has a **dual** for free.

### Functors

```julia
xs = [1, 2, 3]; f(x) = x + 1; g(x) = 2x
@assert map(identity, xs) == xs                      # identity law
@assert map(g ∘ f, xs) == map(g, map(f, xs))         # composition law
```

The second law is **loop fusion**: two `map` passes equal one. Compilers may rewrite with it only if `map` really is a functor. **Contravariant** functors flip arrows ($F f : F B \to F A$). The function type $A \to R$ is contravariant in $A$ and covariant in $R$, which is where subtyping variance rules come from. Bifunctors: products, sums and `Dict{K,V}`-like types, mapping over both arguments.

### Natural transformations

A family $\eta_A : F A \to G A$ such that, for every $f : A \to B$, the square commutes:

$$
\begin{CD}
F A @>{\eta_A}>> G A \\
@V{F f}VV @VV{G f}V \\
F B @>{\eta_B}>> G B
\end{CD}
\qquad\quad G f \circ \eta_A = \eta_B \circ F f
$$

```julia
safe_first(xs) = isempty(xs) ? nothing : xs[1]      # Vector{T} → Union{Nothing,T}
fmap(f, ::Nothing) = nothing; fmap(f, x) = f(x)
@assert safe_first(map(f, xs)) == fmap(f, safe_first(xs))
```

**Parametricity** ("theorems for free"): any function polymorphic in `T` that doesn't inspect `T` is automatically natural. `reverse`, `first` and `flatten` all commute with `map`. This is the license for rewriting `map(f) ∘ reverse` to `reverse ∘ map(f)`.

### Universal properties

A universal property defines an object by a unique-arrow condition. **Product:** for any $f : X \to A$ and $g : X \to B$ there is a unique $\langle f, g\rangle : X \to A \times B$ with $\pi_1 \circ \langle f, g\rangle = f$ and $\pi_2 \circ \langle f, g\rangle = g$. The **coproduct** is the dual: a unique $[f, g] : A + B \to X$, which is exactly `match`/`case` on a tagged union. **Exponential:**

$$
\mathrm{Hom}(A \times B,\; C) \;\cong\; \mathrm{Hom}(A,\; C^B) \qquad (\text{curry} \leftrightarrow \text{uncurry})
$$

A category with finite products and exponentials is **cartesian closed** (CCC). CCCs are the models of the simply typed lambda calculus. The algebra of types follows: $A \times (B + C) \cong A\times B + A\times C$, $C^{A+B} \cong C^A \times C^B$ (a function on a sum is a pair of functions), and $1 + A$ is `Maybe`.

Objects defined by a universal property are unique up to **unique** isomorphism. So "the" product is well defined, and implementations (struct vs tuple) are interchangeable.

### Adjunctions

$F : \mathcal C \to \mathcal D$ is left adjoint to $G : \mathcal D \to \mathcal C$ ($F \dashv G$) when

$$
\mathrm{Hom}_{\mathcal D}(F A, B) \;\cong\; \mathrm{Hom}_{\mathcal C}(A, G B) \quad \text{naturally in } A, B.
$$

| Left $F$             | Right $G$                  | Reading                                                                              |
| -------------------- | -------------------------- | ------------------------------------------------------------------------------------ |
| $- \times S$         | $S \to -$                  | currying. $G \circ F$ = $S \to (- \times S)$ = **state monad**                       |
| free monoid (`List`) | forgetful (underlying set) | a monoid homomorphism out of `List{A}` is just a function out of `A` (= `foldMap`)   |
| abstraction $\alpha$ | concretisation $\gamma$    | Galois connection in abstract interpretation: $\alpha(c) \le a \iff c \le \gamma(a)$ |
| $\exists$            | weakening                  | logic: quantifiers as adjoints                                                       |

Left adjoints preserve colimits (sums, initial objects) and right adjoints preserve limits (products, terminal objects). This is a quick sanity check when guessing whether something is a left or right adjoint. Every adjunction gives a monad $G \circ F$ and a comonad $F \circ G$.

## Other lenses

- **Linear algebra.** Transpose is the original "adjoint": $\langle Ax, y\rangle = \langle x, A^\top y\rangle$ has the shape of the Hom-isomorphism. Matrices form a category, $\otimes$ makes it monoidal, and string diagrams for tensor networks are the graphical calculus of monoidal categories.
- **Order theory.** In a preorder, products are meets ($\wedge$), coproducts are joins ($\vee$), and adjunctions are Galois connections. Most "category theory for static analysis" lives at this level (see lattices & fixpoints).
- **Logic (Curry–Howard–Lambek).** Products are $\wedge$, coproducts $\vee$, exponentials $\Rightarrow$, terminal $\top$, initial $\bot$. Proofs are programs are arrows in a CCC.
- **Automatic differentiation.** Forward mode is a functor sending $f$ to $(f, Df)$, and the chain rule is the composition law. Reverse mode uses the contravariant (transposed) derivative.

## Pitfalls & diagnostics

- Laws are not enforced by most languages. Property-test functor and monad laws for custom types, or rewrites like fusion become unsound.
- "Functor" is overloaded: function object in C++, parameterised module in ML. Neither is the categorical notion.
- Real languages are only approximately categories: non-termination, exceptions, mutation and `seq` break laws at the edges. Reason in the idealised category, then check the edges.
- Variance is direction: arguments are contravariant and results covariant. Getting it wrong is a classic type-system soundness bug (mutable arrays must be invariant).
- Don't climb to the abstraction until a concrete pattern repeats. The payoff is laws you can rely on, not vocabulary.

## See also

Monads & effects · Algebraic data types & recursion schemes · Algebraic structures (monoids, semirings) · Lambda calculus & types · Lattices, fixpoints & dataflow · Automatic differentiation

**References:** Milewski, _Category Theory for Programmers_. Leinster, _Basic Category Theory_. Pierce, _Basic Category Theory for Computer Scientists_. Wadler, "Theorems for free!" (1989).
