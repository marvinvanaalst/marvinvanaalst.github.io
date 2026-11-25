---
title: 'Staging & partial evaluation'
description: 'Multi-stage programming, partial evaluation and binding-time analysis, the Futamura projections, specialisation in JITs, and supercompilation: generating fast code from generic code.'
aliases:
  [
    staging,
    multi-stage programming,
    MSP,
    partial evaluation,
    specialization,
    binding time analysis,
    online partial evaluation,
    offline partial evaluation,
    Futamura projections,
    quasi-quotation,
    macros,
    code generation,
    LMS,
    lightweight modular staging,
    MetaOCaml,
    tracing JIT,
    type specialization,
    supercompilation,
    deforestation,
    generic programming,
    template metaprogramming,
    Julia specialization
  ]
tags: [programming languages, compilers, metaprogramming, performance]
order: 12
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Partial evaluation:* given a program and _some_ of its inputs (static), produce a specialised residual program for the rest (dynamic). Removes interpretive overhead: loops unrolled, branches folded, generic code made concrete.
- *Multi-stage programming:* the programmer marks what is computed now vs generated for later (quotes and splices, staged types); explicit, predictable, type-safe code generation (MetaOCaml, LMS in Scala, Terra).
- *Futamura projections:* specialising an interpreter to a program gives a compiled program; specialising the specialiser to the interpreter gives a compiler; once more gives a compiler generator.
- *In practice:* JIT type specialisation (Julia, PyPy, Truffle/Graal), template metaprogramming and generics monomorphisation, tracing JITs, and the "model → generated simulation code" pipelines of scientific software.
- *Supercompilation* drives programs symbolically, folding repeated configurations, to perform deforestation and specialisation together.

## Decision table

| Technique | Who decides what is static | Guarantees | Use when |
| --- | --- | --- | --- |
| Online partial evaluation | specialiser, during evaluation | may not terminate, precise | small languages, research tools |
| Offline PE (binding-time analysis) | analysis before specialisation | predictable termination (with care) | systematic specialisation (C-Mix, Similix) |
| Multi-stage programming (quotes) | programmer | well-typed generated code (MetaOCaml) | DSLs, high-performance kernels, code generators |
| Macros (hygienic) | programmer (syntax level) | hygiene, no types | syntax extension, compile-time code generation |
| Templates / generics monomorphisation | compiler (types) | type safety | zero-cost abstractions (C++, Rust) |
| JIT type specialisation | runtime types | correct with deoptimisation | dynamic languages (Julia, PyPy, JS engines) |
| Truffle / self-specialising interpreters | interpreter + PE of AST | first Futamura projection in production | language implementations on GraalVM |
| Supercompilation | driving + folding | may not terminate without whistles | deforestation, research |
| String-template code generation | programmer | none | quick hacks (prefer the above) |

## Variants

### Partial evaluation

- *Example:* `power(x, n)` with static `n = 3` specialises to `x * x * x`.
- *Binding-time analysis:* classify each expression as static or dynamic (a dataflow analysis); static parts are evaluated, dynamic parts emitted as residual code.
- *Termination:* unrolling over a static loop is fine; over a dynamic one it must stop (generalise static values to dynamic).
- *Polyvariance:* generate several specialised versions of a function for different static arguments (memoise by static inputs).

### Multi-stage programming

- Quote `.<e>.` builds code, splice `.~e` inserts code into code, `run` compiles and executes it (MetaOCaml syntax).
- Type systems ensure generated code is well typed and well scoped (no unbound variables).
- *Lightweight modular staging:* types distinguish staged values (`Rep[Double]`) from present-stage ones; operations on staged values build IR graphs that are optimised and compiled.
- Typical uses: specialised numerical kernels (FFTs for fixed sizes, ODE right-hand sides from symbolic models), query compilers, parser generators.

### Futamura projections

1. $`\text{spec}(\text{interp}, \text{prog})`$ = compiled program (target code doing what `prog` does).
2. $`\text{spec}(\text{spec}, \text{interp})`$ = compiler.
3. $`\text{spec}(\text{spec}, \text{spec})`$ = compiler generator.

- Production version of (1): Truffle partially evaluates AST interpreters with profiling information, yielding JIT compilers for each language.

### JIT specialisation

- *Method JITs with type specialisation* (Julia): compile a method per concrete argument-type tuple; type inference propagates concrete types; dynamic dispatch disappears where types are known.
- *Tracing JITs* (LuaJIT, PyPy): record a hot loop's actual path, specialise to observed types/values, guard assumptions, deoptimise on failure.
- Key design concern: type stability / predictable specialisation (avoid code paths where types can't be inferred).

### Supercompilation

- Symbolically execute (drive) the program, building a process tree; fold when a configuration repeats (up to renaming); generalise when a "whistle" (homeomorphic embedding) signals possible non-termination.
- Achieves deforestation (removing intermediate data structures), specialisation and some theorem proving.

## Other lenses

- *Lambda calculus.* Partial evaluation is normalisation of the static part; staging is two-level λ-calculus with explicit binding times (see [lambda calculus](/ref/programs/lambda-calculus-types)).
- *Scientific computing.* Symbolic modelling systems (ModelingToolkit, CasADi, SymPy codegen) stage models: build symbolic equations once, simplify, generate specialised (and differentiated) code (see [automatic differentiation](/ref/diff/automatic-differentiation)).
- *Compilers.* Inlining + constant propagation is partial evaluation inside a compiler; specialisation interacts with SSA optimisations (see [SSA & IRs](/ref/programs/ssa-ir)).

## Pitfalls & diagnostics

- *Code explosion:* unrolling large static loops or too many specialisations; bound unrolling, memoise specialisations, cap polyvariance.
- *Compilation latency* (time-to-first-run in JITs, long compile times for template-heavy code); cache compiled code, precompile common specialisations.
- *Non-termination* of specialisers on dynamic recursion without generalisation.
- *Debuggability:* generated code is hard to inspect; keep pretty-printers and source maps for staged code.
- *Type instability* in JIT languages silently disables specialisation; use introspection tools (`@code_warntype` in Julia).

## See also

- [Lambda calculus & types](/ref/programs/lambda-calculus-types)
- [SSA & intermediate representations](/ref/programs/ssa-ir)
- [Term rewriting & e-graphs](/ref/programs/term-rewriting-egraphs)
- [Polyhedral loop transforms](/ref/programs/polyhedral-loop-transforms)

*References:*

- Jones, Gomard & Sestoft, _Partial Evaluation and Automatic Program Generation_ (free online).
- Taha, "A gentle introduction to multi-stage programming" (2004).
- Würthinger et al., "Practical partial evaluation for high-performance dynamic language runtimes", PLDI (2017).
