---
title: 'Lifting: One Scalar Function, Every Shape'
description: 'How languages apply scalar functions to arrays, why Julia moved shape out of the function, and a case for lifting whole regions instead of marking every operation'
categories:
  - musings
date: '2026-11-05'
author: 'Marvin van Aalst'
layout: tutorials
published: false
---

# Lifting: One Scalar Function, Every Shape

Take the most boring function you can write:

```rust
fn add(x: f64, y: f64) -> f64 { x + y }
```

Sooner or later someone wants to apply it to arrays. How a language answers that shapes its type system, its performance model and its syntax. This post walks through the design space, from NumPy's ufuncs through Julia's dots and Fortran's `elemental` procedures, and argues for one particular answer: a **lifted region**, a block in which scalar code is applied to every element. I'll write it as `lift { … }` in a Rust-flavoured pseudocode, since no language I know of ships exactly this design.

A companion post, [Composing execution](/blog/2026-11-composing-execution), covers how lifted code is then _executed_: in parallel, in blocks, in vector lanes, or on a GPU.

## The problem: shape inside the function

NumPy puts shape inside the function. `np.add` accepts any mix of floats and arrays, and returns a float only when every input is one. Writing that down with type hints takes an overload per combination:

```python
@overload
def add(x: float, y: float) -> float: ...
@overload
def add(x: float, y: NDArray) -> NDArray: ...
@overload
def add(x: NDArray, y: float) -> NDArray: ...
@overload
def add(x: NDArray, y: NDArray) -> NDArray: ...
```

Four overloads for two arguments, and still nothing about shapes. A user-defined scalar function has to be wrapped in `np.vectorize`, which is a Python loop in disguise, or compiled with Numba.

Fortran solved the user-function part in 1995 with `elemental` procedures. You write the function for scalars, and the compiler applies it to conforming arrays:

```fortran-free-form
elemental real(8) function gauss(x, mu, s)
    real(8), intent(in) :: x, mu, s
    gauss = exp(-(x - mu)**2 / (2 * s**2))
end function

g = gauss(xs, 0.5d0, 2.0d0)   ! xs is an array, so g is too
```

Chapel goes further with _promotion_: any procedure that takes scalars can be called with arrays, and the call becomes a parallel loop.

Julia moved shape **out** of the function and into the call. The function stays scalar, and the call _lifts_ it over its arguments' shape with a dot. That's the right idea, and the rest of this post keeps it. Lifting is not an execution detail: `add(x, y)` and its lifted form return different types. So the call site should choose it, visibly. Fortran and Chapel lift silently, based on the argument types, which is convenient but means a reader can't see which calls loop.

## From dots to regions

Julia marks lifting with dots:

```julia
z = exp.(.-(x .- mu).^2 ./ (2 * sigma^2))
```

I used to think this was the best possible design. Three things changed my mind:

1. **Two positions for one idea.** Operators take a prefix dot (`.+`, a MATLAB inheritance), calls take a suffix dot (`exp.(x)`).
2. **A marker on every operation.** Long formulas fill up with dots, and one missing dot changes what is computed, or allocates an intermediate array.
3. **`.` usually has other jobs.** In most languages it means method calls and field access. A language that wants `x.f(a)` and dotted lifting has to squeeze both into one character.

Julia also has an outer form, `@. z = exp(-(x - mu)^2 / (2sigma^2))`, which dots every operator and call in the expression. It reads much better, but it works on syntax alone: a whole-array call inside it, like `sum(x)`, is silently lifted too unless you escape it with `$sum(x)`.

A region keyword keeps what `@.` gets right and fixes its escape problem:

```rust
// pseudocode
let z = lift { exp(-(x - mu)^2 / (2.0 * sigma^2)) };
```

One keyword, in one position, and the formula reads like the formula.

## The basic cases

**Scalars stay scalars.** No region, no arrays:

```rust
let s = add(1.0, 2.0);                 // f64
```

**Two arrays of the same shape:**

```rust
let x: Array<f64, 1> = read_samples();  // 1000 elements
let y = Array::<f64, 1>::zeros(1000);
let z = lift { add(x, y) };              // Array<f64, 1>, 1000 elements
```

**A scalar and an array.** A scalar operand is used for every element:

```rust
let shifted = lift { add(2.0, x) };      // Array<f64, 1>
let scaled  = lift { 2.0 * x };          // operators lift the same way
let flipped = lift { -x };               // including unary minus
```

**Your own functions lift exactly like built-ins.** No `np.vectorize`:

```rust
fn gauss(x: f64, mu: f64, s: f64) -> f64 {
    exp(-(x - mu)^2 / (2.0 * s^2))
}

let g = lift { gauss(x, 0.5, 2.0) };     // Array<f64, 1>; mu and s are scalars
```

**Without a region, element-wise operations on arrays are errors, not silent allocations:**

```rust
let bad1 = add(x, y);      // error: no method `add(Array<f64, 1>, Array<f64, 1>)`
                           //   help: lift it over the elements: `lift { add(x, y) }`
let bad2 = 2.0 * x;        // error: `*` is scalar-only on arrays
                           //   help: element-wise: `lift { 2.0 * x }`; matrix product: `@`
let bad3 = exp(x);         // error: no method `exp(Array<f64, 1>)`
                           //   help: `lift { exp(x) }`
```

This strictness is a deliberate choice, and `bad2` is the controversial one. Julia, Fortran and Eigen's `Matrix` type all allow `a + b` and `2.0 * x` on arrays, because vector-space operations on same-shape arrays are unambiguous. I still prefer to require the region, for two reasons. First, uniformity: "element-wise means `lift`" is a rule with no exceptions to memorise. Second, fusion: `2.0 * x + y` written without a region would allocate a temporary for `2.0 * x`, while the lifted version is guaranteed to be one loop (see below). A language that wants the convenience can add vector-space operators later; removing them would break code.

**The braces are always there**, so the extent of a region, and of its loop, is never a question of precedence:

```rust
let loss = mean(lift { (pred - target)^2 });   // the region is exactly the braces
```

Like any block in an expression-oriented language, a region has a value, its last expression, which becomes the new array.

## Blocks: several statements, one loop

Statements in a region write into existing arrays, element by element:

```rust
lift { y = a * x + y }               // in place: one pass, no allocation
lift { row *= d }                    // compound assignment works too
```

**Every statement in a region runs in the same loop.** This is something per-operation dots can't express. A sigmoid and its derivative share `exp`:

```rust
lift {
    s  = 1.0 / (1.0 + exp(-v));
    ds = s * (1.0 - s);              // reads the element of s just written
}
```

That's one pass and one `exp` per element. With dots it is two statements, so two passes: either `exp` is computed twice, or `s` is read back from memory.

**`let` inside a region is a per-element local,** never materialised as an array:

```rust
lift {
    let d = x - mu;                  // a scalar per element, lives in a register
    z = exp(-d * d / c);
    w = d / sigma;
}
```

**Branches are per element,** and become selects or masks:

```rust
lift {
    y = if x > 0.0 { x } else { 0.01 * x }   // leaky ReLU
}
```

**Not allowed inside a region:** parallel constructs, awaiting, and `return`, `break` or `continue` that would leave it. A region is the body of a loop the compiler writes, and that loop has no exits.

For "one loop" to mean the same thing as "one statement after another, on whole arrays", a region needs three rules:

- **Writes must not alias reads.** "Reads the element of `s` just written" only matches whole-array semantics if `s` doesn't overlap `v`. In the example above, if `ds` were a view into `v`, the per-element loop and the statement-by-statement reading would disagree. The compiler proves non-aliasing statically where it can, and otherwise checks it once before the loop.
- **One iteration shape per region.** All statements run over the same index space. Arrays of different shapes go in different regions.
- **No neighbour access.** A region sees one element at a time. Stencils, which read `x[i-1]` and `x[i+1]`, are a different abstraction with boundary handling and halos. Halide is the place to look for how to do that properly.

## How calls inside a region are resolved

Inside a region, an operand of array type stands for its _element_. The question is what happens to calls that make sense on whole arrays, like `mean(x)` or `norm(v)`. There are two consistent answers.

**Type-directed resolution** looks at each call: if only a scalar method applies, it runs per element; if only a whole-array method applies, it runs once, before the loop; if both apply, it's a compile error. This is what a static type system makes possible and Julia's `@.` can't do. It's also what I first wanted. It has two problems that I think are fatal:

- **Matrix functions.** Any serious linear-algebra library defines `exp`, `log`, `sqrt`, `sin` and `cos` for square matrices. Under type-directed resolution, `lift { exp(a) }` is then ambiguous for every matrix `a`, which is exactly the case the region exists for. Julia's distinction between `exp(A)` and `exp.(A)` exists for this reason.
- **Adding a method breaks code.** If a dependency adds a vector method for `norm`, every region that called the scalar `norm` stops compiling. Every new overload becomes a breaking change for the whole ecosystem.

**Element always wins** is the rule I'd choose instead. Inside a region, _every_ operand of array type is its element, and every call is resolved on elements. A whole-array computation has to be bound before the region, or marked explicitly as computed once (the inverse of Julia's `$`):

```rust
// pseudocode
let centred = lift { x - mean(x) };
// error: no method `mean(f64)`; inside `lift`, `x` is an element
//   help: compute it once before the region: `let m = mean(x);`
//   help: or mark it inside the region: `once(mean(x))`

let m = mean(x);
let centred = lift { x - m };                 // fine
let centred = lift { x - once(mean(x)) };     // also fine

let h = lift { relu(once(x @ w) + b) };       // matmul once, relu and + per element
```

What this rule gives you:

- **Readable without types.** A reader knows that every array name inside the braces is an element, and that only `once(…)` and names bound outside run once. No need to know which overloads exist.
- **New methods never change meaning.** Adding a whole-array `norm` doesn't affect regions, because regions never consider whole-array methods.
- **No hidden hoisting.** Moving a call out of the loop is only correct if the call is pure, and it changes when the call runs. In `lift { if c { f(mean(x)) } else { 0.0 } }`, an automatically hoisted `mean(x)` runs even when `c` is false everywhere, and panics on an empty `x`. Making hoisting explicit puts that decision with the person who can see it.

The cost is that `lift { v / norm(v) }` silently means "each element divided by its absolute value" when a scalar `norm` exists. But that is what the rule says every time, so the reader can see it, and it's the same outcome as Julia's `v ./ norm.(v)`.

**Expressions without arrays** are a different matter: they produce the same value in every iteration, so the compiler computes them once, without any marker:

```rust
let z = lift { exp(-(x - mu)^2 / (2.0 * sigma^2)) };   // 2.0 * sigma^2: computed once
```

## How a lifted result is typed

The result of a region is typed by one rule:

1. **The element type** is the type of the element expression.
2. **The shape** is the strict broadcast of the array operands' shapes.
3. **The device** is the operands' common device.

**The output element type can differ from the inputs':**

```rust
fn is_outlier(v: f64, limit: f64) -> bool { abs(v) > limit }
let mask = lift { is_outlier(x, 3.0) };   // Array<bool, 1>
```

**Elements can be structs:**

```rust
struct Vec3 { x: f64, y: f64, z: f64 }
fn len3(v: Vec3) -> f64 { sqrt(v.x^2 + v.y^2 + v.z^2) }

let points: Array<Vec3, 1> = load_points();
let radii = lift { len3(points) };         // Array<f64, 1>
```

**Methods are selected per element type, statically.** If `add` has methods for `f32` and `f64`, `lift { add(xs32, ys32) }` uses the `f32` one, chosen at compile time.

**No implicit conversions, lifted or not:**

```rust
let a: Array<f32, 1> = ...;
let b: Array<f64, 1> = ...;
let c = lift { a + b };       // error: no method `add(f32, f64)`
                              //   help: convert explicitly: `lift { f64::from(a) + b }`
```

**Devices must agree:**

```rust
let gx = gpu.upload(&x);              // Array<f64, 1, Gpu>
let h  = lift { gx + y };             // error: operands on `Gpu` and `Cpu`
                                      //   help: `gpu.upload(&y)` or `gx.read()?`
```

## Shapes: strict broadcasting

NumPy's rules are permissive. Adding a vector of shape `(n,)` to a column of shape `(n, 1)` silently gives an `(n, n)` matrix. That has caused enough wrong results to argue for **strict** broadcasting.

**Shapes must match exactly**, unless an operand is a scalar:

```rust
let u = Array::<f64, 1>::zeros(1000);
let v = Array::<f64, 1>::zeros(999);
let w = lift { u + v };        // dynamic extents: one check before the loop,
                               // a panic naming both shapes if they differ
```

**Static extents turn the check into a compile error:**

```rust
let p: Array<f64, (Static<3>,)> = ...;
let q: Array<f64, (Static<4>,)> = ...;
let r = lift { p + q };        // compile error: extents 3 and 4 don't match
```

**Rank never changes implicitly.** Reductions take their axis as a compile-time argument, because the result's rank depends on it:

```rust
let a: Array<f64, 2> = ...;              // n × m
let mu = mean::<axis = 0>(&a);           // Array<f64, 1>, length m: the column means
let bad = lift { a - mu };               // error: rank 2 vs rank 1
                                         //   help: mark the axis to broadcast: `mu.new_axis(0)`
let centred = lift { a - mu.new_axis(0) };   // each row minus the column means
```

**Broadcast axes are spelled out**, so a reader sees which way the data is stretched without knowing the shapes:

```rust
let row_norms = sqrt(sum::<axis = 1>(&lift { a * a }));   // Array<f64, 1>, length n
let normalised = lift { a / row_norms.new_axis(1) };       // each row divided by its norm
```

NumPy's `(n,) + (n, 1)` needs an explicit `new_axis` on both sides. If you write that, you meant it. Fortran is the native precedent here: array operands must be _conformable_ (same shape, or scalar), with no size-1 stretching. Chapel's zippered iteration likewise requires matching shapes.

There is no `keepdims`. Positional code adds `new_axis` where it means it, and labelled code doesn't need it at all.

## Labelled axes: broadcasting by name

Positions are still easy to mix up: `axis = 0` or `axis = 1`? Which way does this matrix get stretched? xarray, einops and named-tensor proposals all exist because of this. A statically typed language can put **labels in the type**, at no run-time cost:

```rust
// pseudocode
axis Obs;                                // a label: a zero-size type
axis Var;

let x: Array<f64, (Obs: Dyn, Var: Dyn)> = load();
```

**Reductions name the axis they remove:**

```rust
let mu = mean::<axis = Obs>(&x);         // Array<f64, (Var: Dyn,)>
let sd = std::<axis = Obs>(&x);          // Array<f64, (Var: Dyn,)>
```

**When every operand is labelled, broadcasting aligns by label.** No `new_axis` is needed, because the names already say which axis is meant:

```rust
let z = lift { (x - mu) / sd };          // aligned by Var, broadcast over Obs
```

**Labels catch wrong-axis products:**

```rust
axis Hidden;
let w: Array<f32, (Var: Dyn, Hidden: Static<128>)> = ...;
let h = x @ w;                           // (Obs, Var) @ (Var, Hidden) → (Obs, Hidden)
let bad = h @ w;                         // compile error: contracts `Hidden` with `Var`
```

The history here is a warning, though. PyTorch shipped named tensors as a prototype in 2019. They remained a prototype and were effectively abandoned, largely because too few operations supported them: a named tensor passed through an unsupported function lost its names or raised an error, so users stopped bothering. xarray succeeded in the same space because it _is_ the array type for its users, and Dex showed how far typed index sets can go when labels are part of the language from the start.

From that history I'd take three rules:

- **All or nothing.** In one region, either every operand is labelled and alignment is by label, or every operand is positional. Mixing them is an error with a fix-it, not a silent switch to positional rules. Converting is explicit: `x.with_labels::<(Obs, Var)>()` attaches labels, `x.positional()` drops them.
- **Strict alignment.** A label shared by several operands must have equal extents, and a label missing from an operand is broadcast over. The result's axis order must not depend on operand order, so it should be fixed by a declared order or by the output's type, not by which operand happened to come first.
- **Library code doesn't have to know about labels.** A function written for `Array<f64, 2>` accepts labelled arrays too. An unlabelled shape in a parameter is generic over labels, not a conversion, and where the output's axes correspond to the input's, the labels come back out. This is the property PyTorch's named tensors lacked, and I think it decides whether labels get used at all.

## Fusion: one region, one loop

**A region compiles to one loop with no temporaries.** This is guaranteed by the syntax, not left to an optimiser heuristic. NumPy runs the Gaussian above in five passes with up to four temporaries. A region lowers to roughly:

```rust
let c = 2.0 * sigma^2;                 // computed once, outside
let mut z = Array::<f64, 1>::zeros(x.len());
for i in x.indices(0) {                // indices from x itself: no bounds checks
    let d = x.at(i) - mu;
    z.set(i, exp(-d^2 / c));
}
// …then vectorised
```

**Writing into an existing array allocates nothing:**

```rust
lift { z = exp(-(x - mu)^2 / c) }      // reuses z's buffer
```

**Separate regions are separate loops:**

```rust
let t = lift { x * 2.0 };              // loop 1, allocates t
let u = lift { t + 1.0 };              // loop 2
let v = lift { x * 2.0 + 1.0 };        // one loop, no temporary
```

**Reductions take finished arrays**, but the optimiser can fuse a map into the reduction that consumes it:

```rust
let ss = sum(lift { x * x });          // no temporary for x * x, if fusion succeeds
```

That one is an optimisation, not a syntactic guarantee like the region itself, so the compiler should report it when it fails.

For comparison: Julia guarantees fusion per dotted expression, which is the same guarantee for one statement. C++'s Eigen gets fusion from expression templates: `a.array() * b.array() + c.array()` builds a type describing the whole expression, and the loop runs only on assignment. Fortran compilers usually fuse array expressions but may introduce temporaries when they can't rule out aliasing. Futhark and XLA fuse aggressively as compiler optimisations, which is more powerful and less predictable.

## What is not lifted

Lifting means "apply to every element". Operations on whole arrays are ordinary functions:

```rust
let p = a @ b;                         // matrix product
let total = sum(&x);                   // reduction
let probs = softmax(&logits);          // needs the whole row to normalise
let bad = a * b;                       // error: did you mean `a @ b` or `lift { a * b }`?
```

**`@` needs one contraction rule.** If it contracts the last axis of the left operand with the first axis of the right, vectors need no orientation:

```rust
let y = a @ v;                         // matrix × vector: a vector
let w = v @ a;                         // vector × matrix: a vector
let d = u @ v;                         // vector · vector: a scalar
let t = transpose(&v);                 // error: a vector has no orientation
                                       //   help: `u @ v`, `outer(u, v)` or `v.new_axis(0)`
```

**Transposes should be lazy views,** and the complex conjugate transpose should have its own name, so MATLAB's `'` vs `.'` mix-up can't happen:

```rust
let g = transpose(&a) @ a;             // no copy: the kernel reads a transposed
let h = adjoint(&z) @ z;               // conjugate transpose, for complex z
```

**In-place results don't need a second function name.** If assigning to an existing array reuses its buffer, guaranteed, there's no need for Julia-style `mul!`:

```rust
let mut c = Array::<f64, 2>::zeros((n, n));
for step in 0..steps {
    c = a @ b;                         // writes into c's buffer: no allocation
}
```

**Factorise once, solve many:**

```rust
let f = lu(&a)?;                       // Err(Singular) if a pivot is zero
for b in rhs {
    let x = solve(&f, &b)?;            // Err(IllConditioned { rcond }) if the answer would be unreliable
}
```

## Beyond rank 0: lifting functions of arrays

A region lifts _scalar_ code. Many functions take a whole vector: `softmax`, a norm, a small linear solve. Applying one to every row is the same idea one level up. APL and J have had this for decades as the _rank operator_: in J, `f"1` applies `f` to every rank-1 cell of its argument.

**A rank parameter:**

```rust
// pseudocode
let probs = lift::<rank = 1> { softmax(logits) };   // logits: n × k; softmax on each row
let lens  = lift::<rank = 1> { norm(vectors) };     // vectors: n × 3; one length per row
```

Rank is static, so this is typed like rank 0: the result shape is the leading axes plus whatever the function returns per row. The "element always wins" rule carries over unchanged: with `rank = 1` the element _is_ a row, so `norm` can only be the vector method.

**Batching a whole computation** is JAX's `vmap`, which lifts even an ODE solve over a batch. But lanes take different paths: an adaptive solver lifted over 1000 initial conditions runs every lane until the slowest one finishes, with the others masked out. For ensembles of adaptive solves, a parallel loop with one trajectory per task is usually faster, because each task adapts its own step size. Batching wins when the lanes stay in step, or when finished lanes can be refilled with new problems.

## One lift, many containers

What a region means is fixed. How it runs depends on the container, and libraries should be able to extend that. A region can desugar to an open function `broadcast` that takes the element body as a compile-time argument, with methods keyed on the **container and device**, never on the code inside. This is the role Julia's `BroadcastStyle` and NumPy's `__array_ufunc__` protocol play.

**CPU arrays:** one fused, vectorised loop.

**GPU arrays:** the same region builds a lazy graph that is compiled into a kernel:

```rust
let x = gpu.upload(&data);                       // Array<f32, 2, Gpu>
let h = lift { relu(once(x @ w) + b) };          // matmul plus a fused epilogue (bias, relu)
let loss = mean(&h);                             // still on the GPU; read back explicitly
```

**Sparse matrices (library):** a sparse library can map only the stored nonzeros, but only when the code maps zero to zero:

```rust
let s: Csr<f64> = ...;
let t = lift { abs(s) };                // abs(0) == 0: map the nonzeros, keep the pattern
let u = lift { exp(s) };                // exp(0) == 1: every entry changes
let v = lift { s + 1.0 };               // 0 + 1 == 1: every entry changes
```

How a function declares "zero maps to zero" is an open problem, and it's subtler for floats than it looks: `-0.0` is a zero that some functions map differently, and NaN propagation means "zero in, zero out" can't be checked by evaluating `f(0.0)` once. Until there is a way to declare it, a sparse `broadcast` that can't prove it must produce a dense result or reject the call, never silently treat `exp(0)` as `0`.

**Ranges (library):** lifting over a range can return a range in closed form:

```rust
let r = lift { 2 * (0..1000) };         // a stepped range 0, 2, 4, …, 1998
```

**Why the lowering can't be keyed on the function.** It is tempting to let libraries override the lifted form of a specific function, say a faster whole-array `exp`. I'd decide against it: "`lift { f(x) }` is `f` applied to every element" is a guarantee worth more than the flexibility. A better whole-array algorithm for one function is an array function with its own name.

## Lanes: a generic scalar function becomes SIMD

On the CPU, a region is vectorised. The lane code can come from the same scalar code, instantiated at a lane type, an idea Mojo builds its whole numerics story on: its `SIMD[dtype, width]` type makes a scalar just a SIMD value of width 1.

**A generic function over `T: Num` is instantiated at `Simd<T, N>`:**

```rust
fn axpy<T: Num>(a: T, x: T, y: T) -> T { a * x + y }
let r = lift { axpy(2.0, xs, ys) };      // the loop body is axpy at Simd<f64, N>
```

**Simple branches become selects** (if-conversion):

```rust
fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo { lo } else if v > hi { hi } else { v }
}
let c = lift { clip(x, 0.0, 1.0) };      // lane compares and selects, no branches
```

**Branches that can't be converted** fall back to a lane method if one exists, and otherwise to a scalar loop. The compiler should say which, so a slow loop is never a mystery.

The lane width depends on the machine: 128 bits on SSE and NEON, 256 on AVX2, 512 on AVX-512, and a run-time length on Arm's SVE. That's why the lane type is an instantiation chosen by the compiler, and the function is written once.

**Should the vector `exp` give the same bits as the scalar one?** Native vector maths libraries usually don't promise that: glibc's libmvec documents error bounds of a few ULP, and SLEEF ships both 1-ULP and 3.5-ULP variants. Bit-identical scalar and vector versions cost some speed, but they mean that vectorising a loop never changes its result. I think that's worth it, scoped to one target and build. Across instruction sets, see the reproducibility tiers in the [companion post](/blog/2026-11-composing-execution).

## How this compares

| Language       | Marker                                         | User scalar functions               | Fusion                                   | Broadcasting                           | Unmarked element-wise call on arrays                       |
| -------------- | ---------------------------------------------- | ----------------------------------- | ---------------------------------------- | -------------------------------------- | ---------------------------------------------------------- |
| NumPy          | none (ufuncs)                                  | `np.vectorize` (slow), Numba        | none (numexpr, Numba)                    | permissive: `(n,) + (n, 1)` → `(n, n)` | works (shape-polymorphic)                                  |
| Julia          | a dot per operation, or `@.` per expression    | lift directly                       | guaranteed per dotted expression         | permissive, size-1 dims stretch        | `MethodError` at run time, or a matrix function (`exp(A)`) |
| APL / J        | none; rank operator for higher ranks           | lift directly                       | interpreter-dependent                    | rank-based                             | works                                                      |
| JAX            | none for primitives; `vmap` for functions      | `vmap`                              | by the XLA compiler                      | NumPy-like                             | works                                                      |
| Fortran        | none (`elemental`, array syntax)               | `elemental` procedures              | usually; temporaries if aliasing unclear | strict: conformable shapes or scalar   | works if `elemental`, else compile error                   |
| Chapel         | none (promotion)                               | any procedure promotes              | promoted expressions form one loop       | shapes must match                      | works, and runs in parallel                                |
| C++ Eigen      | `.array()` switches to element-wise            | `unaryExpr`, `binaryExpr`           | expression templates, until assignment   | explicit `.colwise()` / `.rowwise()`   | compile error on `Matrix`                                  |
| Rust `ndarray` | operators element-wise; closures for functions | `mapv`, `Zip`                       | within one `Zip` or closure              | NumPy-like for operators               | operators work; functions need a closure                   |
| Mojo           | library helpers (`vectorize`)                  | written generically over SIMD width | manual                                   | n/a                                    | n/a                                                        |
| Proposed       | one `lift` per region                          | lift directly                       | guaranteed per region                    | strict; axes explicit or labelled      | compile error with a fix                                   |

Two columns don't fit the table. **Whole-array calls inside a lifted expression:** NumPy and Fortran don't have the question, Julia's `@.` lifts them unless escaped with `$`, JAX traces them, and the proposal requires them to be bound outside or marked `once`. **Several outputs in one loop:** only the region form (several statements, one loop) offers this as a guarantee. JAX and Futhark may achieve it through compiler fusion, and Eigen and Julia need one loop per assignment.

## What is still open

- **Rank-k lifts:** whether `lift::<rank = k>` and a `vmap` belong in the core language.
- **Zero preservation:** how a function declares `f(0) == 0`, signed zeros and NaN included, so sparse lifts can keep the pattern.
- **If-conversion:** how far it reaches when branchy code is instantiated at a lane type.
- **Labels and parallelism:** whether blocked and parallel loops take a label to choose the axis they split along.
- **Reductions inside a region:** whether a region may accumulate (say a sum alongside its element-wise outputs) in a fixed, reproducible summation order.
- **Vector-space operators:** whether `a + b` on same-shape arrays is convenience worth an exception to "element-wise means `lift`".

## The short version

- Write functions for **one element**.
- Lift them with **one region**, not a marker per operation.
- Inside a region, every array is an element, and anything computed once is said so explicitly.
- One region is one loop, even with several statements.
- Shapes must match unless you say otherwise, by position or by label.
- Containers and devices decide how a region runs. The function never has to know.

`add` stays `(f64, f64) -> f64`, and works on every shape.
