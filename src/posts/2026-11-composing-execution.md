---
title: 'Composing Execution'
description: 'Libraries know where the parallelism is, runtimes know the machine, applications know the priorities. Execution strategies only compose when each of them decides what it actually knows.'
categories:
  - musings
date: '2026-11-05'
author: 'Marvin van Aalst'
layout: tutorials
published: false
---

# Composing Execution

If you write numerical libraries, you make execution decisions all the time, usually without noticing:

- Should this loop run in parallel?
- Which device should this run on?
- How big should a block be?
- Allocate now, or reuse a buffer?
- Use the AVX-512 path, or the baseline one?

The trouble is that the same decisions are also being made by the code you call, by the code that calls you, and by whoever builds the final program. Whether those decisions **compose** or **collide** decides whether a software stack is pleasant to work in, or full of escape hatches like `n_jobs=1`, `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`.

This post argues two things. First, that the mechanism used to pass an execution decision around (an argument, a type, a scoped setting) matters much less than whether the strategy behind it composes. Second, that execution decisions belong to three different parties, and most ecosystems go wrong by letting one of them decide something only another one knows.

A companion post, [Lifting: one scalar function, every shape](/blog/2026-11-lifting-scalar-functions), covers broadcasting, which turned out to be a separate topic.

The code below mixes real languages (Rust, C++, Julia, Python, Zig, Chapel, X10, Cilk) with a Rust-flavoured pseudocode for ideas that no language I know of ships in quite this form. The pseudocode is marked as such.

## The precondition: separate what from how

You can only swap an execution strategy if the code doesn't depend on it. So the starting point is to separate _what_ a program computes from _how_ it executes. Execution decisions (device, parallelism, block size, instruction set, allocation) should change how fast a program runs, never what it returns.

Three neighbouring decisions look like execution decisions but aren't, and should stay visible in the code:

- **Lifting.** Applying a scalar function to every element of an array returns an array, applying it to a scalar returns a scalar. That is a different result type, so the call site has to choose it (see the companion post).
- **Relaxed floating point.** Fused multiply-add and reassociated sums are faster, and change the bits. GCC contracts `a * b + c` into an FMA by default in its GNU C modes (`-ffp-contract=fast`), so the same source gives different results on a machine with FMA than on one without. That is an execution decision leaking into the results, and it should be visible in the function's signature or the build, not implied by `-march=native`.
- **Precision.** PyTorch's `torch.autocast` is a block-scoped setting that silently changes dtypes, and therefore results. Precision belongs in the types.

## Three responsibilities

When a library spawns its own threads and is then called from an already-parallel context, you get the classic thread explosion: an outer loop over 16 items, each calling a matrix multiply that uses 16 threads, gives 256 runnable threads on a 16-core machine. Every attempt to fix this runs into the same tension:

- The **library author** knows _where_ an algorithm admits parallelism, which dependencies exist, and what the useful granularity is. They don't know the context the library will be called in.
- The **caller** knows the context, but not the algorithm.
- **Nobody** in the code knows how loaded the machine is right now.

The way out is to separate the _specification_ of parallelism from its _execution policy_, and to give each party exactly the decisions it has the information for:

1. **The algorithmic layer** (library code) identifies available parallelism and its trade-offs: what may run concurrently, what it costs in extra work or memory, and below which size it isn't worth it.
2. **The execution layer** (the runtime and the compiler) maps that available parallelism onto threads, cores, vector lanes and accelerators, based on the hardware and on what is free right now.
3. **The application layer** (whoever builds the final program) defines global priorities and resource constraints: how much of the machine to use, whether latency or throughput matters, which devices are allowed.

Most ecosystems collapse at least two of these into one. A BLAS that reads `OMP_NUM_THREADS` and spins up its own pool has merged the algorithmic and the execution layer. A library that takes `n_jobs` has handed an application-layer decision to every caller in between. Every case study below is about putting a decision back where its information lives.

## What makes a strategy compose

Six criteria, the same for every mechanism:

1. **One source of truth.** Don't pass what the data already carries.
2. **It nests.** A strategy used inside another strategy still works.
3. **It has a default,** so code that doesn't care never mentions it.
4. **It's open.** Others can add strategies without editing your code.
5. **It doesn't change results,** or the change is visible in types, signatures or the build.
6. **Its cost is predictable, or reported.** If the strategy is chosen implicitly, the toolchain tells you what it chose.

The sixth criterion is easy to forget, because the first five are about correctness. But implicit strategies (lazy evaluation, compiler-inserted allocation, parallelism that may or may not happen) are exactly where performance cliffs hide. Making all of them explicit in the source would defeat the point. What works is a compiler that reports what it did: which loops it fused, where it materialised a lazy expression, where it allocated, where it fell back to scalar code. A slow loop should never be a mystery.

Each layer has its natural ways to express a decision, and the mechanism follows from who knows the answer:

| Mechanism                        | Use when                                                           | Layer       | Examples                                            |
| -------------------------------- | ------------------------------------------------------------------ | ----------- | --------------------------------------------------- |
| An explicit argument (injection) | the caller knows, and the choice varies per call                   | algorithmic | solver algorithms, random keys, allocators          |
| The data                         | it is a property of the data                                       | algorithmic | device, layout, shape                               |
| Declared variants with costs     | several algorithms compute the same thing at different costs       | algorithmic | serial vs parallel scan, blocked vs unblocked       |
| The runtime                      | it is state that changes from moment to moment                     | execution   | free workers, load                                  |
| The compiler or build            | it is a property of the machine, or follows from sizes in types    | execution   | instruction set variants, block sizes               |
| A scoped execution context       | it covers a whole subtree, across libraries, and changes no result | application | worker limits, latency vs throughput, allowed devices |

Note the first row: injection is often the _right_ answer. What goes wrong in practice is injecting something that fails one of the six criteria.

## The algorithmic layer

### Mark parallelism, not threads

The algorithmic layer should say _where_ independent work exists, and nothing about how many threads should do it. The best existing models do exactly that. Cilk has `cilk_spawn` and `cilk_sync`:

```c
int x = cilk_spawn fib(n - 1);   // may run in parallel…
int y = fib(n - 2);
cilk_sync;                        // …and is joined here
return x + y;
```

X10 has `async` and `finish`, Chapel has `begin`, `cobegin` and `forall`, Rust's Rayon has `join` and parallel iterators, Julia has `Threads.@spawn`:

```rust
let (a, b) = rayon::join(|| fit(left), || fit(right));
let scores: Vec<f64> = rows.par_iter().map(score).collect();
```

None of these mention a thread count. The X10 designers were explicit about the consequence: programmers are expected to expose _more_ parallelism than the hardware can use, and the runtime maps it onto fewer resources. That is the whole architecture in one sentence.

This is also why I prefer **explicit parallel regions with implicit scheduling** over the two alternatives. Making parallelism opt-out (every map might be parallel) only works for code without side effects. Automatic parallelisation of sequential-looking code needs alias, effect and exception analysis that a general-purpose compiler can rarely complete. A parallel region is cheap to write, says exactly what the author knows, and leaves everything else to the runtime.

### Expose trade-offs, not just parallelism

"May run in parallel" isn't quite enough information, because parallelism is rarely free:

- A parallel prefix sum does about twice the work of a serial one.
- A tree reduction needs scratch space proportional to the number of workers.
- A blocked algorithm trades extra memory traffic for locality, or the other way round.
- Below some size, a parallel algorithm is just slower.

Only the library author knows these numbers. So the algorithmic layer should be able to offer _alternatives with declared costs_, and leave the choice to the execution layer. In pseudocode:

```rust
// pseudocode
#[variant(work = "n", span = "n", scratch = "0")]
fn prefix_sum(x: &mut [f64]) { /* serial scan */ }

#[variant(work = "2n", span = "log n", scratch = "workers")]
fn prefix_sum(x: &mut [f64]) { /* Blelloch scan */ }
```

The runtime picks the serial variant when there is one worker available, or when `n` is small, and the parallel one when there are idle cores and the extra work pays for itself. This is the same split that Halide makes between an _algorithm_ and its _schedule_, and the work/span vocabulary comes straight from Cilk's performance model. Today's libraries express a crude version through granularity hints such as Rayon's `with_min_len` or TBB's grain size. Those are a good start, because they move the decision about _whether_ to parallelise to the party that knows the machine, while the party that knows the algorithm says what it costs.

### Device: carried by the data

The non-composing version threads the device through every layer:

```python
def train(params, data, *, xp=numpy):
    for batch in batches(data):
        params = step(params, batch, xp=xp)        # forwarded…

def step(params, batch, *, xp=numpy):
    return update(params, grad(params, batch, xp=xp), xp=xp)   # …and forwarded again
```

The injection isn't the problem here, the redundancy is. If `batch` lives on the GPU, `xp` can only ever be CuPy. The argument fails criterion 1: two sources of truth that can disagree.

The Python Array API standard already fixes this without any static types: every conforming array has `__array_namespace__()`, and `array_api_compat.array_namespace(x)` returns the right module for whatever you were handed. The device is _carried by the data_, and code derives it instead of being told:

```python
def step(params, batch):
    xp = array_namespace(params, batch)            # one source of truth: the data
    ...
```

In a statically typed language the device can sit in the array type, and methods dispatch on it:

```rust
// pseudocode
fn normalise<D: Device>(x: &Array<f32, D>) -> Array<f32, D> {
    let n = norm(x);                               // dispatches on D
    lift { x / n }                                 // allocated where x lives
}
```

Notice that the type parameter `D` is a form of threading too, it's just one the compiler does for you. Whether the device is a type or a run-time tag matters less than the principle: one source of truth, and that source is the data. Injection survives in one place where it carries real information. Moving data onto an accelerator is an authority, not a detail, so it is an explicit call on a device handle.

There are two limits to this principle worth naming. Unified memory (Apple Silicon, CUDA managed memory, AMD's MI300A) blurs "where the data lives", because the hardware migrates pages on demand. There, placement becomes an execution-layer decision again. And with several GPUs, the _index_ of the device is a run-time value that cannot sit in a type, so the type can at most say "some GPU".

### Randomness: an exception to the default rule

Random numbers show the other side of injection: here, it is _what makes things compose_. With counter-based generators, keys are plain values, and a key derived from an index produces the same number no matter which worker computes it. JAX does this:

```python
key = jax.random.key(42)
def sample(i):
    return jax.random.normal(jax.random.fold_in(key, i))   # per-element key from the index
```

The same keys give the same numbers on one worker or sixteen, and on CPU or GPU, as long as both implement the same counter-based generator. Compare Julia before 1.7, where each thread had its own global generator, so results depended on which thread happened to run which task.

This design honestly fails criterion 3: there is no default, keys are always explicit, and JAX users complain about it. I think it's the right exception. A hidden global generator is shared mutable state, and shared mutable state is exactly what doesn't survive being parallelised.

## The execution layer

### One scheduler

The non-composing version is a separate thread pool per library. In Python:

- NumPy's BLAS, joblib, Polars and PyTorch each own a pool.
- Nested, 16 cores run 256 threads.
- So every library grows an off switch (`n_jobs=1`, `torch.set_num_threads`, `POLARS_MAX_THREADS`), and `threadpoolctl` patches the rest from the outside. Scikit-learn has a whole documentation page on oversubscription between joblib, OpenMP, MKL and OpenBLAS.

OpenMP avoids the explosion by disabling nested parallel regions by default, so the outermost level wins. That picks the wrong level whenever the outer loop is narrow.

The composing version is a single work-stealing scheduler, as in Cilk, Intel's oneTBB, Rayon, Java's ForkJoinPool and Julia since 1.3. A loop is split only when a worker is idle, so a caller with two outer tasks still gets all cores:

```rust
// both calls run in parallel; idle workers steal fit's inner chunks
let (m1, m2) = rayon::join(|| fit(left), || fit(right));
```

This answers a question that a Boolean "am I already in a parallel region?" flag cannot. Consider 100 small matrix multiplies versus 2 huge ones, both inside a parallel loop. For the 100 small ones, inner parallelism is pure overhead. For the 2 huge ones, it's the only way to use the machine. A flag gives both the same answer. Work stealing gives each the right one, because the inner loop only splits when someone is idle to take the other half. Combined with the declared costs from above, it also knows when splitting isn't worth it at all.

The runtime has a few obligations to make this work:

- **Helping while waiting.** A worker that waits for its children must run other tasks meanwhile. If it blocks the thread instead, a nested wait can deadlock the whole pool once every worker is waiting.
- **Granularity control.** A shared scheduler prevents thread explosion, but not task explosion. A parallel map that calls a function containing another parallel map can produce millions of tiny tasks. Lazy splitting, minimum grain sizes and running small tasks inline keep this in check.
- **Sequential as a first-class policy.** Running everything on one worker should be a supported mode, not a fallback: it is what you want for debugging, benchmarking a single core, or tiny inputs.
- **Resource discovery.** The number of usable cores isn't the number of logical CPUs. Containers set CPU quotas through cgroups, processes have affinity masks, and a CPU may mix performance and efficiency cores. Go only made `GOMAXPROCS` respect container CPU limits in Go 1.25. Answers to "how many cores do I have?" should also be treated as advisory, because another process can take them a millisecond later.

The operating system can't do this job. It sees threads, not tasks or dependencies, and it cannot tell a thread doing useful work from one that is spinning while it waits for a child.

There is also a limit on how greedy the scheduler should be. Maximising CPU utilisation is not the same as maximising performance. A memory-bandwidth-bound kernel saturates on a fraction of the cores, extra threads only thrash the cache, NUMA traffic punishes work that wanders between sockets, and a hot chip lowers its clock speed. That is another reason the algorithmic layer should be able to declare what kind of work it does.

### Task lifetime: structured, and transitive

When does a parallel region end? Chapel's `cobegin` waits for its immediate child tasks. X10's `finish` waits for every task started inside it, including all descendants:

```
finish {
    async compute(a);    // may itself spawn more asyncs…
    async compute(b);
}                        // …all of which have finished here
```

I think the transitive version should be the default. With `finish`-style scopes, no task outlives the block that started it, so a library can parallelise internally without its caller ever noticing, which is the property the whole design depends on. It also lets the runtime account for every task against the budget of the scope it belongs to.

One honest exception: tasks that _talk_ to each other over channels are concurrency, not parallelism. Run one after another, a producer waits forever on a full channel. Such tasks need their own threads, they live outside the parallel budget, and limiting them is a different problem.

### When the shared runtime isn't shared

In native code, a shared runtime is an ideal you never fully reach. The fastest BLAS on your machine probably uses OpenMP or its own pthreads pool. A CUDA library has its own streams. A legacy C library spawns threads in an initialiser. Making a single runtime mandatory would mean refusing to call any of them in parallel.

The useful middle ground is to treat a foreign runtime as a **budget claimant**. When a task calls into such a library, the runtime lends it a number of workers and tells the library to use exactly that many for this call, through whatever hook it offers:

- BLIS accepts thread settings per call through its runtime object.
- MKL has `mkl_set_num_threads_local`, which applies to the calling thread only.
- OpenBLAS has `openblas_set_num_threads`, unfortunately process-wide.

While the call runs, the lent workers are marked busy, so the native scheduler doesn't hand them out twice. This is what `threadpoolctl` does after the fact, done before the call instead. As a last resort, a library that can't be controlled at all can run in its own process with its own share of the cores. Composition then degrades gracefully instead of failing.

### Block size: the execution layer picks the number

A block size needs two facts. The data knows its size, the machine knows its cache. The library knows neither, so the library should only say _that_ a loop can be blocked, and never pick the number. That leaves three ways for the execution layer to choose:

1. **A compile-time budget.** The compiler derives the bytes per iteration from the element types and picks the largest block that fits a per-target cache budget. The block size is then a constant, so the vectoriser sees fixed trip counts and can unroll completely. This is the cheapest option, and it is wrong whenever the budget doesn't match the machine.
2. **Run-time tuning.** Native code can simply ask: `cpuid`, `sysconf` and `/sys/devices/system/cpu` report cache sizes. Taken further, you measure: ATLAS tunes its kernels when it is installed, and FFTW plans transforms at run time and saves the result as "wisdom".
3. **Cache-oblivious recursion.** Split in half until the problem fits in _some_ cache, without knowing which. This is asymptotically optimal for every level of the hierarchy at once, but pays for the recursion overhead and is harder to vectorise.

In pseudocode, the library's side looks the same for all three:

```rust
// pseudocode: the library says "blockable"; the execution layer picks C
for (xc, yc) in blocks(&x, &mut y) {
    let m = max(xc);                       // a reduction over the block…
    lift { yc = exp(xc - m) }              // …then a map that reuses xc from cache
}
```

Whichever option is used, "the largest power of two that fits in L1" is a simplification. Cache associativity, the number of arrays streamed at once, hyperthreads sharing an L1, and hardware prefetchers all move the optimum. That is an argument for leaving it to the execution layer, which can measure, and against hard-coding a number in a library.

There is one constraint from the results side: block boundaries must not change floating-point sums. I'll come back to that under determinism.

### Hardware variants: methods, not branches

The non-composing version picks the instruction set inside the function body, with `#ifdef`, `if constexpr`, or a compile-time `if` on the target as in Mojo:

```rust
// pseudocode for the pattern
fn dot(a: &[f32], b: &[f32]) -> f32 {
    if target_has("avx512f") { /* AVX-512 path */ }
    else if target_has("neon") { /* NEON path */ }
    else { /* baseline */ }
}
```

This fails criterion 4. To support a new target in someone else's function, you have to edit it.

Native toolchains already have the composing version: **function multiversioning**. GCC and Clang have `__attribute__((target_clones("avx2", "avx512f", "default")))` and `ifunc` resolvers, Rust has the `multiversion` crate, and Google's Highway compiles one source for several targets and dispatches at run time. Variants are separate definitions, selected once when the program loads:

```cpp
__attribute__((target("default")))  float dot(const float* a, const float* b, size_t n);
__attribute__((target("avx512f")))  float dot(const float* a, const float* b, size_t n);
```

- **Open.** Anyone can add a variant for their own type or target.
- **A default.** The baseline variant runs everywhere.
- **Visible result changes.** A variant that uses FMA or reassociates sums changes bits, so it should only be selectable by callers who opted into relaxed floating point.

The instruction set is a fact about the run, not about the data: an array's bytes don't depend on which instructions read them. That's why it belongs to the execution layer and not to the array type.

### Memory: injection, and a third way

Zig passes an allocator to everything that allocates. That is injection, and it composes well: an arena can wrap a general-purpose allocator, and a test can inject an allocator that fails on the hundredth call. C++'s polymorphic allocators compose worse: `std::pmr::vector` and `std::vector` are different types, so the strategy splits the API.

A language with value semantics can take a third route for the common case. The compiler knows each temporary's lifetime, so it can place temporaries in scratch arenas itself and reset them when they die. The reset point has to be the loop iteration, not only the function exit, or a function that allocates in a loop grows its arena for as long as the loop runs. The strategy follows from facts the compiler already has, so nobody passes anything.

This costs something compared with Zig. You lose the injected failing allocator in tests, and out-of-memory becomes a panic rather than an error you can handle. For numerical code I think that's the right trade, as long as criterion 6 holds: the compiler should report where it allocates and where it couldn't use an arena.

### Eager vs lazy: one type, not two APIs

Polars has `DataFrame` and `LazyFrame`. TensorFlow went from graphs to eager mode, and PyTorch went from eager mode to `torch.compile`. Each time, an execution strategy became a second API, and code written against one doesn't run on the other.

The alternative is one array type whose values are "a buffer _or_ a pending expression". On an accelerator, operations build a graph, the graph fuses into kernels, and it materialises when you read the result back:

```rust
// pseudocode
let h = lift { relu(x @ w + b) };   // lazy on a GPU: fused into one kernel
let l = mean(h).read()?;           // materialised here
```

The same function then works on eager CPU arrays and lazy GPU arrays. But a single type hides three things, and each needs an answer:

- **Error timing.** A shape error or an out-of-memory error surfaces at `read()`, not where it was caused. The graph has to keep source locations, so the error can point at the line that built the failing node.
- **Peak memory.** It depends on where the graph happens to be cut. Criterion 6 again: the toolchain should report where it materialises.
- **Whole-plan optimisation.** Part of Polars' reason for a separate `LazyFrame` is that pushing a filter down into a file scan requires seeing the whole query before running any of it. Implicit laziness gets this only if the graph reaches back to the scan, which means the reader has to be lazy too.

## The application layer

### An execution context, not a thread count

The application author decides how much of the machine to use. The tempting interface is a thread count, but a thread count is a poor proxy for what an application actually wants to say. It has three kinds of things to say:

- **Hard constraints:** at most this many workers, at most this much memory, only these devices.
- **A soft objective:** optimise for throughput, for latency, or for reproducibility.
- **A sharing policy:** when two computations compete for the same cores, split them evenly, prioritise one, or let each finish as fast as possible in turn. There is no universally correct answer here, which is precisely why it's the application's decision.

So the application layer's tool is a _scoped execution context_ that carries these, with a worker limit as the simplest case:

```rust
// pseudocode
with execution(workers: 1) {
    let m = fit(data);   // every parallel region inside, in every library, runs sequentially
}

with execution(workers: 4, objective: Latency) {
    serve(requests);
}
```

oneTBB's `task_arena` is the closest existing mechanism: it constrains concurrency and placement for everything submitted to it. Rayon's `ThreadPool::install` is a coarser version that runs a closure inside a specific pool.

A program-wide default belongs in the program's build configuration, and a dependency's setting is ignored:

```toml
# build configuration of the program
[execution]
workers = 4
```

### Isn't this dynamic scoping?

Yes. A scoped context is implicit state that flows down the call stack, which is exactly what I criticised in `autocast` above. The difference is criterion 5: `autocast` changes results, and a worker limit cannot. A program run with one worker returns the same bits as with sixteen (see the next section), so a scoped setting that only changes _how_ is harmless in a way that a scoped setting that changes _what_ is not.

Two questions remain open:

- **Inheritance.** Does the limit follow a task that is spawned inside the block but runs later? With transitive `finish` scopes the answer is easy, because no such task exists. Without them, the context has to be captured at spawn time.
- **Library misuse.** Nothing technically stops a library from wrapping its own code in `with execution(workers: 1)`, which brings back the old problem in a new form. A language could reject that in library code, or a linter could flag it. Either way, the convention has to be that libraries describe work and applications set limits.

### Beyond one machine

The layers so far assume one machine. Native code is also where MPI clusters, multi-GPU nodes and CPU+GPU co-scheduling live, and that adds one level above everything else: _which set of nodes and devices runs this?_ That decision belongs to the application layer too. I'll leave it at a sketch, because distribution changes the semantics enough to need its own treatment: communication is no longer free, and parts of the computation can fail independently.

## The other half: execution must not change results

Handing execution decisions to other parties is only safe if they can't change what the program computes. Every ecosystem that got parallelism or caching "for free" learned this the hard way:

| Where execution leaked into results                                  | The fix                                                       |
| -------------------------------------------------------------------- | ------------------------------------------------------------- |
| Per-thread RNGs made streams depend on scheduling (Julia before 1.7) | Counter-based keys derived from the index                     |
| Parallel float reductions vary with thread count (OpenMP)            | A fixed summation tree, independent of the number of workers  |
| FMA contraction depends on the build target (`-march=native`)        | Contraction is opt-in and visible                             |
| `autocast` changes dtypes in a scope (PyTorch)                       | Precision in types; scoped settings never change results      |

How strong should this guarantee be? I think it has to be tiered, because full reproducibility everywhere is expensive.

**Tier 1, always: the same bits for any number of workers and any block size, on one target and build.** This is cheap. Floating-point reductions are summed in fixed-size blocks in a fixed tree shape, every block size the execution layer picks is a multiple of that summation block, and the scheduler only decides which worker sums which block. Then a parallel region really is only a hint: removing it, or running everything on one worker, gives the same answer. That makes adding parallelism to a library a patch release instead of a behaviour change.

**Tier 2, opt-in: the same bits across instruction sets and devices.** This is expensive. SIMD width ranges from 128 bits (SSE, NEON) through 256 (AVX2) to 512 (AVX-512) and beyond (SVE), so a summation tree that is natural for one width wastes the others. GPU and CPU maths libraries implement `exp` and `sin` differently. Getting identical bits means fixing the tree for the narrowest target and using correctly rounded or identically implemented transcendentals everywhere. Intel's MKL calls this Conditional Numerical Reproducibility and documents the slowdown. ReproBLAS gets reproducible sums with a cleverer algorithm, and still pays for it. That's a price some applications should pay (regulatory work, debugging across machines) and most shouldn't, so it's an application-level objective, visible in the build or the types.

Cilk calls the property behind tier 1 _serial elision_: delete every `spawn` and you get the same program. The idea generalises to every execution decision: delete every hint (a parallel region, a pinned block size, a hardware variant), and you get the same bits. But it has a precondition that's easy to gloss over: the parallel bodies must have no observable effects and must not alias each other. If they print, the output order changes. If they fail, which error surfaces first changes. And peak memory changes, because each worker needs its own scratch space. Serial elision is a property of pure code, and a language that promises it has to make purity and non-aliasing checkable.

## The execution decision tree

Put together, every parallel loop or lifted call is executed by walking one tree. Each level names who decides:

```
lift { f(x, y) }                     WHAT (semantic): f, plus the lift
│
└─ which nodes and devices?          the application layer (sketched)
   └─ where does the data live?      the data
      ├─ CPU
      │  └─ across workers?          only where a parallel region is written:
      │     │                         the library exposes it, the runtime steals,
      │     │                         the application limits
      │     └─ which variant?        declared costs vs free workers: the runtime
      │        └─ per block?         block size: the execution layer
      │           └─ per lane?       instruction set variant: chosen at load time
      │              └─ scalar tail
      └─ accelerator
         └─ graph fusion?            the lazy graph
            └─ kernel grid?          workgroup sizes: tuning
               └─ features?          device capabilities
```

- **Parallel execution is a level of the CPU branch, not a sibling of the GPU.** On an accelerator, parallelism is the grid itself.
- **The levels nest.** Workers receive whole blocks, blocks have a fixed size, and the vector lanes see constant trip counts.
- **New accelerators** add a sibling branch with its own levels, never new parameters on everyone's functions.

## Async: the case I haven't settled

Async I/O has the same structure as everything above, and I don't know the answer yet. In Rust, C++20, Python and JavaScript, `async` colours every caller: an async function can only be awaited from another async function, so the decision spreads through the whole call graph. How a suspension is implemented (a state machine, a separate stack, a thread) is clearly an execution decision. The question is whether _that_ a function may suspend is semantic.

Native languages have tried every option:

- **Stackless async** (Rust, C++20 coroutines) compiles each async function into a state machine. Cheap, explicit, and coloured.
- **Stackful coroutines and green threads** (Go's goroutines, Java's virtual threads since JDK 21) give every task its own small stack, so any function can block and the runtime switches tasks underneath. No colour, but every task costs a stack, and calls into C need care.
- **Injected I/O** (Zig's new `Io` interface) passes the I/O implementation as an argument, so the same code runs blocking or asynchronous depending on what the caller hands in. That's injection that passes criterion 4.

What makes me hesitant to declare colour unnecessary is that suspension _can_ be observed:

- **Locks held across a suspension point.** Java's virtual threads used to pin their carrier thread inside `synchronized` blocks, which could starve the scheduler; JDK 24 fixed that.
- **Re-entrancy.** Other code can run while you're suspended, and can change state you assumed was stable.
- **Cancellation.** Structured concurrency makes every suspension point a place where a task can be cancelled, which is more observable, not less.
- **Latency.** A function that may wait for the network has a different cost model than one that can't.

With value semantics and structured concurrency, most of these go away, and what remains is latency. Whether that's enough to justify colour is the question I'd like to settle next.

## Defaults for everyone, overrides for experts

| Strategy       | Default (no thought needed)                                  | Override, and which layer holds it                               |
| -------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| Device         | wherever the data is                                         | explicit upload/download through a device handle: algorithmic   |
| Workers        | sequential unless a region is written; all free workers when it is | execution context and build configuration: application     |
| Variant        | chosen from declared costs and free workers                  | pinning a variant at the call: algorithmic                       |
| Block size     | the execution layer's choice                                 | pinning a size at the call; tuning: execution                    |
| Lanes          | the best result-preserving variant for the machine           | relaxed variants, for callers that opt in: algorithmic           |
| Random streams | none: keys are always explicit                               | seeding, `fold_in`, `split`: algorithmic                         |
| Foreign pools  | lent a share of the budget per call                          | process isolation for uncontrollable libraries: application     |
| Reproducibility | same bits for any worker count, on one target              | cross-target reproducibility as an objective: application        |

Library code holds none of the program-wide overrides. A library says _what_ it computes, _where its independent work is_ and _what the alternatives cost_. Whoever knows the rest decides the rest.

## What is still open

- Whether `async` can leave the type system, or whether latency alone justifies colour.
- How to stop libraries from setting application-level limits: a language rule, or a convention.
- How declared costs should be written down, and how much a runtime can trust them.
- How far a scoped execution context should reach across process and node boundaries.

## The short version

Execution strategies are decided by many people at many layers. Give each layer the decisions it has the information for:

- **The algorithmic layer** exposes parallelism and its trade-offs.
- **The execution layer** maps it onto the machine.
- **The application layer** sets priorities and limits.

Then choose mechanisms by whether the strategies **compose**:

- one source of truth;
- they nest;
- a default;
- open to others;
- results never change, or the change is visible;
- costs are predictable, or reported.

Injection is fine. Two thread pools are not.
