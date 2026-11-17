---
title: 'Pseudo-random number generators'
description: 'LCGs, xorshift/xoshiro, PCG, Mersenne Twister and counter-based Philox/Threefry: quality, speed, state size, and how to get independent streams for parallel work.'
aliases:
  [
    PRNG,
    random number generator,
    RNG,
    LCG,
    linear congruential,
    xorshift,
    xoshiro256,
    xoroshiro,
    PCG,
    Mersenne Twister,
    MT19937,
    Philox,
    Threefry,
    counter-based RNG,
    SplitMix,
    stream splitting,
    jump ahead,
    seeding,
    reproducibility,
    TestU01,
    BigCrush,
    PractRand,
    CSPRNG,
    JAX PRNG keys
  ]
tags: [random numbers, simulation, reproducibility]
order: 1
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Scientific simulation default:** xoshiro256++ / xoshiro256** or PCG64. Fast (~1 ns per number), small state, pass BigCrush/PractRand.
- **Parallel and GPU work:** **counter-based** generators (Philox, Threefry): the output is a pure function of (key, counter), so streams are trivially independent and reproducible regardless of scheduling. This is what JAX keys and cuRAND/Random123 use.
- **Mersenne Twister:** still common, large state (2.5 KB), fails some linear-complexity tests, slow to seed well. Fine but no longer the best choice.
- **LCGs:** only for toy uses; low bits are poor.
- **Security (tokens, keys):** a CSPRNG from the OS (`/dev/urandom`, `getrandom`), never a simulation PRNG.
- **Reproducibility:** record seeds, use one generator per stream, never share one generator across threads.

## Decision table

| Generator          | State         | Period            | Speed      | Quality                       | Parallel streams                         | Use when                         |
| ------------------ | ------------- | ----------------- | ---------- | ----------------------------- | ---------------------------------------- | -------------------------------- |
| LCG (64-bit)       | 8 B           | $2^{64}$          | very fast  | poor (low bits, lattice)      | leapfrog / jump                          | toy examples only                |
| Mersenne Twister   | 2.5 KB        | $2^{19937} - 1$   | fast       | good; fails linear complexity | jump-ahead (expensive), dynamic creation | legacy code, compatibility       |
| xorshift128+       | 16 B          | $2^{128} - 1$     | very fast  | low bits weak                 | jump                                     | superseded by xoshiro            |
| xoshiro256++ / **  | 32 B          | $2^{256} - 1$     | very fast  | excellent                     | `jump()` ($2^{128}$ apart)               | **default** for CPU simulation   |
| PCG64 (DXSM)       | 16 B          | $2^{128}$         | very fast  | excellent                     | streams via increment, jump-ahead        | NumPy default, general use       |
| SplitMix64         | 8 B           | $2^{64}$          | very fast  | good                          | used to seed others                      | seeding, hashing                 |
| Philox4×32-10      | key + counter | $2^{128}$ per key | fast (GPU) | excellent                     | trivial: different keys/counters         | **GPUs, parallel, reproducible** |
| Threefry           | key + counter | $2^{256}$ per key | fast       | excellent (crypto-derived)    | trivial                                  | JAX default, parallel            |
| ChaCha20 / AES-CTR | key + counter | huge              | moderate   | cryptographic                 | trivial                                  | when crypto strength wanted      |

## Variants

### Linear generators

- **LCG:** $x_{k+1} = (ax_k + c) \bmod m$. Points lie on hyperplanes in $d$ dimensions (Marsaglia), low-order bits have short periods.
- **Xorshift / xoshiro:** linear recurrences over $\mathbb F_2$ using shifts and xors, with a nonlinear output scrambler (`++`, `**`) to fix linear artefacts.
- **Mersenne Twister:** twisted GFSR with a huge state; good equidistribution, but large state and slow escape from zero-heavy states.

### PCG

- An LCG state transition with a **permutation output function** (xorshift + random rotation, or DXSM multiply) that hides the LCG's weaknesses.
- Streams by choosing different odd increments; jump-ahead in $O(\log n)$.

### Counter-based generators

- $\text{output}_i = h_k(i)$: a bijective mixing function (reduced-round block cipher) of the counter $i$ under key $k$.
- **Philox:** multiplications + xors, fast on GPUs. **Threefry:** additions, rotations, xors (from Threefish).
- Splitting: derive new keys by hashing (JAX `split`), or assign each thread/particle its own key and use time step / sample index as the counter.

### Parallel streams

- **Never** use one generator from several threads (races, non-reproducible).
- **Options:**
  - One generator per worker, seeded from a master seed via a seed sequence (NumPy `SeedSequence.spawn`) or SplitMix64.
  - Jump-ahead: worker $j$ starts $j\times2^{128}$ steps ahead (xoshiro `jump`).
  - Counter-based: key = worker id (or a hash of a logical identity, e.g. particle id), counter = step. Results independent of scheduling.
- Seeding consecutive integers into some generators (old MT, LCGs) gives correlated streams; always go through a seed-mixing function.

### Floats from integers

- Double in $[0, 1)$: take the top 53 bits, multiply by $2^{-53}$. Avoid dividing by $2^{64}$ (can round to 1.0).
- Bounded integers in $[0, n)$: Lemire's multiply-shift with rejection, not `x % n` (modulo bias).

## Other lenses

- **Number theory and algebra.** Linear generators are linear maps over finite fields or rings; period and equidistribution follow from characteristic polynomials (primitive polynomials give maximal periods).
- **Cryptography.** Counter-based generators are block ciphers in counter mode with fewer rounds; cryptographic strength isn't needed for simulation, statistical quality is.
- **Functional programming.** Explicit key passing (JAX, Haskell's splittable generators) makes randomness pure and composable; splittable PRNGs are a monad-free way to thread state (see [monads & effects](/ref/programs/monads-effects)).

## Pitfalls & diagnostics

- **Reseeding per sample** with time or small integers destroys statistical properties.
- **Correlated parallel streams** from naive seeding (seed = thread id) with weak generators.
- **Statistical test suites** (TestU01 BigCrush, PractRand) check generators, not your use of them; bugs in transforming to distributions are more common than bad generators.
- **Reproducibility across versions:** library default generators and distribution algorithms change between releases; pin both if bitwise reproducibility matters.
- **GPU kernels:** per-thread state in registers; counter-based generators avoid storing state at all.

## See also

- [Sampling distributions](/ref/random/sampling-distributions)
- [Quasi-Monte Carlo](/ref/random/quasi-monte-carlo)
- [Stochastic simulation](/ref/integrate/stochastic-simulation)
- [Variance reduction](/ref/random/variance-reduction)

**References:**

- L'Ecuyer, "Random number generation", in _Handbook of Computational Statistics_ (2012).
- Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC (2011).
- Blackman & Vigna, "Scrambled linear pseudorandom number generators", ACM TOMS (2021).
