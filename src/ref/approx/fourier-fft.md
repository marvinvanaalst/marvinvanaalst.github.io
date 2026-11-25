---
title: 'Fourier analysis & the FFT'
description: 'DFT conventions, FFT algorithms and costs, aliasing and sampling, windowing and leakage, FFT convolution, real-input transforms and the non-uniform FFT.'
aliases:
  [
    DFT,
    discrete Fourier transform,
    FFT,
    fast Fourier transform,
    Cooley-Tukey,
    radix-2,
    Bluestein,
    real FFT,
    rfft,
    DCT,
    aliasing,
    Nyquist,
    sampling theorem,
    spectral leakage,
    windowing,
    Hann window,
    zero padding,
    convolution theorem,
    circular convolution,
    overlap-add,
    NUFFT,
    FFTW
  ]
tags: [Fourier, signal processing, FFT]
order: 3
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- The DFT maps $`n`$ samples to $`n`$ frequency coefficients; the *FFT* computes it in $`O(n\log n)`$ for any $`n`$ (fastest for products of small primes; Bluestein/Rader handle large primes).
- *Real input:* use the real FFT (`rfft`), half the work and storage.
- *Sampling:* frequencies above Nyquist ($`f_s/2`$) alias onto lower ones; filter before downsampling.
- *Leakage:* a finite window of a non-periodic signal smears peaks; apply a window (Hann) for spectral estimates.
- *Convolution theorem:* convolution ↔ pointwise product. FFT convolution beats direct convolution for kernels longer than \~50–100 taps; pad to avoid circular wrap-around.
- *Non-uniform samples:* NUFFT (gridding + FFT), $`O(n\log n + m\log(1/\varepsilon))`$.

## Decision table

| Transform / method | Cost | Use when |
| --- | --- | --- |
| Complex FFT | $`5n\log_2 n`$ flops (radix-2) | general complex data |
| Real FFT (rfft) | \~half | real signals (almost always) |
| DCT / DST | like rfft | even/odd extensions, Chebyshev transforms, compression |
| Direct DFT | $`O(n^2)`$ | a few frequencies only (or Goertzel) |
| Direct convolution | $`O(nm)`$ | short kernels ($`m \lesssim 64`$) |
| FFT convolution (overlap-add/save) | $`O(n\log m)`$ | long kernels, streaming signals |
| NUFFT (type 1/2/3) | $`O(n\log n + m\log(1/\varepsilon)^d)`$ | non-uniform samples or frequencies (MRI, astronomy) |
| Chirp-z / zoom FFT | $`O(n\log n)`$ | fine frequency resolution in a narrow band |
| Lomb–Scargle | $`O(n\log n)`$ (fast versions) | periodograms of unevenly sampled data |

## Variants

### The DFT and its conventions

- $`\hat x_k = \sum_{j=0}^{n-1}x_je^{-2\pi ijk/n}`$, inverse $`x_j = \frac1n\sum_k\hat x_ke^{2\pi ijk/n}`$.
- Libraries differ in sign, normalisation ($`1/n`$ on inverse, or $`1/\sqrt n`$ both ways) and frequency ordering (zero frequency first, negative frequencies in the second half; `fftshift` centres them).
- *Frequency of bin $`k`$:* $`f_k = k f_s/n`$ for $`k \lt n/2`$, negative above.
- *Parseval:* $`\sum\lvert x_j\rvert^2 = \frac1n\sum\lvert\hat x_k\rvert^2`$.

### FFT algorithms

- *Cooley–Tukey:* split a size-$`n = n_1n_2`$ DFT into $`n_1`$ DFTs of size $`n_2`$, twiddle factors, $`n_2`$ DFTs of size $`n_1`$; recurse.
- *Prime sizes:* Rader (convolution of length $`p - 1`$) or Bluestein (chirp-z: convolution via a padded power-of-two FFT).
- Performance comes from memory access patterns and SIMD more than flop counts; libraries (FFTW, MKL, cuFFT) plan per size. Reuse plans.

### Sampling and aliasing

- *Nyquist–Shannon:* a signal band-limited to $`f_{\max}`$ is determined by samples at $`f_s \gt 2f_{\max}`$.
- A component at $`f`$ appears at $`\lvert f - kf_s\rvert`$ for the integer $`k`$ that brings it into $`[0, f_s/2]`$.
- Anti-aliasing: low-pass filter before sampling or decimation.
- In pseudospectral PDE codes, nonlinear products create aliased high frequencies: dealias by zero-padding (3/2 rule) or truncating the top third (2/3 rule).

### Leakage, windows, zero padding

- The DFT assumes periodic extension; a non-integer number of periods in the window creates discontinuities → energy spreads across bins (leakage).
- *Windows* trade main-lobe width (resolution) against side-lobe level (leakage): rectangular (best resolution, worst leakage), Hann (good default), Blackman–Harris (low leakage), flat-top (amplitude accuracy).
- *Zero padding* interpolates the spectrum (smoother plot) but adds no resolution; resolution is $`f_s/n_{\text{data}}`$.

### Convolution and correlation

- Circular convolution theorem: $`\widehat{x \circledast y} = \hat x\cdot\hat y`$.
- *Linear convolution* via FFT: pad both to length $`\ge n + m - 1`$.
- *Overlap-add / overlap-save* for long streams with a fixed kernel.
- *Cross-correlation:* $`\hat x\cdot\overline{\hat y}`$; autocorrelation from the power spectrum (Wiener–Khinchin).

### Non-uniform FFT

- *Type 1:* non-uniform points → uniform frequencies; *type 2:* the reverse; *type 3:* non-uniform to non-uniform.
- Spread to an oversampled uniform grid with a compact kernel (Kaiser–Bessel, "exponential of semicircle"), FFT, deconvolve. Accuracy controlled by kernel width.

## Other lenses

- *Linear algebra.* The DFT matrix is unitary (up to scaling); the FFT is a sparse factorisation of it. Circulant matrices are diagonalised by the DFT, so circulant systems solve in $`O(n\log n)`$ (used as preconditioners for Toeplitz systems).
- *Group theory.* The DFT is the Fourier transform on the cyclic group $`\mathbb Z_n`$; FFTs exist for other finite groups, and the convolution theorem holds whenever there's a group structure.
- *Approximation.* Trigonometric interpolation converges exponentially for smooth periodic functions; Chebyshev methods inherit this through $`x = \cos\theta`$ (see [Chebyshev & spectral methods](/ref/approx/chebyshev-spectral)).
- *ML.* Fourier features, FNO (Fourier neural operators) and long convolutions in sequence models (Hyena, S4) use FFTs to apply global kernels in $`O(n\log n)`$.

## Pitfalls & diagnostics

- *Normalisation and frequency-ordering conventions* differ between libraries and papers; check with a pure sinusoid of known frequency and amplitude.
- *Circular wrap-around* in FFT convolution when padding is forgotten shows up as edge artefacts.
- *Detrend* before spectral analysis; a linear trend dominates low frequencies.
- *Amplitude of a peak* depends on the window (coherent gain); correct for it if amplitudes matter.
- *Odd lengths and primes* are slower but correct; pad to a fast size only if the application allows it.

## See also

- [Spectral estimation](/ref/signals/spectral-estimation)
- [Filtering & convolution](/ref/signals/filtering-convolution)
- [Chebyshev & spectral methods](/ref/approx/chebyshev-spectral)
- [Method of lines](/ref/integrate/pdes-method-of-lines)

*References:*

- Brigham, _The Fast Fourier Transform and Its Applications_.
- Frigo & Johnson, "The design and implementation of FFTW3", Proc. IEEE (2005).
- Barnett, Magland & af Klinteberg, "A parallel nonuniform fast Fourier transform library based on an exponential of semicircle kernel", SIAM J. Sci. Comput. (2019).
