---
title: 'Filtering & convolution'
description: 'FIR vs IIR filters, filter design, zero-phase filtering, moving averages and Savitzky–Golay smoothing, and fast convolution: cleaning and smoothing signals.'
aliases:
  [
    digital filter,
    FIR filter,
    IIR filter,
    Butterworth,
    Chebyshev filter,
    Bessel filter,
    low-pass,
    high-pass,
    band-pass,
    notch filter,
    filtfilt,
    zero-phase filtering,
    moving average,
    exponential smoothing,
    Savitzky-Golay,
    median filter,
    convolution,
    deconvolution,
    impulse response,
    transfer function,
    z-transform,
    numerical differentiation of noisy data
  ]
tags: [signal processing, filtering, smoothing]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Smoothing noisy measurements while keeping peak shapes:** Savitzky–Golay (local polynomial fits), which also gives derivatives.
- **Remove a frequency band cleanly:** design a filter. **FIR** (linear phase, always stable, long) or **IIR** (Butterworth etc.; short, efficient, nonlinear phase).
- **Offline data:** apply filters forwards and backwards (`filtfilt`) for **zero phase** distortion.
- **Spikes/outliers:** median filter (nonlinear, preserves edges).
- **Long kernels:** convolve via FFT (see [Fourier & FFT](/ref/approx/fourier-fft)).
- **Derivatives of noisy data:** never raw finite differences; use Savitzky–Golay, smoothing splines, or total-variation regularised differentiation.

## Decision table

| Method                              | Type                    | Phase                    | Cost                    | Use when                                      |
| ----------------------------------- | ----------------------- | ------------------------ | ----------------------- | --------------------------------------------- |
| Moving average                      | FIR                     | linear (delay)           | $O(n)$ (running sum)    | quick smoothing; poor frequency response      |
| Exponential smoothing               | IIR (1st order)         | nonlinear                | $O(n)$                  | online smoothing, trends                      |
| Savitzky–Golay                      | FIR (polynomial)        | zero (centred)           | $O(nw)$                 | spectra, peaks, smoothed derivatives          |
| Gaussian filter                     | FIR                     | zero (centred)           | $O(nw)$ or FFT          | smooth kernel, scale-space                    |
| Windowed-sinc / Parks–McClellan FIR | FIR                     | linear                   | $O(nw)$ or FFT          | precise passband/stopband specs, linear phase |
| Butterworth IIR                     | IIR                     | nonlinear (use filtfilt) | $O(n\cdot\text{order})$ | flat passband, efficient                      |
| Chebyshev / elliptic IIR            | IIR                     | nonlinear                | low                     | sharp transitions, ripple allowed             |
| Bessel IIR                          | IIR                     | ~linear in passband      | low                     | preserve waveform shape (step responses)      |
| Median filter                       | nonlinear               | —                        | $O(nw\log w)$           | impulsive noise, spikes                       |
| Notch filter                        | IIR (2nd order)         | nonlinear                | low                     | remove mains hum (50/60 Hz)                   |
| Smoothing spline / Whittaker        | penalised least squares | zero                     | $O(n)$ (banded)         | smoothing with automatic parameter (GCV)      |
| TV-regularised differentiation      | optimisation            | —                        | iterative               | derivatives of noisy, piecewise smooth data   |

## Variants

### Linear filters basics

- **Convolution:** $y[n] = \sum_kh[k]\,x[n - k]$; $h$ = impulse response.
- **Difference equation (IIR):** $\sum_ka_ky[n - k] = \sum_kb_kx[n - k]$; transfer function $H(z) = B(z)/A(z)$; stable if all poles are inside the unit circle.
- **Frequency response** $H(e^{i\omega})$: magnitude (attenuation) and phase (delay). Linear phase = constant group delay = no waveform distortion.

### FIR design

- **Windowed sinc:** ideal low-pass impulse response truncated by a window (Hamming, Kaiser); Kaiser gives explicit control of ripple vs transition width.
- **Parks–McClellan (Remez):** equiripple optimal designs for given band edges.
- Length needed grows as (sampling rate / transition width); long FIRs → FFT convolution.

### IIR design

- Analog prototypes (Butterworth, Chebyshev, elliptic, Bessel) mapped to digital via the bilinear transform with frequency prewarping.
- Implement as **second-order sections** (biquads), not high-order direct form polynomials (numerically unstable).
- **filtfilt:** forward + backward pass doubles the order and cancels phase; needs careful edge handling (padding).

### Savitzky–Golay

1. Choose window length $w$ (odd) and polynomial order $p$ (2–4).
2. For each point, least-squares fit a polynomial to the window; the fitted value (or derivative) at the centre is a fixed linear combination of the window samples.
3. Precompute the convolution coefficients once; apply as an FIR filter.

- Preserves peak heights and widths better than moving averages for the same noise reduction; derivatives come from derivative coefficients.

### Smoothing for derivatives

- **Smoothing splines / Whittaker smoother:** minimise $\sum(y_i - s_i)^2 + \lambda\sum(\Delta^2s_i)^2$; banded linear solve; choose $\lambda$ by GCV.
- **Total-variation regularised differentiation:** minimise data misfit + $\lambda\lVert\Delta u\rVert_1$ of the derivative; preserves jumps.
- **GP regression** gives derivatives with uncertainty (see [Gaussian processes](/ref/approx/gaussian-processes)).

## Other lenses

- **Linear algebra.** Convolution is multiplication by a Toeplitz (or circulant) matrix; deconvolution is solving that system, usually ill-conditioned and needing regularisation (Wiener filter, Tikhonov; see [least squares](/ref/solve/least-squares)).
- **ODEs.** IIR filters are discretised linear ODEs; a first-order low-pass is the ODE $\tau y' = x - y$ (exponential smoothing = explicit Euler of it).
- **Statistics.** Smoothers are linear estimators; kernel regression, local polynomial regression (LOESS) and Savitzky–Golay are closely related; the Kalman smoother is the optimal linear smoother for a state-space model (see [Kalman filters](/ref/signals/kalman-filters)).
- **ML.** Convolutional layers are learned FIR filters; dilated and causal convolutions are filter-design choices.

## Pitfalls & diagnostics

- **Phase distortion** from causal IIR filters shifts peaks and changes apparent timing; use zero-phase filtering offline.
- **Edge effects:** filters need padding (reflect, odd extension); inspect the first and last window lengths.
- **Over-smoothing** flattens real features; compare residuals (data − smoothed) for structure.
- **Filtering before fitting a model** can bias parameters (e.g. smoothing shrinks peak amplitudes); prefer modelling the noise directly when possible.
- **Aliasing:** filter before downsampling.

## See also

- [Fourier & FFT](/ref/approx/fourier-fft)
- [Spectral estimation](/ref/signals/spectral-estimation)
- [Finite differences](/ref/diff/finite-differences)
- [Interpolation](/ref/approx/interpolation) (smoothing splines)

**References:**

- Oppenheim & Schafer, _Discrete-Time Signal Processing_ (3rd ed.).
- Smith, _The Scientist and Engineer's Guide to Digital Signal Processing_ (free online).
- Schafer, "What is a Savitzky–Golay filter?", IEEE Signal Process. Mag. (2011).
