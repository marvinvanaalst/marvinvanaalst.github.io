---
title: 'Spectral estimation'
description: 'Periodograms, Welch and multitaper estimates, Lomb–Scargle for uneven sampling, autocorrelation, spectrograms and wavelets: finding frequencies and rhythms in noisy data.'
aliases:
  [
    power spectral density,
    PSD,
    periodogram,
    Welch method,
    Bartlett,
    multitaper,
    Thomson,
    Lomb-Scargle,
    uneven sampling,
    autocorrelation,
    Wiener-Khinchin,
    spectrogram,
    STFT,
    wavelet transform,
    continuous wavelet transform,
    cross-spectrum,
    coherence,
    circadian rhythm detection,
    JTK_CYCLE,
    cosinor,
    peak detection
  ]
tags: [signal processing, time series, frequency]
order: 4
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Raw periodogram** $\lvert\text{FFT}\rvert^2$ is an inconsistent estimator: its variance doesn't shrink with more data. Average it.
- **Welch** (averaged, windowed, overlapping segments) is the default PSD estimate; **multitaper** gives better bias/variance trade-offs, especially for short records.
- **Uneven sampling or gaps:** Lomb–Scargle periodogram (with false-alarm probabilities).
- **Short, noisy biological time series** (circadian rhythms with few cycles): fit periodic models (cosinor, harmonic regression) and test, rather than spectral estimation.
- **Frequencies changing over time:** spectrogram (STFT) or continuous wavelet transform.

## Decision table

| Method                        | Sampling | Variance                | Resolution                   | Use when                                          |
| ----------------------------- | -------- | ----------------------- | ---------------------------- | ------------------------------------------------- |
| Periodogram                   | uniform  | high (doesn't decrease) | best ($1/T$)                 | quick look, long records with strong peaks        |
| Bartlett / Welch              | uniform  | reduced by averaging    | reduced ($1/T_{\text{seg}}$) | **default PSD**                                   |
| Multitaper (Thomson)          | uniform  | reduced, low leakage    | set by bandwidth $W$         | short records, high dynamic range                 |
| Autoregressive (Burg)         | uniform  | low                     | high (parametric)            | short data, sharp peaks (model order matters)     |
| Lomb–Scargle                  | uneven   | like periodogram        | $1/T$                        | astronomy, field data, missing samples            |
| Cosinor / harmonic regression | any      | parametric              | known period                 | testing a known rhythm (24 h), few cycles         |
| Autocorrelation function      | uniform  | —                       | time domain                  | periodicity, memory, model identification         |
| Spectrogram (STFT)            | uniform  | per window              | time–frequency trade-off     | slowly changing frequencies                       |
| Continuous wavelet transform  | uniform  | per scale               | multiresolution              | transients, period changes (e.g. circadian drift) |
| Coherence / cross-spectrum    | uniform  | averaged                | as Welch                     | relationships between two signals                 |

## Variants

### Periodogram and its problems

- $\hat S(f_k) = \frac{\Delta t}{N}\lvert\sum_nx_ne^{-2\pi ikn/N}\rvert^2$ for detrended (mean-removed) data.
- Each estimate has ~100% relative standard deviation (exponential/χ² with 2 degrees of freedom) regardless of $N$.
- Leakage from the rectangular window; use a taper.

### Welch's method

1. Split the record into segments of length $L$ with 50% overlap.
2. Window each segment (Hann), compute periodograms.
3. Average across segments.

- Variance reduced by ~number of segments; frequency resolution $\approx f_s/L$. Choose $L$ for the resolution you need.

### Multitaper

- Average periodograms computed with $K$ orthogonal Slepian (DPSS) tapers; $K \approx 2NW - 1$ for time-bandwidth product $NW$ (e.g. 3–4).
- Near-optimal leakage protection and $2K$ degrees of freedom per frequency; F-test for line components.

### Lomb–Scargle

- Least-squares fit of a sinusoid at each trial frequency, accounting for uneven sampling (time offset $\tau$ makes sine/cosine terms orthogonal).
- Choose the frequency grid carefully (oversampling ~5–10× the $1/T$ spacing); false-alarm probability via analytic approximations or bootstrap.
- Generalised version includes a floating mean and measurement errors.

### Rhythm detection in short series

- **Cosinor:** $y = M + A\cos(2\pi t/P + \phi) + \varepsilon$ (linear in $\beta_1\cos + \beta_2\sin$), F-test for amplitude; estimate amplitude and phase with CIs.
- Rank-based tests (JTK_CYCLE, RAIN) for many features (transcriptomics) with multiple-testing control.

### Time–frequency

- **STFT:** windowed FFTs over sliding windows; fixed resolution trade-off (uncertainty principle).
- **Continuous wavelet transform** (Morlet): good time resolution at high frequencies, good frequency resolution at low frequencies; ridge extraction tracks period changes.

## Other lenses

- **Statistics.** The PSD is the Fourier transform of the autocovariance (Wiener–Khinchin); AR models give parametric spectra (see [time-series models](/ref/signals/time-series-models)).
- **Dynamics.** Peaks indicate oscillations, broad spectra noise or chaos; a spectrum alone cannot distinguish chaos from coloured noise (see [periodic orbits & chaos](/ref/dynamics/periodic-orbits-chaos)).
- **Regression.** Lomb–Scargle and cosinor are least-squares fits of sinusoids; spectral estimation is regression on a Fourier basis (see [least squares](/ref/solve/least-squares)).

## Pitfalls & diagnostics

- **Detrend first:** trends and offsets create spurious low-frequency power.
- **Resolution vs variance:** you can't have both; the record length limits resolution to $1/T$.
- **Peak significance:** compare against a red-noise (AR(1)) background, not white noise, for many natural signals.
- **Aliasing** in coarsely sampled data (e.g. every 4 h for a 24 h rhythm) limits detectable periods; check Nyquist.
- **Few cycles:** 2–3 cycles of a rhythm can't support sophisticated spectral claims; use parametric models with honest uncertainty.

## See also

- [Fourier & FFT](/ref/approx/fourier-fft)
- [Filtering & convolution](/ref/signals/filtering-convolution)
- [Time-series models](/ref/signals/time-series-models)
- [Periodic orbits & chaos](/ref/dynamics/periodic-orbits-chaos)

**References:**

- Percival & Walden, _Spectral Analysis for Physical Applications_.
- VanderPlas, "Understanding the Lomb–Scargle periodogram", ApJS (2018).
- Cornelissen, "Cosinor-based rhythmometry", Theor. Biol. Med. Model. (2014).
