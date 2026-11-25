---
title: 'Time-series models'
description: 'Stationarity, autocorrelation, AR, MA, ARMA and ARIMA models, their state-space form, fitting and forecasting, and how they relate to ODE and SDE models.'
aliases:
  [
    time series,
    stationarity,
    autocorrelation,
    ACF,
    PACF,
    AR model,
    MA model,
    ARMA,
    ARIMA,
    SARIMA,
    differencing,
    unit root,
    Dickey-Fuller,
    Yule-Walker,
    state-space model,
    exponential smoothing,
    ETS,
    forecasting,
    Box-Jenkins,
    Ornstein-Uhlenbeck,
    VAR,
    Granger causality
  ]
tags: [time series, statistics, forecasting]
order: 5
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Stationarity first:* remove trends and seasonality (differencing, detrending, transforms) or model them explicitly.
- *AR($`p`$):* value depends on past values; *MA($`q`$):* on past shocks; *ARMA* combines; *ARIMA* adds differencing for non-stationary series; *SARIMA* adds seasonality.
- *Identification:* ACF cuts off after lag $`q`$ → MA($`q`$); PACF cuts off after lag $`p`$ → AR($`p`$). In practice choose orders by AIC on a small grid.
- *Fit* by maximum likelihood through the *state-space form + Kalman filter* (handles missing data); Yule–Walker for quick AR estimates.
- *Forecasting baselines:* exponential smoothing (ETS) and ARIMA; always compare against naive and seasonal-naive forecasts.
- Mechanistic dynamics + noise is often better modelled as an ODE/SDE state-space model than as ARMA.

## Decision table

| Model | Equation (schematic) | Use when |
| --- | --- | --- |
| AR($`p`$) | $`x_t = \sum_{i=1}^p\phi_ix_{t-i} + \varepsilon_t`$ | persistence, PACF cut-off |
| MA($`q`$) | $`x_t = \varepsilon_t + \sum_{j=1}^q\theta_j\varepsilon_{t-j}`$ | short-memory shocks, ACF cut-off |
| ARMA($`p, q`$) | both | parsimonious stationary models |
| ARIMA($`p, d, q`$) | ARMA on $`d`$-times differenced series | trends / random-walk behaviour |
| SARIMA | + seasonal AR/MA/differencing at lag $`s`$ | seasonal data |
| ETS (exponential smoothing) | level/trend/season updated by smoothing | robust forecasting baseline |
| VAR($`p`$) | vector AR | several interacting series, Granger causality |
| Structural / state-space | explicit components (level, trend, cycles) + noise | interpretability, missing data |
| GARCH | time-varying variance | volatility clustering (finance) |
| Continuous-time (OU, CAR) | SDE observed at times $`t_i`$ | irregular sampling, links to mechanistic models |

## Variants

### Stationarity and autocorrelation

- Weakly stationary: constant mean and variance, autocovariance depends only on lag.
- *ACF* $`\rho_k`$ and *PACF* (correlation at lag $`k`$ after removing intermediate lags).
- *Unit root tests:* augmented Dickey–Fuller (null: unit root) and KPSS (null: stationary); use both.
- *Transforms:* log for multiplicative growth, Box–Cox; differencing $`\nabla x_t = x_t - x_{t-1}`$, seasonal differencing $`x_t - x_{t-s}`$.

### AR models

- Stationary iff roots of $`1 - \sum\phi_iz^i`$ lie outside the unit circle.
- *Yule–Walker:* solve the Toeplitz system of autocorrelations (Levinson–Durbin, $`O(p^2)`$); fast, slightly biased. Burg's method is better for short series.
- AR(1) $`x_t = \phi x_{t-1} + \varepsilon_t`$ is the discrete-time Ornstein–Uhlenbeck process with $`\phi = e^{-\Delta t/\tau}`$.

### ARMA fitting via state space

1. Write the ARMA model in state-space form (companion matrix).
2. Evaluate the exact Gaussian likelihood with the Kalman filter (prediction error decomposition).
3. Maximise over parameters with constraints for stationarity and invertibility (reparametrise via partial autocorrelations).
4. Missing observations: skip the update step.

### Forecasting

- Point forecasts from the conditional mean; intervals from the forecast error variance (widen with horizon; for ARIMA with $`d \ge 1`$ they grow without bound).
- *Evaluation:* rolling-origin (time-series) cross-validation with MAE/RMSE/MASE; never random splits.

### Multivariate

- *VAR:* each series regressed on lags of all series; impulse responses describe propagation of shocks.
- *Granger causality:* past of $`x`$ improves prediction of $`y`$; a statement about predictability, not causation (see [causal graphs](/ref/explain/causal-graphs-scms)).

## Other lenses

- *Dynamics.* ARMA models are discretised linear SDEs; AR poles correspond to eigenvalues of the continuous system (decay rates and oscillation frequencies). Nonlinear mechanistic models are state-space models with ODE/SDE transitions (see [SDEs](/ref/integrate/sdes)).
- *Spectral analysis.* The ARMA spectrum is $`\sigma^2\lvert\theta(e^{-i\omega})\rvert^2/\lvert\phi(e^{-i\omega})\rvert^2`$; AR models give high-resolution parametric spectral estimates (see [spectral estimation](/ref/signals/spectral-estimation)).
- *Filtering.* ARMA is an IIR filter driven by white noise; the Kalman filter is the exact inference engine (see [Kalman filters](/ref/signals/kalman-filters)).
- *ML.* Sequence models (RNNs, state-space models such as S4/Mamba, transformers) generalise AR models with nonlinear, learned dynamics; for small data, classical models are hard to beat.

## Pitfalls & diagnostics

- *Spurious regression:* regressing one trending series on another gives significant nonsense; difference or test for cointegration.
- *Over-differencing* introduces MA structure and inflates variance.
- *Residual checks:* residual ACF and Ljung–Box test should show white noise.
- *Irregular sampling:* discrete ARMA assumes equal spacing; use continuous-time models or state-space formulations with time-varying transitions.
- *Forecast evaluation leakage:* any use of future data in fitting or preprocessing invalidates the evaluation.

## See also

- [Kalman filters](/ref/signals/kalman-filters)
- [Spectral estimation](/ref/signals/spectral-estimation)
- [SDEs](/ref/integrate/sdes)
- [Resampling](/ref/freq/resampling)

*References:*

- Hyndman & Athanasopoulos, _Forecasting: Principles and Practice_ (3rd ed., free online).
- Shumway & Stoffer, _Time Series Analysis and Its Applications_ (4th ed.).
- Durbin & Koopman, _Time Series Analysis by State Space Methods_ (2nd ed.).
