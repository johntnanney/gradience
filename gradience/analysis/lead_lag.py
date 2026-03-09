"""
Lead-lag analysis: cross-correlation, Granger causality, ridge forecasting.

Determines whether geometric telemetry signals (grad_norm aggregates)
temporally lead validation performance changes during LoRA fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# Cross-Correlation Functions (CCF)
# ---------------------------------------------------------------------------


@dataclass
class CCFResult:
    """Cross-correlation function result for one (feature, target) pair."""

    feature_name: str
    target_name: str
    lags: np.ndarray
    ccf_values: np.ndarray
    peak_lag: int
    peak_corr: float
    ci_upper: float
    ci_lower: float
    significant_lags: list[int]
    n_obs: int
    detrend_method: str


def compute_ccf(
    geometric_series: np.ndarray,
    validation_series: np.ndarray,
    max_lag: int = 10,
    detrend_method: str = "first_diff",
    feature_name: str = "feature",
    target_name: str = "target",
) -> CCFResult:
    """
    Compute sample cross-correlation function.

    Convention:
      - Negative lag = geometric feature LEADS validation metric
      - Positive lag = geometric feature LAGS validation metric

    Parameters
    ----------
    geometric_series : array of geometric feature values at eval steps
    validation_series : array of validation metric values at eval steps
    max_lag : maximum lag to compute (will be capped at N//2 - 1)
    detrend_method : "first_diff" (default) or "linear"

    Returns
    -------
    CCFResult with correlation values, peak lag, and significance info
    """
    x = np.asarray(geometric_series, dtype=float)
    y = np.asarray(validation_series, dtype=float)

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)

    if n < 4:
        return CCFResult(
            feature_name=feature_name,
            target_name=target_name,
            lags=np.array([0]),
            ccf_values=np.array([np.nan]),
            peak_lag=0,
            peak_corr=np.nan,
            ci_upper=np.nan,
            ci_lower=np.nan,
            significant_lags=[],
            n_obs=n,
            detrend_method=detrend_method,
        )

    # Detrend
    if detrend_method == "first_diff":
        x = np.diff(x)
        y = np.diff(y)
        n = len(x)
    elif detrend_method == "linear":
        x = sp_signal.detrend(x, type="linear")
        y = sp_signal.detrend(y, type="linear")

    if n < 4:
        return CCFResult(
            feature_name=feature_name,
            target_name=target_name,
            lags=np.array([0]),
            ccf_values=np.array([np.nan]),
            peak_lag=0,
            peak_corr=np.nan,
            ci_upper=np.nan,
            ci_lower=np.nan,
            significant_lags=[],
            n_obs=n,
            detrend_method=detrend_method,
        )

    # Cap max_lag
    max_lag = min(max_lag, n // 2 - 1)
    if max_lag < 1:
        max_lag = 1

    # Standardize
    x_std = (x - x.mean()) / (x.std() + 1e-12)
    y_std = (y - y.mean()) / (y.std() + 1e-12)

    # Compute CCF at each lag
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_values = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag < 0:
            # x leads y: correlate x[:-|lag|] with y[|lag|:]
            k = abs(lag)
            if k >= n:
                ccf_values[i] = np.nan
            else:
                ccf_values[i] = np.mean(x_std[: n - k] * y_std[k:])
        elif lag > 0:
            # x lags y: correlate x[lag:] with y[:-lag]
            if lag >= n:
                ccf_values[i] = np.nan
            else:
                ccf_values[i] = np.mean(x_std[lag:] * y_std[: n - lag])
        else:
            ccf_values[i] = np.mean(x_std * y_std)

    # 95% CI (Bartlett approximation)
    ci = 1.96 / np.sqrt(n)

    # Peak lag (maximum absolute correlation)
    valid_ccf = ~np.isnan(ccf_values)
    if valid_ccf.any():
        abs_ccf = np.abs(np.where(valid_ccf, ccf_values, 0))
        peak_idx = np.argmax(abs_ccf)
        peak_lag = int(lags[peak_idx])
        peak_corr = float(ccf_values[peak_idx])
    else:
        peak_lag = 0
        peak_corr = np.nan

    # Significant lags
    significant = [int(lags[i]) for i in range(len(lags)) if not np.isnan(ccf_values[i]) and abs(ccf_values[i]) > ci]

    return CCFResult(
        feature_name=feature_name,
        target_name=target_name,
        lags=lags,
        ccf_values=ccf_values,
        peak_lag=peak_lag,
        peak_corr=peak_corr,
        ci_upper=ci,
        ci_lower=-ci,
        significant_lags=significant,
        n_obs=n,
        detrend_method=detrend_method,
    )


def aggregate_ccfs(ccf_results: list[CCFResult]) -> dict[str, Any]:
    """
    Aggregate CCF results across multiple runs.

    Uses the UNION of lag ranges (with NaN for runs missing a lag)
    rather than intersection, so short-series runs don't collapse the range.
    """
    if not ccf_results:
        return {}

    # Find union lag range
    all_lags = [r.lags for r in ccf_results]
    min_lag = min(l.min() for l in all_lags)
    max_lag_val = max(l.max() for l in all_lags)
    union_lags = np.arange(min_lag, max_lag_val + 1)

    # Collect CCF values aligned to union lags (NaN where run doesn't cover)
    ccf_matrix = []
    for r in ccf_results:
        row = []
        for lag in union_lags:
            idx = np.where(r.lags == lag)[0]
            if len(idx) > 0:
                row.append(r.ccf_values[idx[0]])
            else:
                row.append(np.nan)
        ccf_matrix.append(row)

    ccf_matrix = np.array(ccf_matrix)
    mean_ccf = np.nanmean(ccf_matrix, axis=0)
    std_ccf = np.nanstd(ccf_matrix, axis=0)
    count_per_lag = np.sum(~np.isnan(ccf_matrix), axis=0).tolist()

    # Peak lag statistics
    peak_lags = [r.peak_lag for r in ccf_results if not np.isnan(r.peak_corr)]
    leads_count = sum(1 for pl in peak_lags if pl < 0)

    return {
        "feature_name": ccf_results[0].feature_name,
        "target_name": ccf_results[0].target_name,
        "n_runs": len(ccf_results),
        "common_lags": union_lags.tolist(),
        "mean_ccf": mean_ccf.tolist(),
        "std_ccf": std_ccf.tolist(),
        "count_per_lag": count_per_lag,
        "peak_lag_mean": float(np.mean(peak_lags)) if peak_lags else np.nan,
        "peak_lag_std": float(np.std(peak_lags)) if peak_lags else np.nan,
        "peak_corr_mean": float(np.mean([r.peak_corr for r in ccf_results if not np.isnan(r.peak_corr)])),
        "leads_count": leads_count,
        "total_runs": len(peak_lags),
    }


# ---------------------------------------------------------------------------
# Granger Causality
# ---------------------------------------------------------------------------


@dataclass
class GrangerResult:
    """Granger causality test result."""

    feature_name: str
    target_name: str
    selected_lag: int
    f_statistic: float
    p_value: float
    reject_null: bool
    restricted_sse: float
    unrestricted_sse: float
    delta_r_squared: float
    n_obs: int


def granger_causality_test(
    geometric_series: np.ndarray,
    validation_series: np.ndarray,
    max_lag: int = 3,
    feature_name: str = "feature",
    target_name: str = "target",
) -> GrangerResult:
    """
    Manual Granger causality test using OLS.

    Tests whether past values of geometric_series improve prediction of
    validation_series beyond an autoregressive model.

    Uses first-differenced series to handle trends.
    """
    from scipy import stats as sp_stats

    x = np.asarray(geometric_series, dtype=float)
    y = np.asarray(validation_series, dtype=float)

    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    # First-difference
    dx = np.diff(x)
    dy = np.diff(y)
    n_total = len(dy)

    # Select lag via AIC (try 1 to max_lag)
    max_lag = min(max_lag, n_total // 3)
    if max_lag < 1:
        max_lag = 1

    best_lag = 1
    best_aic = np.inf

    for p in range(1, max_lag + 1):
        n = n_total - p
        if n < p + 2:
            continue

        # Build restricted model: y_t = c + sum(beta_i * y_{t-i})
        Y = dy[p:]
        X_r = np.column_stack([dy[p - i - 1 : n_total - i - 1] for i in range(p)])
        X_r = np.column_stack([np.ones(n), X_r])

        try:
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            resid_r = Y - X_r @ beta_r
            sse_r = float(np.sum(resid_r**2))
            aic = n * np.log(sse_r / n + 1e-12) + 2 * X_r.shape[1]
            if aic < best_aic:
                best_aic = aic
                best_lag = p
        except np.linalg.LinAlgError:
            continue

    p = best_lag
    n = n_total - p
    if n < p + 2:
        return GrangerResult(
            feature_name=feature_name,
            target_name=target_name,
            selected_lag=p,
            f_statistic=np.nan,
            p_value=1.0,
            reject_null=False,
            restricted_sse=np.nan,
            unrestricted_sse=np.nan,
            delta_r_squared=np.nan,
            n_obs=n,
        )

    Y = dy[p:]

    # Restricted: AR(p) on y only
    X_r = np.column_stack([np.ones(n)] + [dy[p - i - 1 : n_total - i - 1] for i in range(p)])
    beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
    resid_r = Y - X_r @ beta_r
    sse_r = float(np.sum(resid_r**2))

    # Unrestricted: AR(p) on y + lags of x
    X_u = np.column_stack([X_r] + [dx[p - i - 1 : n_total - i - 1] for i in range(p)])
    beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
    resid_u = Y - X_u @ beta_u
    sse_u = float(np.sum(resid_u**2))

    # F-test
    df1 = p  # number of restrictions
    df2 = n - X_u.shape[1]

    if df2 <= 0 or sse_u <= 0:
        return GrangerResult(
            feature_name=feature_name,
            target_name=target_name,
            selected_lag=p,
            f_statistic=np.nan,
            p_value=1.0,
            reject_null=False,
            restricted_sse=sse_r,
            unrestricted_sse=sse_u,
            delta_r_squared=np.nan,
            n_obs=n,
        )

    f_stat = ((sse_r - sse_u) / df1) / (sse_u / df2)
    p_value = float(1.0 - sp_stats.f.cdf(f_stat, df1, df2))

    # R² improvement
    ss_total = float(np.sum((Y - Y.mean()) ** 2))
    r2_r = 1 - sse_r / (ss_total + 1e-12)
    r2_u = 1 - sse_u / (ss_total + 1e-12)
    delta_r2 = r2_u - r2_r

    return GrangerResult(
        feature_name=feature_name,
        target_name=target_name,
        selected_lag=p,
        f_statistic=float(f_stat),
        p_value=p_value,
        reject_null=p_value < 0.05,
        restricted_sse=sse_r,
        unrestricted_sse=sse_u,
        delta_r_squared=delta_r2,
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# Ridge Forecasting
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Ridge regression forecast evaluation."""

    predictions: np.ndarray
    actuals: np.ndarray
    persistence_predictions: np.ndarray
    rmse_model: float
    rmse_persistence: float
    rmse_reduction_pct: float
    r_squared_oos: float
    n_predictions: int
    feature_importances: dict[str, float]
    horizon: int
    include_ar: bool


def ridge_forecast(
    aligned_df: pd.DataFrame,
    geometric_features: list[str],
    target: str,
    horizon: int = 1,
    ar_lags: int = 2,
    include_ar: bool = False,
    alpha: float = 1.0,
    min_train_size: int = 5,
) -> ForecastResult:
    """
    Walk-forward ridge regression forecast.

    Parameters
    ----------
    aligned_df : DataFrame with eval-step-indexed features and targets
    geometric_features : column names of geometric features to use
    target : column name of the target variable (e.g., "eval_loss")
    horizon : forecast horizon in eval steps (1 = next step)
    ar_lags : number of autoregressive lags to include (if include_ar)
    include_ar : whether to include past target values as features
    alpha : ridge regularization strength
    min_train_size : minimum training window size
    """
    from sklearn.linear_model import Ridge

    df = aligned_df.copy()

    # Build feature matrix with lags
    feature_cols = []
    for feat in geometric_features:
        if feat not in df.columns:
            continue
        feature_cols.append(feat)
        # Add 1-2 lags of each geometric feature
        for lag in range(1, min(3, len(df) // 3)):
            col_name = f"{feat}_lag{lag}"
            df[col_name] = df[feat].shift(lag)
            feature_cols.append(col_name)

    if include_ar:
        for lag in range(1, ar_lags + 1):
            col_name = f"{target}_lag{lag}"
            df[col_name] = df[target].shift(lag)
            feature_cols.append(col_name)

    # Create target: y_{t+h}
    df["_target"] = df[target].shift(-horizon)

    # Drop rows with NaN in features or target
    cols_needed = feature_cols + ["_target", target]
    df_clean = df.dropna(subset=[c for c in cols_needed if c in df.columns])

    if len(df_clean) < min_train_size + 2:
        return ForecastResult(
            predictions=np.array([]),
            actuals=np.array([]),
            persistence_predictions=np.array([]),
            rmse_model=np.nan,
            rmse_persistence=np.nan,
            rmse_reduction_pct=np.nan,
            r_squared_oos=np.nan,
            n_predictions=0,
            feature_importances={},
            horizon=horizon,
            include_ar=include_ar,
        )

    X = df_clean[feature_cols].values
    y = df_clean["_target"].values
    y_current = df_clean[target].values  # for persistence baseline

    # Walk-forward CV: first 50% as initial window, expand
    initial_train = max(min_train_size, len(X) // 2)
    predictions = []
    actuals = []
    persistence_preds = []
    all_coefs = []

    for t in range(initial_train, len(X)):
        X_train, y_train = X[:t], y[:t]
        X_test = X[t : t + 1]
        y_test = y[t]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]
        predictions.append(pred)
        actuals.append(y_test)
        persistence_preds.append(y_current[t])  # persistence: predict current value
        all_coefs.append(model.coef_)

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    persistence_preds = np.array(persistence_preds)

    if len(predictions) == 0:
        return ForecastResult(
            predictions=predictions,
            actuals=actuals,
            persistence_predictions=persistence_preds,
            rmse_model=np.nan,
            rmse_persistence=np.nan,
            rmse_reduction_pct=np.nan,
            r_squared_oos=np.nan,
            n_predictions=0,
            feature_importances={},
            horizon=horizon,
            include_ar=include_ar,
        )

    rmse_model = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    rmse_persist = float(np.sqrt(np.mean((persistence_preds - actuals) ** 2)))
    rmse_reduction = (1 - rmse_model / (rmse_persist + 1e-12)) * 100

    ss_res = np.sum((predictions - actuals) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2_oos = float(1 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else np.nan

    # Feature importances: mean absolute coefficient
    mean_coefs = np.mean(np.abs(np.array(all_coefs)), axis=0)
    importances = {col: float(c) for col, c in zip(feature_cols, mean_coefs)}

    return ForecastResult(
        predictions=predictions,
        actuals=actuals,
        persistence_predictions=persistence_preds,
        rmse_model=rmse_model,
        rmse_persistence=rmse_persist,
        rmse_reduction_pct=rmse_reduction,
        r_squared_oos=r2_oos,
        n_predictions=len(predictions),
        feature_importances=importances,
        horizon=horizon,
        include_ar=include_ar,
    )


# ---------------------------------------------------------------------------
# Surrogate Null Tests
# ---------------------------------------------------------------------------


@dataclass
class SurrogateResult:
    """Result of surrogate null hypothesis test."""

    actual_rmse_reduction: float
    surrogate_reductions: np.ndarray
    p_value: float
    z_score: float
    method: str
    n_surrogates: int


def surrogate_null_test(
    aligned_df: pd.DataFrame,
    geometric_features: list[str],
    target: str,
    horizon: int = 1,
    n_surrogates: int = 200,
    method: str = "circular_rotation",
    alpha: float = 1.0,
    rng_seed: int = 42,
) -> SurrogateResult:
    """
    Test whether the ridge forecast improvement is significantly better than
    chance using surrogate time series.

    Methods:
    - "circular_rotation": shift geometric series by random offset (wrapping)
    - "phase_randomization": FFT, randomize phases, inverse FFT
    """
    rng = np.random.default_rng(rng_seed)

    # Get actual forecast result
    actual = ridge_forecast(aligned_df, geometric_features, target, horizon=horizon, alpha=alpha)
    actual_reduction = actual.rmse_reduction_pct

    surrogate_reductions = []

    for _ in range(n_surrogates):
        df_surr = aligned_df.copy()

        for feat in geometric_features:
            if feat not in df_surr.columns:
                continue
            series = df_surr[feat].values.copy()
            valid_mask = ~np.isnan(series)
            valid_vals = series[valid_mask]

            if len(valid_vals) < 3:
                continue

            if method == "circular_rotation":
                shift = rng.integers(1, len(valid_vals))
                series[valid_mask] = np.roll(valid_vals, shift)
            elif method == "phase_randomization":
                fft = np.fft.rfft(valid_vals)
                phases = rng.uniform(0, 2 * np.pi, len(fft))
                fft_rand = np.abs(fft) * np.exp(1j * phases)
                randomized = np.fft.irfft(fft_rand, n=len(valid_vals))
                series[valid_mask] = randomized

            df_surr[feat] = series

        surr_result = ridge_forecast(df_surr, geometric_features, target, horizon=horizon, alpha=alpha)
        surrogate_reductions.append(surr_result.rmse_reduction_pct)

    surrogate_reductions = np.array(surrogate_reductions)

    # p-value: fraction of surrogates >= actual
    valid_surr = surrogate_reductions[~np.isnan(surrogate_reductions)]
    if len(valid_surr) > 0:
        p_value = float(np.mean(valid_surr >= actual_reduction))
        z_score = float((actual_reduction - np.mean(valid_surr)) / (np.std(valid_surr) + 1e-12))
    else:
        p_value = 1.0
        z_score = 0.0

    return SurrogateResult(
        actual_rmse_reduction=actual_reduction,
        surrogate_reductions=surrogate_reductions,
        p_value=p_value,
        z_score=z_score,
        method=method,
        n_surrogates=n_surrogates,
    )


# ---------------------------------------------------------------------------
# Full Pipeline Runner
# ---------------------------------------------------------------------------


def run_lead_lag_analysis(
    aligned_df: pd.DataFrame,
    geometric_features: list[str] | None = None,
    target: str = "eval_loss",
    max_ccf_lag: int = 5,
    run_label: str = "run",
) -> dict[str, Any]:
    """
    Run the complete lead-lag analysis pipeline on a single aligned DataFrame.

    Returns a dict with ccf, granger, and forecast results.
    """
    if geometric_features is None:
        # Auto-detect: grad_norm aggregates + structural metrics (base columns, not deltas)
        _prefixes = (
            "grad_norm_",
            "stable_rank_",
            "effective_rank_",
            "energy_rank_90_",
            "adapter_frob_norm_",
            "sigma_max_",
        )
        geometric_features = [
            c
            for c in aligned_df.columns
            if any(c.startswith(p) for p in _prefixes)
            and not c.endswith("_delta")
            and not c.endswith("_accel")
            and "_lag" not in c
        ]

    # Include delta and acceleration features
    all_geo = []
    for f in geometric_features:
        all_geo.append(f)
        if f"{f}_delta" in aligned_df.columns:
            all_geo.append(f"{f}_delta")
    if "grad_norm_mean_accel" in aligned_df.columns:
        all_geo.append("grad_norm_mean_accel")

    results: dict[str, Any] = {"run_label": run_label, "n_eval_events": len(aligned_df)}

    # --- CCF ---
    ccf_results = []
    for feat in all_geo:
        if feat not in aligned_df.columns:
            continue
        series = aligned_df[feat].dropna().values
        target_series = aligned_df[target].dropna().values
        min_len = min(len(series), len(target_series))
        if min_len < 4:
            continue

        ccf = compute_ccf(
            series[:min_len],
            target_series[:min_len],
            max_lag=max_ccf_lag,
            feature_name=feat,
            target_name=target,
        )
        ccf_results.append(ccf)

    results["ccf"] = ccf_results

    # --- Granger ---
    granger_results = []
    for feat in all_geo:
        if feat not in aligned_df.columns:
            continue
        series = aligned_df[feat].dropna().values
        target_series = aligned_df[target].dropna().values
        min_len = min(len(series), len(target_series))
        if min_len < 5:
            continue

        gr = granger_causality_test(
            series[:min_len],
            target_series[:min_len],
            max_lag=2,
            feature_name=feat,
            target_name=target,
        )
        granger_results.append(gr)

    results["granger"] = granger_results

    # --- Ridge Forecast ---
    base_features = [f for f in geometric_features if f in aligned_df.columns]
    forecast_results = {}
    for h in [1]:  # only h=1 for short series
        for include_ar in [False, True]:
            label = f"h{h}_{'geo+ar' if include_ar else 'geo_only'}"
            fr = ridge_forecast(
                aligned_df,
                geometric_features=base_features,
                target=target,
                horizon=h,
                include_ar=include_ar,
                min_train_size=3,
            )
            forecast_results[label] = fr

    results["forecast"] = forecast_results

    return results
