"""
Phase Transition Detection for Training Dynamics Research

Implements detection of phase transition signatures in training:

- Critical slowing down (diverging autocorrelation time)
- Fluctuation amplification (diverging variance)
- Susceptibility measures
- Order parameter tracking

These signatures, borrowed from statistical mechanics, may indicate
approach to phenomena like grokking or sudden generalization.

Theoretical Background
----------------------
Near a second-order phase transition, systems exhibit:

1. Critical slowing down: Relaxation time τ → ∞
   - Manifests as increasing autocorrelation in observables
   - System takes longer to "forget" perturbations

2. Diverging fluctuations: Variance of order parameter → ∞
   - Near criticality, fluctuations span all scales

3. Power-law distributions: P(x) ~ x^(-α)
   - Scale invariance at criticality

4. Universal exponents: Critical behavior characterized by exponents
   - Different systems in same "universality class" share exponents

In training:
- "Order parameter" might be: loss gap, accuracy, gradient norm, spectral quantities
- Phase transition might be: grokking, mode collapse, loss of rank, generalization
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import time


@dataclass
class PhaseTransitionMetrics:
    """Metrics for detecting phase transition signatures."""
    
    step: int
    timestamp: float
    
    # Autocorrelation
    autocorr_time: Optional[float]  # Integrated autocorrelation time
    autocorr_1: Optional[float]     # Lag-1 autocorrelation
    
    # Fluctuation measures
    variance: float                 # Variance of observable
    variance_ratio: Optional[float]  # Ratio to baseline variance
    
    # Susceptibility (response to perturbation)
    susceptibility: Optional[float]
    
    # Stationarity
    is_stationary: bool            # Augmented Dickey-Fuller style test
    trend: Optional[float]         # Linear trend
    
    # Phase classification
    phase: str                     # "stable", "critical", "transitioning", "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "autocorr_time": self.autocorr_time,
            "autocorr_1": self.autocorr_1,
            "variance": self.variance,
            "variance_ratio": self.variance_ratio,
            "susceptibility": self.susceptibility,
            "is_stationary": self.is_stationary,
            "trend": self.trend,
            "phase": self.phase,
        }


def compute_autocorrelation(series: List[float], max_lag: int = 50) -> List[float]:
    """
    Compute autocorrelation function up to max_lag.
    
    ACF(k) = Cov(X_t, X_{t+k}) / Var(X)
    """
    n = len(series)
    if n < max_lag + 1:
        max_lag = n - 1
    
    mean = sum(series) / n
    var = sum((x - mean)**2 for x in series) / n
    
    if var < 1e-10:
        return [1.0] + [0.0] * max_lag
    
    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            cov = sum((series[i] - mean) * (series[i + lag] - mean) 
                     for i in range(n - lag)) / (n - lag)
            acf.append(cov / var)
    
    return acf


def compute_integrated_autocorr_time(acf: List[float], cutoff: float = 0.05) -> float:
    """
    Compute integrated autocorrelation time.
    
    τ_int = 1 + 2 * Σ_{k=1}^{∞} ACF(k)
    
    Truncate when ACF drops below cutoff.
    """
    tau = 1.0
    for k, a in enumerate(acf[1:], 1):
        if abs(a) < cutoff:
            break
        tau += 2 * a
    
    return max(tau, 1.0)


def compute_variance_ratio(
    series: List[float],
    window1: int = 20,
    window2: int = 100,
) -> Optional[float]:
    """
    Compute ratio of recent variance to baseline variance.
    
    Ratio > 1 indicates amplifying fluctuations (critical signature).
    """
    if len(series) < window2:
        return None
    
    recent = series[-window1:]
    baseline = series[-window2:-window1] if len(series) > window2 else series[:-window1]
    
    if len(baseline) < 5:
        return None
    
    var_recent = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
    var_baseline = sum((x - sum(baseline)/len(baseline))**2 for x in baseline) / len(baseline)
    
    if var_baseline < 1e-10:
        return None
    
    return var_recent / var_baseline


def compute_trend(series: List[float]) -> Tuple[float, float]:
    """
    Compute linear trend via OLS.
    
    Returns (slope, r_squared).
    """
    n = len(series)
    if n < 3:
        return 0.0, 0.0
    
    x = list(range(n))
    mean_x = sum(x) / n
    mean_y = sum(series) / n
    
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, series))
    den = sum((xi - mean_x)**2 for xi in x)
    
    if den < 1e-10:
        return 0.0, 0.0
    
    slope = num / den
    
    # R²
    y_pred = [slope * xi + (mean_y - slope * mean_x) for xi in x]
    ss_res = sum((yi - yp)**2 for yi, yp in zip(series, y_pred))
    ss_tot = sum((yi - mean_y)**2 for yi in series)
    
    r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0
    
    return slope, r2


def detect_stationarity(series: List[float], window: int = 50) -> bool:
    """
    Simple stationarity test based on comparing window means.
    
    Non-stationary if windows have significantly different means.
    """
    if len(series) < 2 * window:
        return True  # Insufficient data
    
    first_half = series[:window]
    second_half = series[-window:]
    
    mean1 = sum(first_half) / len(first_half)
    mean2 = sum(second_half) / len(second_half)
    
    std1 = math.sqrt(sum((x - mean1)**2 for x in first_half) / len(first_half))
    std2 = math.sqrt(sum((x - mean2)**2 for x in second_half) / len(second_half))
    
    pooled_std = math.sqrt((std1**2 + std2**2) / 2)
    
    if pooled_std < 1e-10:
        return True
    
    # Simple z-test (not rigorous, but indicative)
    z = abs(mean2 - mean1) / (pooled_std * math.sqrt(2 / window))
    
    return z < 2.0  # ~95% confidence


def classify_phase(
    autocorr_time: Optional[float],
    variance_ratio: Optional[float],
    is_stationary: bool,
    trend_r2: float,
) -> str:
    """
    Classify current phase based on observables.
    
    Returns: "stable", "critical", "transitioning", "unknown"
    """
    # Insufficient data
    if autocorr_time is None or variance_ratio is None:
        return "unknown"
    
    # Critical signatures: high autocorr + amplified variance
    if autocorr_time > 10 and variance_ratio > 2.0:
        return "critical"
    
    # Transitioning: strong trend + non-stationary
    if not is_stationary and trend_r2 > 0.5:
        return "transitioning"
    
    # Stable: low autocorr, stable variance, stationary
    if autocorr_time < 5 and variance_ratio < 1.5 and is_stationary:
        return "stable"
    
    return "unknown"


@dataclass
class PhaseTransitionTracker:
    """
    Track observables for phase transition detection.
    
    Maintains history of an observable (e.g., loss, accuracy, spectral metric)
    and computes phase transition signatures.
    """
    
    window_size: int = 200
    observable_name: str = "loss"
    
    _history: deque = field(default_factory=deque)
    _steps: deque = field(default_factory=deque)
    _baseline_variance: Optional[float] = None
    
    def __post_init__(self):
        self._history = deque(maxlen=self.window_size)
        self._steps = deque(maxlen=self.window_size)
    
    def add(self, step: int, value: float) -> None:
        """Record observable value."""
        self._history.append(value)
        self._steps.append(step)
        
        # Compute baseline variance from first 50 observations
        if len(self._history) == 50 and self._baseline_variance is None:
            values = list(self._history)
            mean = sum(values) / len(values)
            self._baseline_variance = sum((v - mean)**2 for v in values) / len(values)
    
    def compute_metrics(self) -> PhaseTransitionMetrics:
        """Compute current phase transition metrics."""
        step = self._steps[-1] if self._steps else 0
        values = list(self._history)
        
        if len(values) < 10:
            return PhaseTransitionMetrics(
                step=step,
                timestamp=time.time(),
                autocorr_time=None,
                autocorr_1=None,
                variance=0.0,
                variance_ratio=None,
                susceptibility=None,
                is_stationary=True,
                trend=None,
                phase="unknown",
            )
        
        # Autocorrelation
        acf = compute_autocorrelation(values, max_lag=min(50, len(values) // 2))
        autocorr_time = compute_integrated_autocorr_time(acf)
        autocorr_1 = acf[1] if len(acf) > 1 else None
        
        # Variance
        mean = sum(values) / len(values)
        variance = sum((v - mean)**2 for v in values) / len(values)
        variance_ratio = compute_variance_ratio(values)
        
        # Trend
        trend, trend_r2 = compute_trend(values)
        
        # Stationarity
        is_stationary = detect_stationarity(values)
        
        # Phase classification
        phase = classify_phase(autocorr_time, variance_ratio, is_stationary, trend_r2)
        
        return PhaseTransitionMetrics(
            step=step,
            timestamp=time.time(),
            autocorr_time=autocorr_time,
            autocorr_1=autocorr_1,
            variance=variance,
            variance_ratio=variance_ratio,
            susceptibility=None,  # Would need perturbation experiment
            is_stationary=is_stationary,
            trend=trend,
            phase=phase,
        )


@dataclass
class GrokDetector:
    """
    Specialized detector for grokking phenomenon.
    
    Grokking = delayed generalization: train loss decreases early,
    but test accuracy suddenly improves much later.
    
    Signatures:
    - Long plateau in test metrics
    - Sudden improvement after plateau
    - Train/test gap dynamics
    """
    
    window_size: int = 100
    plateau_threshold: float = 0.01  # Max change during plateau
    improvement_threshold: float = 0.1  # Min change for "sudden improvement"
    
    _train_history: deque = field(default_factory=deque)
    _test_history: deque = field(default_factory=deque)
    _steps: deque = field(default_factory=deque)
    
    def __post_init__(self):
        self._train_history = deque(maxlen=self.window_size * 2)
        self._test_history = deque(maxlen=self.window_size * 2)
        self._steps = deque(maxlen=self.window_size * 2)
    
    def add(self, step: int, train_metric: float, test_metric: float) -> None:
        """Record train and test metrics."""
        self._train_history.append(train_metric)
        self._test_history.append(test_metric)
        self._steps.append(step)
    
    def detect_grokking(self) -> Dict[str, Any]:
        """
        Detect grokking signatures.
        
        Returns dict with:
        - phase: "pre_grok", "grokking", "post_grok", "no_grok", "unknown"
        - plateau_length: steps in plateau (if detected)
        - generalization_gap: current train/test gap
        """
        if len(self._test_history) < self.window_size:
            return {"phase": "unknown", "plateau_length": None, "generalization_gap": None}
        
        test_values = list(self._test_history)
        train_values = list(self._train_history)
        
        # Generalization gap
        gen_gap = abs(train_values[-1] - test_values[-1])
        
        # Check for plateau in test metrics
        recent_test = test_values[-self.window_size:]
        test_range = max(recent_test) - min(recent_test)
        in_plateau = test_range < self.plateau_threshold
        
        # Check for recent sudden improvement
        if len(test_values) > self.window_size:
            old_test = test_values[-self.window_size * 2:-self.window_size]
            recent_mean = sum(recent_test) / len(recent_test)
            old_mean = sum(old_test) / len(old_test)
            improvement = recent_mean - old_mean  # Assumes higher is better
            sudden_improvement = improvement > self.improvement_threshold
        else:
            sudden_improvement = False
        
        # Phase classification
        if sudden_improvement:
            phase = "grokking" if in_plateau else "post_grok"
        elif in_plateau and gen_gap > 0.1:
            phase = "pre_grok"
        else:
            phase = "no_grok"
        
        return {
            "phase": phase,
            "plateau_length": self.window_size if in_plateau else None,
            "generalization_gap": gen_gap,
            "test_improvement": improvement if len(test_values) > self.window_size else None,
        }
