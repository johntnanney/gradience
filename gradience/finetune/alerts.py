"""
Fine-Tuning Alerts - Automated warnings and recommendations

Translates spectral signals into actionable alerts:
- Overfitting warnings (before validation loss shows it)
- Backbone destabilization (catastrophic forgetting risk)
- Training saturation (diminishing returns)
- Learning rate suggestions
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Callable, Dict
import time


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    # ===========================================
    # FINE-TUNING PATHOLOGIES
    # ===========================================
    
    # Model forgetting general capabilities
    # Signal: Stable rank / effective rank of model weights DROPS
    CAPACITY_COLLAPSE = "capacity_collapse"
    
    # Feature distortion from aggressive updates
    # Signal: σ_max (spectral norm) SPIKES
    FEATURE_DISTORTION = "feature_distortion"
    
    # Model wandering too far from base
    # Signal: ||W_t - W_0||_* (nuclear norm of delta) growing too fast
    EXCESSIVE_DRIFT = "excessive_drift"
    
    # ===========================================
    # CAPACITY AND OVERFITTING
    # ===========================================
    RANK_GROWING_FAST = "rank_growing_fast"
    RANK_EXCEEDS_DATA_CAPACITY = "rank_exceeds_data_capacity"
    
    # ===========================================
    # BACKBONE STABILITY
    # ===========================================
    BACKBONE_DESTABILIZING = "backbone_destabilizing"
    EARLY_LAYER_CHANGING = "early_layer_changing"
    
    # ===========================================
    # TRAINING PROGRESS
    # ===========================================
    TRAINING_SATURATED = "training_saturated"
    NO_PROGRESS = "no_progress"
    
    # ===========================================
    # LEARNING RATE
    # ===========================================
    LR_TOO_HIGH = "lr_too_high"
    LR_TOO_LOW = "lr_too_low"
    
    # ===========================================
    # LORA SPECIFIC
    # ===========================================
    LORA_EFFECTIVE_RANK_LOW = "lora_effective_rank_low"  # Adapter not using its rank
    LORA_ADAPTER_UNSTABLE = "lora_adapter_unstable"      # A/B matrices ill-conditioned
    LORA_SCALING_SUBOPTIMAL = "lora_scaling_suboptimal"  # Alpha/rank ratio issues
    
    # ===========================================
    # RECOMMENDATIONS
    # ===========================================
    CONSIDER_EARLY_STOPPING = "consider_early_stopping"
    CONSIDER_LORA = "consider_lora"
    CHECKPOINT_RECOMMENDED = "checkpoint_recommended"


@dataclass
class Alert:
    """A single alert."""
    type: AlertType
    severity: AlertSeverity
    step: int
    message: str
    details: dict
    recommendation: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __str__(self):
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}[self.severity.value]
        return f"{icon} [{self.severity.value.upper()}] Step {self.step}: {self.message}"


class AlertCondition:
    """Base class for alert conditions."""
    
    def __init__(self, alert_type: AlertType, severity: AlertSeverity):
        self.alert_type = alert_type
        self.severity = severity
        self.last_triggered_step: Optional[int] = None
        self.cooldown_steps: int = 100  # Don't re-trigger for N steps
        
    def check(self, state: "FineTuneState") -> Optional[Alert]:
        """Check if condition is met. Override in subclasses."""
        raise NotImplementedError
    
    def should_check(self, step: int) -> bool:
        """Check if we're past cooldown."""
        if self.last_triggered_step is None:
            return True
        return step - self.last_triggered_step > self.cooldown_steps


@dataclass
class FineTuneState:
    """Current state of fine-tuning for alert evaluation."""
    step: int
    
    # From DeltaTracker
    delta_rank: float
    delta_rank_history: List[float]
    backbone_stable: bool
    backbone_kappa_changes: dict  # layer_name -> change_pct
    
    # From training
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Dataset info
    num_train_examples: Optional[int] = None
    
    # History
    val_loss_history: List[float] = None
    train_loss_history: List[float] = None
    
    # === NEW: For pathology detection ===
    
    # Model rank history (not delta rank - the actual model's rank)
    # Used for Lobotomy detection
    model_rank_history: List[float] = None
    
    # Spectral norm history (σ_max)
    # Used for Brain Damage detection
    sigma_max_history: List[float] = None
    
    # Delta magnitude history (||W - W_0||)
    # Used for Drift detection
    delta_magnitude_history: List[float] = None
    
    # LoRA metrics (if using LoRA)
    # Dict of adapter_name -> {nominal_rank, effective_rank, kappa_A, kappa_B}
    lora_metrics: Dict[str, Dict] = None
    

# ============================================================
# FINE-TUNING PATHOLOGY CONDITIONS
# ============================================================

class CapacityCollapseCondition(AlertCondition):
    """
    Detect capacity collapse - model losing representational capacity.
    
    Signal: Effective rank of model weights DROPS during fine-tuning.
    This indicates the model is collapsing its representational capacity,
    potentially forgetting capabilities outside the fine-tuning distribution.
    
    This is dangerous because loss may still look good while the model
    is losing general capability.
    """
    
    def __init__(self, drop_threshold: float = 0.10):
        super().__init__(AlertType.CAPACITY_COLLAPSE, AlertSeverity.CRITICAL)
        self.drop_threshold = drop_threshold  # 10% drop is concerning
        self.baseline_rank: Optional[float] = None
        self.window = 5
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
        
        # Need model_rank_history in state (current model rank, not delta rank)
        if not hasattr(state, 'model_rank_history') or state.model_rank_history is None:
            return None
        if len(state.model_rank_history) < self.window * 2:
            return None
        
        # Establish baseline from early training
        if self.baseline_rank is None:
            self.baseline_rank = sum(state.model_rank_history[:self.window]) / self.window
        
        # Check recent rank
        recent_rank = sum(state.model_rank_history[-self.window:]) / self.window
        
        drop = (self.baseline_rank - recent_rank) / (self.baseline_rank + 1e-10)
        
        if drop > self.drop_threshold:
            self.last_triggered_step = state.step
            return Alert(
                type=self.alert_type,
                severity=self.severity,
                step=state.step,
                message=f"Model capacity declining: effective rank dropped {drop:.0%}",
                details={
                    'baseline_rank': self.baseline_rank,
                    'current_rank': recent_rank,
                    'drop_pct': drop,
                },
                recommendation=(
                    "Model is losing representational capacity. This may indicate "
                    "the model is forgetting general capabilities.\n"
                    "Consider:\n"
                    "  - Reducing learning rate\n"
                    "  - Adding regularization (weight decay)\n"
                    "  - Using LoRA instead of full fine-tuning\n"
                    "  - Checking if fine-tuning data is too narrow\n"
                    "  - Rolling back to earlier checkpoint"
                ),
            )
        return None


class FeatureDistortionCondition(AlertCondition):
    """
    Detect feature distortion from aggressive updates.
    
    Signal: σ_max (spectral norm) SPIKES during fine-tuning.
    This indicates pretrained features are being distorted by
    updates that are too large.
    """
    
    def __init__(self, spike_threshold: float = 0.30):
        super().__init__(AlertType.FEATURE_DISTORTION, AlertSeverity.CRITICAL)
        self.spike_threshold = spike_threshold  # 30% spike is concerning
        self.baseline_sigma_max: Optional[float] = None
        self.window = 5
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
        
        if not hasattr(state, 'sigma_max_history') or state.sigma_max_history is None:
            return None
        if len(state.sigma_max_history) < self.window * 2:
            return None
        
        # Establish baseline
        if self.baseline_sigma_max is None:
            self.baseline_sigma_max = sum(state.sigma_max_history[:self.window]) / self.window
        
        # Check for spike
        recent_sigma = sum(state.sigma_max_history[-self.window:]) / self.window
        
        spike = (recent_sigma - self.baseline_sigma_max) / (self.baseline_sigma_max + 1e-10)
        
        if spike > self.spike_threshold:
            self.last_triggered_step = state.step
            return Alert(
                type=self.alert_type,
                severity=self.severity,
                step=state.step,
                message=f"Feature distortion detected: spectral norm increased {spike:.0%}",
                details={
                    'baseline_sigma_max': self.baseline_sigma_max,
                    'current_sigma_max': recent_sigma,
                    'spike_pct': spike,
                },
                recommendation=(
                    "Pretrained features are being distorted by large updates.\n"
                    "Consider:\n"
                    "  - Reducing learning rate (try 0.5x or 0.1x)\n"
                    "  - Adding gradient clipping\n"
                    "  - Using layer-wise learning rates (lower for early layers)\n"
                    "  - Rolling back to earlier checkpoint"
                ),
            )
        return None


class ExcessiveDriftCondition(AlertCondition):
    """
    Detect excessive drift from pretrained weights.
    
    Signal: ||W_t - W_0|| growing faster than expected.
    This indicates the model is diverging too far from the pretrained
    initialization, potentially losing the benefits of pretraining.
    """
    
    def __init__(self, drift_rate_threshold: float = 0.50):
        super().__init__(AlertType.EXCESSIVE_DRIFT, AlertSeverity.WARNING)
        self.drift_rate_threshold = drift_rate_threshold  # 50% growth rate
        self.window = 5
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
        
        # Use delta_magnitude_history if available, else delta_rank_history
        if hasattr(state, 'delta_magnitude_history') and state.delta_magnitude_history:
            history = state.delta_magnitude_history
        else:
            history = state.delta_rank_history
            
        if len(history) < self.window * 2:
            return None
        
        recent = sum(history[-self.window:]) / self.window
        earlier = sum(history[-self.window*2:-self.window]) / self.window
        
        if earlier < 1e-10:
            return None
        
        growth_rate = (recent - earlier) / earlier
        
        if growth_rate > self.drift_rate_threshold:
            self.last_triggered_step = state.step
            return Alert(
                type=self.alert_type,
                severity=self.severity,
                step=state.step,
                message=f"Model diverging from pretrained weights: {growth_rate:.0%} drift rate",
                details={
                    'recent_distance': recent,
                    'earlier_distance': earlier,
                    'growth_rate': growth_rate,
                },
                recommendation=(
                    "Model is drifting away from pretrained weights quickly.\n"
                    "This may reduce the benefits of pretraining.\n"
                    "Consider:\n"
                    "  - Reducing learning rate\n"
                    "  - Using LoRA to constrain updates\n"
                    "  - Adding weight decay toward base weights\n"
                    "  - Checking if training duration is appropriate"
                ),
            )
        return None


# ============================================================
# LORA-SPECIFIC CONDITIONS
# ============================================================

class LoRAEffectiveRankCondition(AlertCondition):
    """
    Detect when LoRA adapter isn't using its full rank.
    
    Users configure rank-64 but if A or B becomes ill-conditioned,
    they effectively have rank-1. They're wasting VRAM.
    """
    
    def __init__(self, effective_rank_ratio: float = 0.25):
        super().__init__(AlertType.LORA_EFFECTIVE_RANK_LOW, AlertSeverity.WARNING)
        self.effective_rank_ratio = effective_rank_ratio  # Below 25% of nominal rank
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
        
        if not hasattr(state, 'lora_metrics') or state.lora_metrics is None:
            return None
        
        for adapter_name, metrics in state.lora_metrics.items():
            nominal_rank = metrics.get('nominal_rank', 0)
            effective_rank = metrics.get('effective_rank', 0)
            
            if nominal_rank < 1:
                continue
            
            ratio = effective_rank / nominal_rank
            
            if ratio < self.effective_rank_ratio:
                self.last_triggered_step = state.step
                return Alert(
                    type=self.alert_type,
                    severity=self.severity,
                    step=state.step,
                    message=f"LoRA adapter '{adapter_name}' using {ratio:.0%} of configured rank-{nominal_rank}",
                    details={
                        'adapter': adapter_name,
                        'nominal_rank': nominal_rank,
                        'effective_rank': effective_rank,
                        'ratio': ratio,
                    },
                    recommendation=(
                        f"Your rank-{nominal_rank} adapter has effective rank ~{effective_rank:.0f}.\n"
                        "Consider:\n"
                        f"  - Reducing LoRA rank to {max(4, int(effective_rank*2))} (same quality, less memory)\n"
                        "  - Adjusting lora_alpha for better conditioning\n"
                        "  - Increasing learning rate if adapter is under-training"
                    ),
                )
        return None


class LoRADominanceCondition(AlertCondition):
    """
    Detect when LoRA adapter is overwhelming the base model.
    
    In LoRA, the "gravity" is the frozen base model. If the adapter
    becomes too large relative to base, it's overwriting rather than
    fine-tuning. This is LoRA's equivalent of the muon ratio.
    
    ρ_lora = σ_max(BA) / σ_max(W_base)
    """
    
    def __init__(
        self, 
        warning_threshold: float = 0.30,
        critical_threshold: float = 0.50,
    ):
        super().__init__(AlertType.LORA_SCALING_SUBOPTIMAL, AlertSeverity.WARNING)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
        
        if not hasattr(state, 'lora_structural') or state.lora_structural is None:
            return None
        
        structural = state.lora_structural
        dominance = structural.get('scaled_mean_dominance', 0)
        
        if dominance > self.critical_threshold:
            self.last_triggered_step = state.step
            return Alert(
                type=self.alert_type,
                severity=AlertSeverity.CRITICAL,
                step=state.step,
                message=f"LoRA adapter dominating base model: {dominance:.0%} dominance",
                details={
                    'scaled_dominance': dominance,
                    'threshold': self.critical_threshold,
                    'layers_dominating': structural.get('layers_dominating', []),
                },
                recommendation=(
                    "CRITICAL: Adapter is overwhelming the base model.\n"
                    "You're no longer fine-tuning — you're overwriting.\n"
                    "Consider:\n"
                    "  - Reducing learning rate significantly\n"
                    "  - Reducing lora_alpha\n"
                    "  - Using a lower rank\n"
                    "  - Adding regularization\n"
                    "  - Rolling back to earlier checkpoint"
                ),
            )
        
        elif dominance > self.warning_threshold:
            self.last_triggered_step = state.step
            return Alert(
                type=self.alert_type,
                severity=AlertSeverity.WARNING,
                step=state.step,
                message=f"LoRA adapter significantly modifying base: {dominance:.0%} dominance",
                details={
                    'scaled_dominance': dominance,
                    'threshold': self.warning_threshold,
                },
                recommendation=(
                    "Adapter is making significant modifications to base model.\n"
                    "This may be intentional for large domain shifts.\n"
                    "If unexpected, consider:\n"
                    "  - Reducing learning rate\n"
                    "  - Monitoring for overfitting\n"
                    "  - Checking validation metrics"
                ),
            )
        
        return None


# ============================================================
# Original conditions (kept for compatibility)
# ============================================================

class RankGrowingFastCondition(AlertCondition):
    """Alert when ΔW rank is growing faster than expected."""
    
    def __init__(self, growth_threshold: float = 0.5):
        super().__init__(AlertType.RANK_GROWING_FAST, AlertSeverity.WARNING)
        self.growth_threshold = growth_threshold  # 50% growth in window
        self.window = 10
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
            
        history = state.delta_rank_history
        if len(history) < self.window * 2:
            return None
        
        recent = history[-self.window:]
        earlier = history[-self.window*2:-self.window]
        
        recent_mean = sum(recent) / len(recent)
        earlier_mean = sum(earlier) / len(earlier)
        
        if earlier_mean < 1e-6:
            return None
            
        growth = (recent_mean - earlier_mean) / earlier_mean
        
        if growth > self.growth_threshold:
            self.last_triggered_step = state.step
            return Alert(
                type=self.alert_type,
                severity=self.severity,
                step=state.step,
                message=f"ΔW rank growing rapidly ({growth:.0%} increase)",
                details={
                    'recent_rank': recent_mean,
                    'earlier_rank': earlier_mean,
                    'growth_rate': growth,
                },
                recommendation=(
                    "This may indicate overfitting. Consider:\n"
                    "  - Reducing learning rate\n"
                    "  - Adding regularization (weight decay, dropout)\n"
                    "  - Early stopping if validation metrics plateau"
                ),
            )
        return None


class RankExceedsCapacityCondition(AlertCondition):
    """Alert when effective DoF exceeds safe level for dataset size."""
    
    def __init__(self, samples_per_dof: int = 10):
        super().__init__(AlertType.RANK_EXCEEDS_DATA_CAPACITY, AlertSeverity.WARNING)
        self.samples_per_dof = samples_per_dof
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
            
        if state.num_train_examples is None:
            return None
        
        # Rough estimate: effective DoF ~ delta_rank * num_layers
        # This is hand-wavy but directionally correct
        num_layers = len(state.backbone_kappa_changes)
        effective_dof = state.delta_rank * num_layers
        
        safe_dof = state.num_train_examples / self.samples_per_dof
        
        if effective_dof > safe_dof:
            self.last_triggered_step = state.step
            return Alert(
                type=self.alert_type,
                severity=self.severity,
                step=state.step,
                message=f"Effective capacity ({effective_dof:.0f}) exceeds safe level for dataset ({safe_dof:.0f})",
                details={
                    'effective_dof': effective_dof,
                    'safe_dof': safe_dof,
                    'num_examples': state.num_train_examples,
                    'delta_rank': state.delta_rank,
                },
                recommendation=(
                    "Model may be overfitting due to limited data. Consider:\n"
                    "  - Using LoRA with smaller rank\n"
                    "  - Increasing regularization\n"
                    "  - Data augmentation\n"
                    "  - Early stopping"
                ),
            )
        return None


class BackboneDestabilizingCondition(AlertCondition):
    """Alert when backbone layers are changing too much."""
    
    def __init__(self, threshold: float = 0.20):
        super().__init__(AlertType.BACKBONE_DESTABILIZING, AlertSeverity.CRITICAL)
        self.threshold = threshold
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
            
        if state.backbone_stable:
            return None
        
        # Find the worst offender
        worst_layer = None
        worst_change = 0
        for layer, change in state.backbone_kappa_changes.items():
            if abs(change) > abs(worst_change):
                worst_change = change
                worst_layer = layer
        
        self.last_triggered_step = state.step
        return Alert(
            type=self.alert_type,
            severity=self.severity,
            step=state.step,
            message=f"Backbone destabilizing: {worst_layer} changed {worst_change:.0%}",
            details={
                'worst_layer': worst_layer,
                'worst_change': worst_change,
                'all_changes': state.backbone_kappa_changes,
            },
            recommendation=(
                "Backbone layers are changing significantly, risking catastrophic forgetting.\n"
                "Consider:\n"
                "  - Reducing learning rate immediately\n"
                "  - Freezing early layers\n"
                "  - Rolling back to earlier checkpoint\n"
                "  - Using LoRA instead of full fine-tuning"
            ),
        )


class TrainingSaturatedCondition(AlertCondition):
    """Alert when training appears to have saturated."""
    
    def __init__(self, window: int = 5, threshold: float = 0.05):
        super().__init__(AlertType.TRAINING_SATURATED, AlertSeverity.INFO)
        self.window = window
        self.threshold = threshold
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
            
        history = state.delta_rank_history
        if len(history) < self.window * 2:
            return None
        
        recent = history[-self.window:]
        earlier = history[-self.window*2:-self.window]
        
        recent_mean = sum(recent) / len(recent)
        earlier_mean = sum(earlier) / len(earlier)
        
        if earlier_mean < 1e-6:
            return None
        
        growth = (recent_mean - earlier_mean) / earlier_mean
        
        if abs(growth) < self.threshold:
            self.last_triggered_step = state.step
            return Alert(
                type=self.alert_type,
                severity=self.severity,
                step=state.step,
                message=f"Training appears saturated (ΔW rank growth < {self.threshold:.0%})",
                details={
                    'recent_rank': recent_mean,
                    'earlier_rank': earlier_mean,
                    'growth_rate': growth,
                },
                recommendation=(
                    "The model's adaptation has plateaued. This could mean:\n"
                    "  - Training is complete (check validation metrics)\n"
                    "  - Learning rate is too low to make progress\n"
                    "Consider stopping if validation metrics are satisfactory."
                ),
            )
        return None


class ConsiderLoRACondition(AlertCondition):
    """Suggest LoRA when delta rank is consistently low."""
    
    def __init__(self, rank_threshold: float = 32):
        super().__init__(AlertType.CONSIDER_LORA, AlertSeverity.INFO)
        self.rank_threshold = rank_threshold
        self.min_steps = 200  # Don't suggest too early
    
    def check(self, state: FineTuneState) -> Optional[Alert]:
        if not self.should_check(state.step):
            return None
        
        if state.step < self.min_steps:
            return None
        
        if state.delta_rank < self.rank_threshold:
            self.last_triggered_step = state.step
            
            suggested_r = max(4, int(state.delta_rank * 1.5))
            # Round up to power of 2
            suggested_r = 2 ** (suggested_r - 1).bit_length()
            
            return Alert(
                type=self.alert_type,
                severity=self.severity,
                step=state.step,
                message=f"ΔW rank ({state.delta_rank:.1f}) suggests LoRA would be effective",
                details={
                    'delta_rank': state.delta_rank,
                    'suggested_lora_rank': suggested_r,
                },
                recommendation=(
                    f"Your fine-tuning update appears low-rank. Consider using LoRA with r={suggested_r}.\n"
                    f"This could reduce memory usage by ~{90}% with minimal accuracy loss."
                ),
            )
        return None


# ============================================================
# Alert Manager
# ============================================================

class FineTuneAlertManager:
    """
    Manages alert conditions and tracks alert history.
    
    Usage:
        manager = FineTuneAlertManager()
        
        for step in training:
            state = FineTuneState(
                step=step,
                delta_rank=tracker.current_rank,
                delta_rank_history=tracker.get_rank_trajectory(),
                backbone_stable=tracker.is_backbone_stable(),
                backbone_kappa_changes=tracker.get_backbone_changes(),
                num_train_examples=len(train_dataset),
            )
            
            alerts = manager.check(state)
            for alert in alerts:
                print(alert)
    """
    
    def __init__(self, conditions: List[AlertCondition] = None):
        if conditions is None:
            conditions = self._default_conditions()
        
        self.conditions = conditions
        self.alert_history: List[Alert] = []
        self.callbacks: List[Callable[[Alert], None]] = []
    
    def _default_conditions(self) -> List[AlertCondition]:
        """Create default set of alert conditions."""
        return [
            # Fine-tuning pathologies
            CapacityCollapseCondition(),     # Rank drop - losing capacity
            FeatureDistortionCondition(),    # σ_max spike - features distorted
            ExcessiveDriftCondition(),       # Diverging from base
            
            # LoRA-specific
            LoRAEffectiveRankCondition(),    # Adapter not using its rank
            
            # Capacity and overfitting
            RankGrowingFastCondition(),
            RankExceedsCapacityCondition(),
            
            # Backbone stability
            BackboneDestabilizingCondition(),
            
            # Training progress
            TrainingSaturatedCondition(),
            
            # Recommendations
            ConsiderLoRACondition(),
        ]
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Add callback to be called when alerts are triggered."""
        self.callbacks.append(callback)
    
    def check(self, state: FineTuneState) -> List[Alert]:
        """Check all conditions and return any triggered alerts."""
        triggered = []
        
        for condition in self.conditions:
            alert = condition.check(state)
            if alert is not None:
                triggered.append(alert)
                self.alert_history.append(alert)
                
                # Call callbacks
                for callback in self.callbacks:
                    callback(alert)
        
        return triggered
    
    def get_alerts(
        self, 
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
    ) -> List[Alert]:
        """Get alert history, optionally filtered."""
        alerts = self.alert_history
        
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type is not None:
            alerts = [a for a in alerts if a.type == alert_type]
        
        return alerts
    
    def summary(self) -> dict:
        """Get summary of alerts."""
        by_severity = {s: 0 for s in AlertSeverity}
        by_type = {}
        
        for alert in self.alert_history:
            by_severity[alert.severity] += 1
            by_type[alert.type] = by_type.get(alert.type, 0) + 1
        
        return {
            'total_alerts': len(self.alert_history),
            'by_severity': {s.value: c for s, c in by_severity.items()},
            'by_type': {t.value: c for t, c in by_type.items()},
            'critical_alerts': self.get_alerts(severity=AlertSeverity.CRITICAL),
        }


# ============================================================
# Convenience function
# ============================================================

def print_alert_callback(alert: Alert):
    """Simple callback that prints alerts."""
    print(alert)
    if alert.severity == AlertSeverity.CRITICAL:
        print(f"   💡 {alert.recommendation}")
