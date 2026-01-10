"""
LoRA Analyzer - Spectral analysis for LoRA adapters

Monitors the health and efficiency of LoRA (Low-Rank Adaptation) fine-tuning:
- Effective rank utilization (are you using the rank you paid for?)
- Adapter conditioning (are A/B matrices stable?)
- Adapter dominance (is the adapter overwhelming the base model?)

Key insight: A rank-64 LoRA adapter can collapse to effective rank-1
if training dynamics are poor. This wastes VRAM and hurts quality.

Structural insight for LoRA:
- In pretraining, ρ = λ × σ_max measures expansion vs regularization
- In LoRA, the "gravity" is the frozen base model itself
- ρ_lora = σ_max(BA) / σ_max(W_base) measures adapter dominance
- Healthy fine-tuning: adapter is small perturbation (ρ_lora < 0.1)
- Overwriting base: adapter dominates (ρ_lora > 0.3)

Usage:
    from gradience.finetune.lora import LoRAAnalyzer
    
    analyzer = LoRAAnalyzer(model, base_model)
    
    # During training
    metrics = analyzer.analyze()
    print(f"Rank utilization: {metrics.overall_utilization:.0%}")
    
    structural = analyzer.analyze_structural()
    print(f"Adapter dominance: {structural.mean_dominance:.1%}")
    
    # Get recommendations
    suggestion = analyzer.suggest_rank()
    print(f"Suggested rank: {suggestion['recommended_rank']}")
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


@dataclass
class LoRALayerMetrics:
    """Metrics for a single LoRA adapter layer."""
    
    # Identity
    name: str
    layer_type: str  # "attention_q", "attention_v", "ffn_up", etc.
    
    # Configuration
    nominal_rank: int
    alpha: float
    scaling: float  # α/r
    
    # A matrix (r × k, the "up" projection)
    a_shape: Tuple[int, int]
    a_kappa: float
    a_effective_rank: float
    a_sigma_max: float
    
    # B matrix (d × r, the "down" projection)
    b_shape: Tuple[int, int]
    b_kappa: float
    b_effective_rank: float
    b_sigma_max: float
    
    # Product BA
    ba_effective_rank: float
    ba_sigma_max: float
    ba_frobenius: float
    
    # Derived
    rank_utilization: float  # ba_effective_rank / nominal_rank
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'layer_type': self.layer_type,
            'nominal_rank': self.nominal_rank,
            'alpha': self.alpha,
            'rank_utilization': self.rank_utilization,
            'ba_effective_rank': self.ba_effective_rank,
            'a_kappa': self.a_kappa,
            'b_kappa': self.b_kappa,
        }


@dataclass
class LoRAModelMetrics:
    """Aggregate metrics for all LoRA adapters in a model."""
    
    # Per-layer metrics
    layers: Dict[str, LoRALayerMetrics]
    
    # Aggregates
    num_adapters: int
    total_nominal_rank: int
    total_effective_rank: float
    overall_utilization: float
    
    # Worst cases (for alerts)
    min_utilization: float
    min_utilization_layer: str
    max_kappa: float
    max_kappa_layer: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_adapters': self.num_adapters,
            'overall_utilization': self.overall_utilization,
            'min_utilization': self.min_utilization,
            'min_utilization_layer': self.min_utilization_layer,
            'max_kappa': self.max_kappa,
            'layers': {k: v.to_dict() for k, v in self.layers.items()},
        }


@dataclass
class LoRAStructuralMetrics:
    """
    Structural metrics for LoRA adapters.
    
    In LoRA, the "gravity" is the frozen base model.
    The adapter should be a small perturbation, not a replacement.
    
    Key metric: dominance_ratio = σ_max(BA) / σ_max(W_base)
    
    Interpretation:
        ρ_lora < 0.05  →  Minimal modification (very conservative)
        ρ_lora ~ 0.10  →  Healthy fine-tuning
        ρ_lora > 0.30  →  Significant modification (warning)
        ρ_lora > 0.50  →  Adapter dominating (critical)
    """
    
    # Adapter dominance (LoRA's version of muon ratio)
    mean_dominance: float           # Mean σ_max(BA) / σ_max(W_base)
    max_dominance: float            # Worst case
    scaled_mean_dominance: float    # Mean (α/r) × σ_max(BA) / σ_max(W_base)
    
    # Per-layer breakdown
    per_layer_dominance: Dict[str, float]
    per_layer_scaled_dominance: Dict[str, float]
    
    # Conditioning health (gradient flow stability)
    adapter_health: float           # 1 / max(κ(A), κ(B)) across all layers
    worst_kappa_A: float
    worst_kappa_A_layer: str
    worst_kappa_B: float
    worst_kappa_B_layer: str
    
    # Alerts
    is_dominating: bool             # Any layer with dominance > 0.3
    is_ill_conditioned: bool        # Any κ(A) or κ(B) > 1000
    layers_dominating: List[str]    # Which layers are dominating
    layers_ill_conditioned: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_dominance': self.mean_dominance,
            'max_dominance': self.max_dominance,
            'scaled_mean_dominance': self.scaled_mean_dominance,
            'adapter_health': self.adapter_health,
            'worst_kappa_A': self.worst_kappa_A,
            'worst_kappa_B': self.worst_kappa_B,
            'is_dominating': self.is_dominating,
            'is_ill_conditioned': self.is_ill_conditioned,
            'layers_dominating': self.layers_dominating,
            'layers_ill_conditioned': self.layers_ill_conditioned,
        }
    
    def status_summary(self) -> str:
        """Get a one-line status summary."""
        issues = []
        if self.is_dominating:
            issues.append(f"dominating({len(self.layers_dominating)} layers)")
        if self.is_ill_conditioned:
            issues.append(f"ill-conditioned({len(self.layers_ill_conditioned)} layers)")
        
        if not issues:
            return f"healthy (dominance={self.mean_dominance:.1%})"
        else:
            return f"warning: {', '.join(issues)}"


def _compute_effective_rank(S: torch.Tensor) -> float:
    """Compute effective rank from singular values using entropy."""
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    
    S_norm = S / S.sum()
    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
    return math.exp(entropy)


def _compute_kappa(S: torch.Tensor) -> float:
    """Compute condition number from singular values."""
    S = S[S > 1e-10]
    if len(S) < 2:
        return 1.0
    return (S[0] / S[-1]).item()


class LoRAAnalyzer:
    """
    Analyze LoRA adapter spectral properties.
    
    Detects:
    - Rank underutilization (paying for rank-64, using rank-5)
    - Ill-conditioned adapters (κ(A) or κ(B) too high)
    - Adapter dominance (adapter overwhelming base model)
    
    Works with:
    - HuggingFace PEFT LoRA
    - Custom LoRA implementations
    - Any model with lora_A/lora_B weight patterns
    
    Usage:
        # Basic usage
        analyzer = LoRAAnalyzer(model)
        metrics = analyzer.analyze()
        
        # With structural analysis (requires base model reference)
        analyzer = LoRAAnalyzer(model, base_model=pretrained_model)
        structural = analyzer.analyze_structural(lora_alpha=16, lora_rank=16)
        print(f"Adapter dominance: {structural.mean_dominance:.1%}")
    """
    
    # Thresholds for structural alerts
    DOMINANCE_WARNING = 0.30    # 30% of base σ_max
    DOMINANCE_CRITICAL = 0.50   # 50% of base σ_max
    KAPPA_WARNING = 1000        # Condition number threshold
    
    def __init__(
        self, 
        model: nn.Module,
        base_model: Optional[nn.Module] = None,
        adapter_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize analyzer.
        
        Args:
            model: Model with LoRA adapters
            base_model: Original model before LoRA (for structural analysis)
                       If not provided, structural analysis will estimate from
                       the combined weights.
            adapter_patterns: Patterns to match LoRA weight names
                             (default: ["lora_A", "lora_B"])
        """
        self.model = model
        self.adapter_patterns = adapter_patterns or ["lora_A", "lora_B"]
        
        # Discover adapters
        self.adapters = self._find_adapters()
        
        # Store base model σ_max for structural analysis
        self.base_sigma_max: Dict[str, float] = {}
        if base_model is not None:
            self._cache_base_sigma_max(base_model)
        
        # History for tracking
        self.history: List[LoRAModelMetrics] = []
        self.structural_history: List[LoRAStructuralMetrics] = []
    
    def _cache_base_sigma_max(self, base_model: nn.Module):
        """Cache σ_max values from base model for structural analysis."""
        for name, param in base_model.named_parameters():
            if param.dim() < 2:
                continue
            if min(param.shape) < 10:
                continue
            
            # Map to adapter base name
            # Try to match with our adapter names
            base_name = self._param_to_adapter_name(name)
            if base_name is None:
                continue
            
            with torch.no_grad():
                W = param.float()
                if W.dim() > 2:
                    W = W.view(W.size(0), -1)
                
                try:
                    S = torch.linalg.svdvals(W)
                    self.base_sigma_max[base_name] = S[0].item()
                except Exception:
                    continue
    
    def _param_to_adapter_name(self, param_name: str) -> Optional[str]:
        """Try to map a parameter name to an adapter base name."""
        # Common patterns to remove
        suffixes = ['.weight', '.bias']
        name = param_name
        for suffix in suffixes:
            name = name.replace(suffix, '')
        
        # Check if this matches any adapter
        for adapter_name in self.adapters:
            # Check if the adapter name is contained in or matches the param name
            if adapter_name in name or name in adapter_name:
                return adapter_name
            
            # Also check by removing common prefixes
            clean_adapter = adapter_name.replace('.default', '').replace('.lora', '')
            clean_param = name.replace('.default', '').replace('.lora', '')
            if clean_adapter in clean_param or clean_param in clean_adapter:
                return adapter_name
        
        return None
    
    def set_base_sigma_max(self, base_sigma_max: Dict[str, float]):
        """
        Manually set base σ_max values.
        
        Useful when base model is no longer in memory.
        
        Args:
            base_sigma_max: Dict mapping adapter names to σ_max values
        """
        self.base_sigma_max = base_sigma_max
    
    def analyze_structural(
        self, 
        lora_alpha: float,
        lora_rank: int,
    ) -> LoRAStructuralMetrics:
        """
        Analyze structural properties of LoRA adapters.
        
        Computes the "adapter dominance ratio" - LoRA's equivalent of the
        muon ratio. This measures how much the adapter is modifying the
        base model's behavior.
        
        Args:
            lora_alpha: LoRA alpha scaling parameter
            lora_rank: LoRA rank
            
        Returns:
            LoRAStructuralMetrics with dominance ratios and health metrics
        """
        scaling = lora_alpha / lora_rank
        
        per_layer_dominance = {}
        per_layer_scaled = {}
        kappa_As = {}
        kappa_Bs = {}
        
        for name, adapter in self.adapters.items():
            with torch.no_grad():
                A = adapter['A'].float()
                B = adapter['B'].float()
                
                # Compute BA
                BA = B @ A
                S_BA = torch.linalg.svdvals(BA)
                sigma_max_BA = S_BA[0].item() if len(S_BA) > 0 else 0.0
                
                # Compute κ(A) and κ(B)
                S_A = torch.linalg.svdvals(A)
                S_B = torch.linalg.svdvals(B)
                kappa_A = _compute_kappa(S_A)
                kappa_B = _compute_kappa(S_B)
                
                kappa_As[name] = kappa_A
                kappa_Bs[name] = kappa_B
                
                # Compute dominance ratio
                if name in self.base_sigma_max and self.base_sigma_max[name] > 0:
                    base_sigma = self.base_sigma_max[name]
                    dominance = sigma_max_BA / base_sigma
                    scaled_dominance = scaling * dominance
                else:
                    # Estimate: use σ_max(BA) as fraction of typical weight magnitude
                    # This is a fallback when we don't have base model
                    dominance = sigma_max_BA / 10.0  # Rough estimate
                    scaled_dominance = scaling * dominance
                
                per_layer_dominance[name] = dominance
                per_layer_scaled[name] = scaled_dominance
        
        if not per_layer_dominance:
            return LoRAStructuralMetrics(
                mean_dominance=0.0,
                max_dominance=0.0,
                scaled_mean_dominance=0.0,
                per_layer_dominance={},
                per_layer_scaled_dominance={},
                adapter_health=1.0,
                worst_kappa_A=1.0,
                worst_kappa_A_layer="",
                worst_kappa_B=1.0,
                worst_kappa_B_layer="",
                is_dominating=False,
                is_ill_conditioned=False,
                layers_dominating=[],
                layers_ill_conditioned=[],
            )
        
        # Aggregate
        dominances = list(per_layer_dominance.values())
        scaled_dominances = list(per_layer_scaled.values())
        
        mean_dominance = sum(dominances) / len(dominances)
        max_dominance = max(dominances)
        scaled_mean = sum(scaled_dominances) / len(scaled_dominances)
        
        # Find worst conditioning
        worst_kappa_A = max(kappa_As.values())
        worst_kappa_A_layer = max(kappa_As, key=kappa_As.get)
        worst_kappa_B = max(kappa_Bs.values())
        worst_kappa_B_layer = max(kappa_Bs, key=kappa_Bs.get)
        
        # Adapter health: inverse of worst conditioning
        worst_kappa = max(worst_kappa_A, worst_kappa_B)
        adapter_health = 1.0 / worst_kappa if worst_kappa > 0 else 1.0
        
        # Find problematic layers
        layers_dominating = [
            name for name, d in per_layer_scaled.items() 
            if d > self.DOMINANCE_WARNING
        ]
        
        layers_ill_conditioned = []
        for name in self.adapters:
            if kappa_As.get(name, 0) > self.KAPPA_WARNING:
                layers_ill_conditioned.append(f"{name}:A")
            if kappa_Bs.get(name, 0) > self.KAPPA_WARNING:
                layers_ill_conditioned.append(f"{name}:B")
        
        metrics = LoRAStructuralMetrics(
            mean_dominance=mean_dominance,
            max_dominance=max_dominance,
            scaled_mean_dominance=scaled_mean,
            per_layer_dominance=per_layer_dominance,
            per_layer_scaled_dominance=per_layer_scaled,
            adapter_health=adapter_health,
            worst_kappa_A=worst_kappa_A,
            worst_kappa_A_layer=worst_kappa_A_layer,
            worst_kappa_B=worst_kappa_B,
            worst_kappa_B_layer=worst_kappa_B_layer,
            is_dominating=len(layers_dominating) > 0,
            is_ill_conditioned=len(layers_ill_conditioned) > 0,
            layers_dominating=layers_dominating,
            layers_ill_conditioned=layers_ill_conditioned,
        )
        
        self.structural_history.append(metrics)
        return metrics
    
    def _find_adapters(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """Find all LoRA adapter pairs in the model."""
        adapters = {}
        
        # Find all lora_A and lora_B parameters
        lora_params = {}
        for name, param in self.model.named_parameters():
            for pattern in self.adapter_patterns:
                if pattern in name:
                    lora_params[name] = param
                    break
        
        # Group into A/B pairs
        # Typical naming: "layer.0.attention.q.lora_A" / "layer.0.attention.q.lora_B"
        for name in lora_params:
            if "lora_A" in name:
                base_name = name.replace(".lora_A", "").replace("lora_A", "")
                b_name = name.replace("lora_A", "lora_B")
                
                if b_name in lora_params:
                    adapters[base_name] = {
                        'A': lora_params[name],
                        'B': lora_params[b_name],
                        'A_name': name,
                        'B_name': b_name,
                    }
        
        return adapters
    
    def _infer_layer_type(self, name: str) -> str:
        """Infer the layer type from parameter name."""
        name_lower = name.lower()
        
        if 'q_proj' in name_lower or '.q.' in name_lower:
            return 'attention_q'
        elif 'k_proj' in name_lower or '.k.' in name_lower:
            return 'attention_k'
        elif 'v_proj' in name_lower or '.v.' in name_lower:
            return 'attention_v'
        elif 'o_proj' in name_lower or '.o.' in name_lower:
            return 'attention_o'
        elif 'up_proj' in name_lower or 'fc1' in name_lower or 'ffn.lin1' in name_lower:
            return 'ffn_up'
        elif 'down_proj' in name_lower or 'fc2' in name_lower or 'ffn.lin2' in name_lower:
            return 'ffn_down'
        elif 'gate' in name_lower:
            return 'ffn_gate'
        else:
            return 'unknown'
    
    def _analyze_adapter(
        self, 
        name: str, 
        A: torch.Tensor, 
        B: torch.Tensor,
        alpha: float = None,
    ) -> LoRALayerMetrics:
        """Analyze a single LoRA adapter pair."""
        
        with torch.no_grad():
            A = A.float()
            B = B.float()
            
            # Dimensions
            # A is typically (r, k) - projects from hidden to rank
            # B is typically (d, r) - projects from rank to output
            # But conventions vary, so we handle both
            
            r = min(A.shape[0], A.shape[1], B.shape[0], B.shape[1])
            nominal_rank = r
            
            # Default alpha = r if not specified
            if alpha is None:
                alpha = float(r)
            scaling = alpha / r
            
            # Analyze A
            S_A = torch.linalg.svdvals(A)
            a_kappa = _compute_kappa(S_A)
            a_effective_rank = _compute_effective_rank(S_A)
            a_sigma_max = S_A[0].item() if len(S_A) > 0 else 0.0
            
            # Analyze B
            S_B = torch.linalg.svdvals(B)
            b_kappa = _compute_kappa(S_B)
            b_effective_rank = _compute_effective_rank(S_B)
            b_sigma_max = S_B[0].item() if len(S_B) > 0 else 0.0
            
            # Analyze product BA
            BA = B @ A
            S_BA = torch.linalg.svdvals(BA)
            ba_effective_rank = _compute_effective_rank(S_BA)
            ba_sigma_max = S_BA[0].item() if len(S_BA) > 0 else 0.0
            ba_frobenius = torch.norm(BA, 'fro').item()
            
            # Rank utilization
            rank_utilization = ba_effective_rank / nominal_rank if nominal_rank > 0 else 0.0
            
            return LoRALayerMetrics(
                name=name,
                layer_type=self._infer_layer_type(name),
                nominal_rank=nominal_rank,
                alpha=alpha,
                scaling=scaling,
                a_shape=tuple(A.shape),
                a_kappa=a_kappa,
                a_effective_rank=a_effective_rank,
                a_sigma_max=a_sigma_max,
                b_shape=tuple(B.shape),
                b_kappa=b_kappa,
                b_effective_rank=b_effective_rank,
                b_sigma_max=b_sigma_max,
                ba_effective_rank=ba_effective_rank,
                ba_sigma_max=ba_sigma_max,
                ba_frobenius=ba_frobenius,
                rank_utilization=rank_utilization,
            )
    
    def analyze(self, alpha: float = None) -> LoRAModelMetrics:
        """
        Analyze all LoRA adapters in the model.
        
        Args:
            alpha: LoRA alpha value (if known). If None, assumes α=r.
            
        Returns:
            LoRAModelMetrics with per-layer and aggregate metrics
        """
        if not self.adapters:
            return LoRAModelMetrics(
                layers={},
                num_adapters=0,
                total_nominal_rank=0,
                total_effective_rank=0.0,
                overall_utilization=0.0,
                min_utilization=0.0,
                min_utilization_layer="",
                max_kappa=0.0,
                max_kappa_layer="",
            )
        
        layer_metrics = {}
        
        for name, adapter in self.adapters.items():
            metrics = self._analyze_adapter(
                name,
                adapter['A'],
                adapter['B'],
                alpha=alpha,
            )
            layer_metrics[name] = metrics
        
        # Aggregate
        total_nominal = sum(m.nominal_rank for m in layer_metrics.values())
        total_effective = sum(m.ba_effective_rank for m in layer_metrics.values())
        overall_utilization = total_effective / total_nominal if total_nominal > 0 else 0.0
        
        # Find worst cases
        min_util_layer = min(layer_metrics.values(), key=lambda m: m.rank_utilization)
        max_kappa_layer = max(layer_metrics.values(), key=lambda m: max(m.a_kappa, m.b_kappa))
        
        result = LoRAModelMetrics(
            layers=layer_metrics,
            num_adapters=len(layer_metrics),
            total_nominal_rank=total_nominal,
            total_effective_rank=total_effective,
            overall_utilization=overall_utilization,
            min_utilization=min_util_layer.rank_utilization,
            min_utilization_layer=min_util_layer.name,
            max_kappa=max(max_kappa_layer.a_kappa, max_kappa_layer.b_kappa),
            max_kappa_layer=max_kappa_layer.name,
        )
        
        self.history.append(result)
        return result
    
    def suggest_rank(self, safety_margin: float = 1.5) -> Dict[str, Any]:
        """
        Suggest optimal LoRA rank based on observed utilization.
        
        Args:
            safety_margin: Multiply effective rank by this for suggestion
            
        Returns:
            Dictionary with rank recommendation and reasoning
        """
        if not self.history:
            metrics = self.analyze()
        else:
            metrics = self.history[-1]
        
        if metrics.num_adapters == 0:
            return {
                'recommended_rank': 8,
                'reasoning': "No LoRA adapters found. Suggesting default rank-8.",
                'confidence': 'low',
            }
        
        # Base recommendation on max effective rank observed
        max_effective = max(m.ba_effective_rank for m in metrics.layers.values())
        recommended = int(max_effective * safety_margin)
        
        # Round up to common LoRA ranks
        common_ranks = [4, 8, 16, 32, 64, 128]
        for r in common_ranks:
            if r >= recommended:
                recommended = r
                break
        else:
            recommended = common_ranks[-1]
        
        # Current nominal rank
        current_rank = metrics.layers[list(metrics.layers.keys())[0]].nominal_rank
        
        # Build reasoning
        if recommended < current_rank:
            reasoning = (
                f"Your adapters are using ~{metrics.overall_utilization:.0%} of rank-{current_rank}. "
                f"Rank-{recommended} would capture the same adaptation with less memory."
            )
            potential_savings = (current_rank - recommended) / current_rank
        else:
            reasoning = (
                f"Your adapters are fully utilizing rank-{current_rank}. "
                f"Current configuration appears appropriate."
            )
            potential_savings = 0.0
        
        return {
            'current_rank': current_rank,
            'recommended_rank': recommended,
            'max_effective_rank': max_effective,
            'overall_utilization': metrics.overall_utilization,
            'reasoning': reasoning,
            'potential_memory_savings': potential_savings,
            'confidence': 'high' if len(self.history) > 5 else 'medium',
        }
    
    def get_layer_recommendations(self) -> Dict[str, int]:
        """
        Get per-layer rank recommendations.
        
        Returns:
            Dict mapping layer name to recommended rank
        """
        if not self.history:
            self.analyze()
        
        metrics = self.history[-1]
        recommendations = {}
        
        common_ranks = [4, 8, 16, 32, 64, 128]
        
        for name, layer in metrics.layers.items():
            effective = layer.ba_effective_rank
            recommended = int(effective * 1.5)
            
            for r in common_ranks:
                if r >= recommended:
                    recommended = r
                    break
            
            recommendations[name] = recommended
        
        return recommendations
    
    def get_utilization_by_type(self) -> Dict[str, float]:
        """
        Get average utilization grouped by layer type.
        
        Returns:
            Dict mapping layer type to average utilization
        """
        if not self.history:
            self.analyze()
        
        metrics = self.history[-1]
        by_type = {}
        
        for layer in metrics.layers.values():
            layer_type = layer.layer_type
            if layer_type not in by_type:
                by_type[layer_type] = []
            by_type[layer_type].append(layer.rank_utilization)
        
        return {t: sum(v)/len(v) for t, v in by_type.items()}
    
    def report(self, lora_alpha: float = None, lora_rank: int = None) -> str:
        """
        Generate human-readable report.
        
        Args:
            lora_alpha: LoRA alpha for structural analysis (optional)
            lora_rank: LoRA rank for structural analysis (optional)
        """
        if not self.history:
            self.analyze()
        
        metrics = self.history[-1]
        
        lines = []
        lines.append("=" * 60)
        lines.append("LoRA ADAPTER ANALYSIS")
        lines.append("=" * 60)
        lines.append("")
        
        # Overview
        lines.append(f"Adapters found: {metrics.num_adapters}")
        lines.append(f"Overall rank utilization: {metrics.overall_utilization:.0%}")
        lines.append("")
        
        # Recommendation
        suggestion = self.suggest_rank()
        lines.append(f"Current rank: {suggestion['current_rank']}")
        lines.append(f"Recommended rank: {suggestion['recommended_rank']}")
        lines.append(f"  {suggestion['reasoning']}")
        lines.append("")
        
        # By layer type
        by_type = self.get_utilization_by_type()
        if by_type:
            lines.append("Utilization by layer type:")
            for layer_type, util in sorted(by_type.items()):
                lines.append(f"  {layer_type:20s}: {util:.0%}")
            lines.append("")
        
        # Worst performers
        lines.append(f"Lowest utilization: {metrics.min_utilization_layer}")
        lines.append(f"  ({metrics.min_utilization:.0%} utilization)")
        lines.append("")
        
        # Conditioning concerns
        if metrics.max_kappa > 1000:
            lines.append(f"⚠️  Conditioning concern: {metrics.max_kappa_layer}")
            lines.append(f"   κ = {metrics.max_kappa:.0f} (high condition number)")
            lines.append("")
        
        # Structural analysis (if parameters provided)
        if lora_alpha is not None and lora_rank is not None:
            lines.append("-" * 60)
            lines.append("STRUCTURAL ANALYSIS (Adapter Dominance)")
            lines.append("-" * 60)
            lines.append("")
            
            structural = self.analyze_structural(lora_alpha, lora_rank)
            
            lines.append(f"Mean adapter dominance: {structural.mean_dominance:.1%}")
            lines.append(f"Max adapter dominance: {structural.max_dominance:.1%}")
            lines.append(f"Scaled dominance (with α/r): {structural.scaled_mean_dominance:.1%}")
            lines.append("")
            
            # Interpretation
            if structural.scaled_mean_dominance < 0.05:
                lines.append("Status: ✓ Minimal modification (very conservative)")
            elif structural.scaled_mean_dominance < 0.10:
                lines.append("Status: ✓ Healthy fine-tuning")
            elif structural.scaled_mean_dominance < 0.30:
                lines.append("Status: ⚠️ Significant modification")
            else:
                lines.append("Status: 🚨 Adapter dominating base model")
            lines.append("")
            
            # Warnings
            if structural.is_dominating:
                lines.append(f"⚠️  Dominating layers ({len(structural.layers_dominating)}):")
                for layer in structural.layers_dominating[:5]:
                    dom = structural.per_layer_scaled_dominance[layer]
                    lines.append(f"     {layer}: {dom:.1%}")
                if len(structural.layers_dominating) > 5:
                    lines.append(f"     ... and {len(structural.layers_dominating) - 5} more")
                lines.append("")
            
            if structural.is_ill_conditioned:
                lines.append(f"⚠️  Ill-conditioned adapters ({len(structural.layers_ill_conditioned)}):")
                for layer in structural.layers_ill_conditioned[:5]:
                    lines.append(f"     {layer}")
                lines.append("")
            
            lines.append(f"Adapter health score: {structural.adapter_health:.4f}")
            lines.append(f"  (1.0 = perfect, <0.001 = problematic)")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_dominance_trajectory(self) -> List[Tuple[int, float]]:
        """Get adapter dominance over time (from structural_history)."""
        return [(i, m.scaled_mean_dominance) for i, m in enumerate(self.structural_history)]
