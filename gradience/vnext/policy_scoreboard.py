"""
Policy Scoreboard - Self-Improving Policy Framework

Tracks policy performance across benchmark runs to provide evidence-based 
policy selection and continuous improvement insights.

Key metrics:
- Pass rate: How often policy passes Bench validation
- Optimality: How close to best-performing rank  
- Conservatism bias: Tendency to be too aggressive/conservative
- Reliability: Consistency across different models/tasks
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import os


@dataclass
class PolicyBenchmarkResult:
    """Single benchmark result for a policy."""
    config_name: str
    date: str
    model_name: str
    task_name: str
    policy_name: str
    suggested_rank: int
    actual_rank: int
    optimal_rank: int  # Best performing rank among all candidates
    passed: bool  # Whether it passed Bench validation
    performance_delta: float  # Performance vs baseline
    best_candidate_delta: float  # Best performance among all candidates
    seed: int = 42


@dataclass 
class PolicyMetrics:
    """Aggregated metrics for a single policy."""
    policy_name: str
    total_attempts: int
    passes: int
    pass_rate: float
    optimality_scores: List[float]
    avg_optimality: float
    conservatism_bias: float
    reliability_score: float
    trend: str  # "improving", "stable", "declining"
    last_5_performance: List[float]
    note: Optional[str] = None


class PolicyScoreboard:
    """Manages policy performance tracking and scoreboard generation."""
    
    def __init__(self, scoreboard_path: Optional[Path] = None):
        """Initialize scoreboard with optional custom storage path."""
        if scoreboard_path is None:
            # Default to user's .gradience directory
            home = Path.home()
            gradience_dir = home / ".gradience"
            gradience_dir.mkdir(exist_ok=True)
            scoreboard_path = gradience_dir / "policy_scoreboard.json"
        
        self.scoreboard_path = scoreboard_path
        self.data = self._load_scoreboard()
    
    def _load_scoreboard(self) -> Dict[str, Any]:
        """Load existing scoreboard or create new one."""
        if self.scoreboard_path.exists():
            with open(self.scoreboard_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "policy_scoreboard_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_benchmarks": 0,
                "policies": {},
                "benchmark_history": []
            }
    
    def add_benchmark_results(self, 
                            config_name: str,
                            model_name: str, 
                            task_name: str,
                            policy_results: List[PolicyBenchmarkResult]) -> None:
        """Add results from a benchmark run."""
        
        # Update total benchmark count
        self.data["total_benchmarks"] += 1
        self.data["last_updated"] = datetime.now().isoformat()
        
        # Add to benchmark history
        benchmark_entry = {
            "config": config_name,
            "model": model_name,
            "task": task_name,
            "date": datetime.now().isoformat(),
            "policies_tested": len(policy_results)
        }
        self.data["benchmark_history"].append(benchmark_entry)
        
        # Process each policy result
        for result in policy_results:
            policy_name = result.policy_name
            
            # Initialize policy if new
            if policy_name not in self.data["policies"]:
                self.data["policies"][policy_name] = {
                    "total_attempts": 0,
                    "passes": 0,
                    "benchmarks": [],
                    "optimality_scores": [],
                    "performance_history": []
                }
            
            policy_data = self.data["policies"][policy_name]
            
            # Add this benchmark result
            benchmark_result = {
                "config": config_name,
                "model": model_name,
                "task": task_name,
                "date": result.date,
                "suggested_rank": result.suggested_rank,
                "actual_rank": result.actual_rank,
                "optimal_rank": result.optimal_rank,
                "passed": result.passed,
                "performance_delta": result.performance_delta,
                "optimality": self._calculate_optimality(result),
                "seed": result.seed
            }
            
            policy_data["benchmarks"].append(benchmark_result)
            policy_data["total_attempts"] += 1
            if result.passed:
                policy_data["passes"] += 1
            
            # Track optimality and performance over time
            optimality = self._calculate_optimality(result)
            policy_data["optimality_scores"].append(optimality)
            policy_data["performance_history"].append(result.performance_delta)
        
        # Recalculate all metrics
        self._recalculate_metrics()
        
        # Save updated scoreboard
        self.save()
    
    def _calculate_optimality(self, result: PolicyBenchmarkResult) -> float:
        """Calculate how close policy was to optimal performance."""
        if result.best_candidate_delta == 0:
            return 1.0  # If best was 0, any non-negative is optimal
        
        if result.best_candidate_delta < 0:
            # Best candidate had negative performance, policy optimality depends on relative performance
            if result.performance_delta >= result.best_candidate_delta:
                # Policy performed at least as well as best
                return 1.0
            else:
                # Policy performed worse than best
                return max(0.0, 1.0 - abs(result.performance_delta - result.best_candidate_delta) / abs(result.best_candidate_delta))
        else:
            # Best candidate had positive performance
            return max(0.0, min(1.0, result.performance_delta / result.best_candidate_delta))
    
    def _recalculate_metrics(self) -> None:
        """Recalculate derived metrics for all policies."""
        for policy_name, policy_data in self.data["policies"].items():
            if policy_data["total_attempts"] == 0:
                continue
                
            # Calculate basic metrics
            policy_data["pass_rate"] = policy_data["passes"] / policy_data["total_attempts"]
            policy_data["avg_optimality"] = statistics.mean(policy_data["optimality_scores"]) if policy_data["optimality_scores"] else 0.0
            
            # Calculate conservatism bias
            rank_biases = []
            for benchmark in policy_data["benchmarks"]:
                if benchmark["optimal_rank"] > 0:
                    bias = (benchmark["actual_rank"] - benchmark["optimal_rank"]) / benchmark["optimal_rank"]
                    rank_biases.append(bias)
            
            policy_data["conservatism_bias"] = statistics.mean(rank_biases) if rank_biases else 0.0
            
            # Calculate reliability (consistency)
            if len(policy_data["optimality_scores"]) > 1:
                std_dev = statistics.stdev(policy_data["optimality_scores"])
                policy_data["reliability_score"] = max(0.0, 1.0 - std_dev)
            else:
                policy_data["reliability_score"] = 1.0 if policy_data["optimality_scores"] else 0.0
            
            # Determine trend (last 5 vs previous 5)
            policy_data["trend"] = self._calculate_trend(policy_data["performance_history"])
            
            # Last 5 performance scores  
            policy_data["last_5_performance"] = policy_data["performance_history"][-5:]
            
            # Add interpretive note
            policy_data["note"] = self._generate_policy_note(policy_data)
    
    def _calculate_trend(self, performance_history: List[float]) -> str:
        """Calculate performance trend."""
        if len(performance_history) < 4:
            return "insufficient_data"
        
        # Compare last 5 to previous 5 (or available data)
        recent = performance_history[-5:]
        if len(performance_history) >= 10:
            previous = performance_history[-10:-5]
        else:
            previous = performance_history[:-5] if len(performance_history) > 5 else performance_history[:len(performance_history)//2]
        
        if not previous:
            return "stable"
            
        recent_avg = statistics.mean(recent)
        previous_avg = statistics.mean(previous)
        
        change = (recent_avg - previous_avg) / abs(previous_avg) if previous_avg != 0 else 0
        
        if change > 0.05:  # 5% improvement
            return "improving"
        elif change < -0.05:  # 5% decline
            return "declining"
        else:
            return "stable"
    
    def _generate_policy_note(self, policy_data: Dict[str, Any]) -> str:
        """Generate interpretive note for policy."""
        bias = policy_data.get("conservatism_bias", 0)
        reliability = policy_data.get("reliability_score", 0)
        pass_rate = policy_data.get("pass_rate", 0)
        
        notes = []
        
        # Bias interpretation
        if bias > 0.15:
            notes.append("Very conservative - suggests high ranks")
        elif bias > 0.05:
            notes.append("Slightly conservative")
        elif bias < -0.15:
            notes.append("Very aggressive - suggests low ranks")  
        elif bias < -0.05:
            notes.append("Slightly aggressive")
        else:
            notes.append("Well-calibrated")
        
        # Reliability
        if reliability > 0.9:
            notes.append("highly reliable")
        elif reliability > 0.75:
            notes.append("reliable")
        elif reliability < 0.6:
            notes.append("inconsistent")
        
        # Pass rate
        if pass_rate > 0.85:
            notes.append("high success rate")
        elif pass_rate < 0.6:
            notes.append("low success rate")
        
        return ", ".join(notes)
    
    def get_policy_metrics(self, policy_name: str) -> Optional[PolicyMetrics]:
        """Get metrics for a specific policy."""
        if policy_name not in self.data["policies"]:
            return None
        
        policy_data = self.data["policies"][policy_name]
        
        return PolicyMetrics(
            policy_name=policy_name,
            total_attempts=policy_data.get("total_attempts", 0),
            passes=policy_data.get("passes", 0),
            pass_rate=policy_data.get("pass_rate", 0.0),
            optimality_scores=policy_data.get("optimality_scores", []),
            avg_optimality=policy_data.get("avg_optimality", 0.0),
            conservatism_bias=policy_data.get("conservatism_bias", 0.0),
            reliability_score=policy_data.get("reliability_score", 0.0),
            trend=policy_data.get("trend", "unknown"),
            last_5_performance=policy_data.get("last_5_performance", []),
            note=policy_data.get("note")
        )
    
    def get_best_policies(self, top_n: int = 5) -> List[Tuple[str, PolicyMetrics]]:
        """Get top N policies by overall score."""
        policy_scores = []
        
        for policy_name in self.data["policies"]:
            metrics = self.get_policy_metrics(policy_name)
            if metrics and metrics.total_attempts >= 3:  # Minimum attempts for ranking
                # Weighted score: optimality (50%) + pass_rate (30%) + reliability (20%)
                score = (metrics.avg_optimality * 0.5 + 
                        metrics.pass_rate * 0.3 + 
                        metrics.reliability_score * 0.2)
                policy_scores.append((score, policy_name, metrics))
        
        # Sort by score descending
        policy_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [(name, metrics) for score, name, metrics in policy_scores[:top_n]]
    
    def generate_summary_insights(self) -> Dict[str, Any]:
        """Generate summary insights for the scoreboard."""
        if not self.data["policies"]:
            return {"note": "No policy data available yet"}
        
        best_policies = self.get_best_policies(5)
        
        if not best_policies:
            return {"note": "Insufficient data for insights (need 3+ attempts per policy)"}
        
        # Find extremes
        most_conservative = None
        most_aggressive = None
        most_reliable = None
        best_overall = best_policies[0] if best_policies else None
        
        max_bias = -999
        min_bias = 999
        max_reliability = 0
        
        for policy_name in self.data["policies"]:
            metrics = self.get_policy_metrics(policy_name)
            if metrics and metrics.total_attempts >= 3:
                if metrics.conservatism_bias > max_bias:
                    max_bias = metrics.conservatism_bias
                    most_conservative = policy_name
                if metrics.conservatism_bias < min_bias:
                    min_bias = metrics.conservatism_bias
                    most_aggressive = policy_name
                if metrics.reliability_score > max_reliability:
                    max_reliability = metrics.reliability_score
                    most_reliable = policy_name
        
        # Generate recommendations
        recommendations = []
        if best_overall:
            name, metrics = best_overall
            recommendations.append(
                f"{name} shows best overall performance ({metrics.avg_optimality:.0%} optimality, {metrics.pass_rate:.0%} pass rate)"
            )
        
        if most_conservative and most_conservative != (best_overall[0] if best_overall else None):
            conservative_metrics = self.get_policy_metrics(most_conservative)
            recommendations.append(
                f"{most_conservative} is most conservative (+{conservative_metrics.conservatism_bias:.0%} bias) - good for safety-critical applications"
            )
        
        if most_aggressive and most_aggressive != (best_overall[0] if best_overall else None):
            aggressive_metrics = self.get_policy_metrics(most_aggressive)
            recommendations.append(
                f"{most_aggressive} is most aggressive ({aggressive_metrics.conservatism_bias:.0%} bias) - good for maximum compression"
            )
        
        return {
            "best_overall_policy": best_overall[0] if best_overall else None,
            "most_reliable": most_reliable,
            "most_aggressive": most_aggressive,
            "most_conservative": most_conservative,
            "total_benchmarks_analyzed": self.data["total_benchmarks"],
            "policies_tracked": len(self.data["policies"]),
            "recommendations": recommendations
        }
    
    def save(self) -> None:
        """Save scoreboard to disk."""
        self.scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scoreboard_path, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def export_snapshot(self, output_path: Path) -> None:
        """Export current scoreboard snapshot to specific location."""
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def generate_markdown_table(self) -> str:
        """Generate markdown scoreboard table for reports."""
        best_policies = self.get_best_policies(10)  # Show up to 10
        
        if not best_policies:
            return "## 📊 Policy Scoreboard\n\n*No sufficient policy data available yet (need 3+ attempts per policy)*\n"
        
        lines = [
            "## 📊 Policy Scoreboard",
            "",
            "| Policy | Pass Rate | Optimality | Bias | Reliability | Trend | Notes |",
            "|--------|-----------|------------|------|-------------|--------|-------|"
        ]
        
        for i, (policy_name, metrics) in enumerate(best_policies):
            # Add star for best overall
            name_display = f"{policy_name} ⭐" if i == 0 else policy_name
            
            # Format bias with emoji
            bias_str = f"{metrics.conservatism_bias:+.0%}"
            if metrics.conservatism_bias > 0.1:
                bias_emoji = "🔒"  # Conservative
            elif metrics.conservatism_bias < -0.1:
                bias_emoji = "⚡"  # Aggressive
            else:
                bias_emoji = "⚖️"  # Balanced
            
            # Trend emoji
            trend_emoji = {"improving": "↗️", "declining": "↘️", "stable": "➡️"}.get(metrics.trend, "❓")
            
            # Format numbers
            pass_rate = f"{metrics.pass_rate:.0%} ({metrics.passes}/{metrics.total_attempts})"
            optimality = f"{metrics.avg_optimality:.0%}"
            reliability = f"{metrics.reliability_score:.0%}"
            
            line = f"| {name_display} | {pass_rate} | {optimality} | {bias_str} {bias_emoji} | {reliability} | {trend_emoji} | {metrics.note or ''} |"
            lines.append(line)
        
        # Add insights
        insights = self.generate_summary_insights()
        if insights.get("recommendations"):
            lines.extend([
                "",
                "### Key Insights:"
            ])
            for rec in insights["recommendations"]:
                lines.append(f"- {rec}")
        
        # Add legend
        lines.extend([
            "",
            "**Legend**: ⭐ Best Overall • 🔒 Conservative • ⚡ Aggressive • ⚖️ Balanced • ↗️ Improving • ↘️ Declining • ➡️ Stable"
        ])
        
        return "\n".join(lines)


def create_policy_result_from_bench_data(config_name: str,
                                       model_name: str,
                                       task_name: str, 
                                       policy_name: str,
                                       suggested_rank: int,
                                       actual_rank: int,
                                       performance_delta: float,
                                       passed: bool,
                                       all_results: Dict[str, float],
                                       seed: int = 42) -> PolicyBenchmarkResult:
    """Helper to create PolicyBenchmarkResult from Bench output data."""
    
    # Find optimal rank (best performing among all candidates)
    if all_results:
        best_performance = max(all_results.values())
        best_rank = None
        for variant_name, perf in all_results.items():
            if perf == best_performance:
                # Extract rank from variant name if possible
                try:
                    if 'r' in variant_name:
                        rank_part = variant_name.split('r')[-1].split('_')[0]
                        best_rank = int(rank_part)
                        break
                except:
                    pass
        
        optimal_rank = best_rank if best_rank else actual_rank
        best_candidate_delta = best_performance
    else:
        optimal_rank = actual_rank
        best_candidate_delta = performance_delta
    
    return PolicyBenchmarkResult(
        config_name=config_name,
        date=datetime.now().isoformat(),
        model_name=model_name,
        task_name=task_name,
        policy_name=policy_name,
        suggested_rank=suggested_rank,
        actual_rank=actual_rank,
        optimal_rank=optimal_rank,
        passed=passed,
        performance_delta=performance_delta,
        best_candidate_delta=best_candidate_delta,
        seed=seed
    )