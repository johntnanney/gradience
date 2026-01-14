"""
Protocol invariant checker for Gradience Bench.

Validates critical assumptions and fails loudly when reality breaks.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter


class InvariantChecker:
    """Check protocol invariants and generate diagnostic report."""
    
    def __init__(self, min_probe_accuracy: float = 0.75):
        self.min_probe_accuracy = min_probe_accuracy
        self.invariants = {}
        self.failures = []
        
    def check_probe_quality(self, probe_accuracy: float) -> Dict[str, Any]:
        """Check if probe passes quality gate."""
        passed = probe_accuracy >= self.min_probe_accuracy
        
        result = {
            "name": "probe_quality",
            "status": "PASSED" if passed else "FAILED",
            "probe_accuracy": probe_accuracy,
            "threshold": self.min_probe_accuracy,
            "message": f"Probe accuracy {probe_accuracy:.3f} {'meets' if passed else 'below'} threshold {self.min_probe_accuracy}"
        }
        
        if not passed:
            self.failures.append(f"Probe quality check failed: {probe_accuracy:.3f} < {self.min_probe_accuracy}")
            
        self.invariants["probe_quality"] = result
        return result
    
    def check_rank_heterogeneity(self, rank_pattern: Dict[str, int], 
                                 variant_name: str = "per_layer") -> Dict[str, Any]:
        """Check if per_layer has sufficient rank heterogeneity."""
        if not rank_pattern:
            result = {
                "name": "rank_heterogeneity",
                "status": "SKIPPED",
                "message": "No rank pattern provided",
                "variant": variant_name
            }
            self.invariants["rank_heterogeneity"] = result
            return result
        
        # Count unique ranks
        unique_ranks = set(rank_pattern.values())
        rank_histogram = Counter(rank_pattern.values())
        
        # For per_layer, we expect at least 2 distinct ranks
        min_distinct = 2 if variant_name == "per_layer" else 1
        passed = len(unique_ranks) >= min_distinct
        
        result = {
            "name": "rank_heterogeneity",
            "status": "PASSED" if passed else "FAILED",
            "variant": variant_name,
            "distinct_ranks": len(unique_ranks),
            "min_required": min_distinct,
            "rank_histogram": dict(rank_histogram),
            "unique_ranks": sorted(list(unique_ranks)),
            "message": f"{variant_name} has {len(unique_ranks)} distinct ranks (requires >= {min_distinct})"
        }
        
        if not passed and variant_name == "per_layer":
            self.failures.append(f"Rank heterogeneity failed for {variant_name}: only {len(unique_ranks)} distinct ranks")
            
        self.invariants["rank_heterogeneity"] = result
        return result
    
    def check_layer_consistency(self, rank_pattern: Dict[str, int], 
                                alpha_pattern: Dict[str, int]) -> Dict[str, Any]:
        """Check if rank and alpha patterns have matching layer keys."""
        if not rank_pattern or not alpha_pattern:
            result = {
                "name": "layer_consistency",
                "status": "SKIPPED",
                "message": "Missing rank or alpha pattern"
            }
            self.invariants["layer_consistency"] = result
            return result
        
        rank_layers = set(rank_pattern.keys())
        alpha_layers = set(alpha_pattern.keys())
        
        passed = rank_layers == alpha_layers
        
        missing_in_alpha = rank_layers - alpha_layers
        missing_in_rank = alpha_layers - rank_layers
        
        result = {
            "name": "layer_consistency",
            "status": "PASSED" if passed else "FAILED",
            "rank_layers_count": len(rank_layers),
            "alpha_layers_count": len(alpha_layers),
            "layers_match": passed,
            "message": "Layer keys match" if passed else "Layer key mismatch detected"
        }
        
        if not passed:
            result["missing_in_alpha"] = list(missing_in_alpha)
            result["missing_in_rank"] = list(missing_in_rank)
            self.failures.append(f"Layer consistency failed: rank/alpha pattern keys don't match")
            
        self.invariants["layer_consistency"] = result
        return result
    
    def check_param_counting(self, compression_configs: Dict[str, Any],
                           probe_audit: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate parameter counting methodology."""
        result = {
            "name": "param_counting",
            "status": "UNKNOWN",
            "param_count_source": "unknown",
            "message": "Parameter counting validation"
        }
        
        # Check if we have audit-based counting
        if probe_audit and "total_lora_params" in probe_audit:
            result["param_count_source"] = "audit_total_lora_params"
            result["audit_params"] = probe_audit["total_lora_params"]
            result["status"] = "PASSED"
            result["message"] = "Using audit-based total_lora_params for consistency"
        else:
            # Fall back to config-based counting
            result["param_count_source"] = "config_calculation"
            result["status"] = "WARNING"
            result["message"] = "No audit data available, using config-based calculation"
            
        self.invariants["param_counting"] = result
        return result
    
    def check_compression_validity(self, verdicts: Dict[str, Any]) -> Dict[str, Any]:
        """Check if compression results are internally consistent."""
        result = {
            "name": "compression_validity",
            "status": "PASSED",
            "issues": [],
            "message": "Compression results validation"
        }
        
        for variant, verdict in verdicts.items():
            if verdict.get("status") == "evaluated":
                # Check delta calculation
                if verdict.get("compressed_accuracy") is not None and verdict.get("probe_accuracy") is not None:
                    expected_delta = verdict["compressed_accuracy"] - verdict["probe_accuracy"]
                    reported_delta = verdict.get("delta_vs_probe", 0)
                    
                    if abs(expected_delta - reported_delta) > 0.001:  # Tolerance for float precision
                        issue = f"{variant}: Delta mismatch (expected={expected_delta:.4f}, reported={reported_delta:.4f})"
                        result["issues"].append(issue)
                        result["status"] = "FAILED"
                
                # Check param reduction calculation
                if verdict.get("compressed_params") and verdict.get("probe_params"):
                    expected_reduction = 1 - (verdict["compressed_params"] / verdict["probe_params"])
                    reported_reduction = verdict.get("param_reduction", 0)
                    
                    if abs(expected_reduction - reported_reduction) > 0.001:
                        issue = f"{variant}: Param reduction mismatch"
                        result["issues"].append(issue)
                        result["status"] = "FAILED"
        
        if result["issues"]:
            result["message"] = f"Found {len(result['issues'])} consistency issues"
            self.failures.extend(result["issues"])
        else:
            result["message"] = "All compression results internally consistent"
            
        self.invariants["compression_validity"] = result
        return result
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive invariant report."""
        total_checks = len(self.invariants)
        passed = sum(1 for inv in self.invariants.values() if inv.get("status") == "PASSED")
        failed = sum(1 for inv in self.invariants.values() if inv.get("status") == "FAILED")
        skipped = sum(1 for inv in self.invariants.values() if inv.get("status") == "SKIPPED")
        warnings = sum(1 for inv in self.invariants.values() if inv.get("status") == "WARNING")
        
        return {
            "protocol_invariants": self.invariants,
            "summary": {
                "total_checks": total_checks,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "warnings": warnings,
                "overall_status": "FAILED" if failed > 0 else "PASSED" if passed > 0 else "UNKNOWN"
            },
            "failures": self.failures
        }
    
    def validate_bench_output(self, bench_json_path: Path) -> Dict[str, Any]:
        """Validate a complete bench output file."""
        with open(bench_json_path) as f:
            data = json.load(f)
        
        # Check probe quality
        if "probe" in data or "probe_baseline" in data:
            probe_acc = data.get("probe", {}).get("accuracy") or data.get("probe_baseline", {}).get("accuracy")
            if probe_acc:
                self.check_probe_quality(probe_acc)
        
        # Check compression validity
        if "verdicts" in data:
            self.check_compression_validity(data["verdicts"])
        
        # Check rank heterogeneity for per_layer
        compression_configs = None
        if Path(bench_json_path.parent / "compression_configs.json").exists():
            with open(bench_json_path.parent / "compression_configs.json") as f:
                compression_configs = json.load(f)
            
            if "per_layer" in compression_configs:
                per_layer_config = compression_configs["per_layer"]
                rank_pattern = per_layer_config.get("rank_pattern", {})
                alpha_pattern = per_layer_config.get("alpha_pattern", {})
                
                if rank_pattern:
                    self.check_rank_heterogeneity(rank_pattern, "per_layer")
                    
                if rank_pattern and alpha_pattern:
                    self.check_layer_consistency(rank_pattern, alpha_pattern)
        
        # Check param counting
        probe_audit = None
        probe_audit_path = bench_json_path.parent / "probe_r32" / "audit.json"
        if probe_audit_path.exists():
            with open(probe_audit_path) as f:
                probe_audit = json.load(f)
        
        if compression_configs:
            self.check_param_counting(compression_configs, probe_audit)
        
        return self.generate_report()


def add_invariants_to_bench_output(bench_dir: Path) -> None:
    """Add invariant checks to existing bench output."""
    checker = InvariantChecker()
    
    bench_json = bench_dir / "bench.json"
    if not bench_json.exists():
        print(f"Warning: {bench_json} not found")
        return
    
    # Run validation
    report = checker.validate_bench_output(bench_json)
    
    # Load existing bench.json
    with open(bench_json) as f:
        bench_data = json.load(f)
    
    # Add invariants section
    bench_data["protocol_invariants"] = report["protocol_invariants"]
    bench_data["invariant_summary"] = report["summary"]
    
    if report["failures"]:
        bench_data["invariant_failures"] = report["failures"]
    
    # Write updated bench.json
    with open(bench_json, "w") as f:
        json.dump(bench_data, f, indent=2)
    
    # Also save standalone invariants report
    invariants_json = bench_dir / "invariants.json"
    with open(invariants_json, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Invariant checks added to {bench_json}")
    if report["failures"]:
        print(f"⚠️  Found {len(report['failures'])} invariant failures:")
        for failure in report["failures"]:
            print(f"   - {failure}")
    else:
        print(f"✅ All invariants passed!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        bench_dir = Path(sys.argv[1])
        add_invariants_to_bench_output(bench_dir)
    else:
        print("Usage: python invariants.py <bench_output_dir>")