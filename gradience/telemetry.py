"""
Telemetry Logging

STABILITY: Stable

Provides structured JSONL logging for training metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


class TelemetryWriter:
    """
    Write training telemetry to JSONL file.
    
    Usage:
        writer = TelemetryWriter("training.jsonl")
        
        writer.log_event("start", {"config": {...}})
        writer.log_spectral(step, metrics)
        writer.log_eval(step, {"accuracy": 0.95})
        
        writer.close()
    """
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w")
        self._closed = False
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a generic event."""
        if self._closed:
            return
            
        record = {
            "timestamp": time.time(),
            "event": event_type,
            **data,
        }
        self._write(record)
    
    def log_spectral(self, step: int, metrics: Dict[str, Any]):
        """Log spectral metrics (κ, rank, σ_max)."""
        self.log_event("spectral", {"step": step, **metrics})
    
    def log_structural(self, step: int, metrics: Dict[str, Any]):
        """
        Log structural metrics (muon ratio ρ, expansion pressure).
        
        Args:
            step: Training step
            metrics: From StructuralAnalyzer.analyze().to_dict() or raw dict
        """
        self.log_event("structural", {"step": step, **metrics})
    
    def log_muon_ratio(self, step: int, muon_ratio: float, weight_decay: float, sigma_max: float):
        """
        Convenience method to log muon ratio components.
        
        Args:
            step: Training step
            muon_ratio: ρ = λ × σ_max
            weight_decay: λ from optimizer
            sigma_max: Mean spectral norm across layers
        """
        self.log_event("structural", {
            "step": step,
            "muon_ratio": muon_ratio,
            "weight_decay": weight_decay,
            "sigma_max": sigma_max,
            "is_stable": muon_ratio <= 1.0,
        })
    
    def log_eval(self, step: int, metrics: Dict[str, Any]):
        """Log evaluation metrics."""
        self.log_event("eval", {"step": step, **metrics})
    
    def log_loss(self, step: int, loss: float):
        """Log training loss."""
        self.log_event("loss", {"step": step, "loss": loss})
    
    def _write(self, record: Dict[str, Any]):
        """Write a record to the file."""
        if self._closed:
            return
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
    
    def close(self):
        """Close the telemetry file."""
        if not self._closed:
            self._closed = True
            self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


class TelemetryReader:
    """
    Read and analyze telemetry from JSONL file.
    
    Usage:
        reader = TelemetryReader("training.jsonl")
        
        # Get all spectral events
        spectral = reader.get_events("spectral")
        
        # Get muon ratio trajectory
        muon = reader.get_trajectory("structural", "muon_ratio")
        
        # Get summary
        print(reader.summary())
    """
    
    def __init__(self, path: str):
        self.path = Path(path)
        self._events = None
    
    def _load(self):
        """Load all events from file."""
        if self._events is not None:
            return
        
        self._events = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    def get_events(self, event_type: Optional[str] = None) -> list:
        """
        Get events, optionally filtered by type.
        
        Args:
            event_type: Filter to specific event type (e.g., "spectral", "structural")
            
        Returns:
            List of event dictionaries
        """
        self._load()
        
        if event_type is None:
            return self._events
        
        return [e for e in self._events if e.get("event") == event_type]
    
    def get_trajectory(self, event_type: str, field: str) -> list:
        """
        Get trajectory of a specific field over time.
        
        Args:
            event_type: Event type to filter
            field: Field name to extract
            
        Returns:
            List of (step, value) tuples
        """
        events = self.get_events(event_type)
        result = []
        
        for e in events:
            step = e.get("step")
            value = e.get(field)
            if step is not None and value is not None:
                result.append((step, value))
        
        return result
    
    def get_muon_trajectory(self) -> list:
        """Get muon ratio trajectory. Convenience method."""
        return self.get_trajectory("structural", "muon_ratio")
    
    def get_kappa_trajectory(self) -> list:
        """Get kappa trajectory. Convenience method."""
        return self.get_trajectory("spectral", "kappa_mean")
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the training run."""
        self._load()
        
        spectral = self.get_events("spectral")
        structural = self.get_events("structural")
        loss_events = self.get_events("loss")
        
        result = {
            "total_events": len(self._events),
            "spectral_events": len(spectral),
            "structural_events": len(structural),
            "loss_events": len(loss_events),
        }
        
        # Spectral summary
        if spectral:
            kappas = [e.get("kappa_mean", 0) for e in spectral if "kappa_mean" in e]
            if kappas:
                result["kappa_mean_final"] = kappas[-1]
                result["kappa_mean_max"] = max(kappas)
        
        # Structural summary
        if structural:
            muons = [e.get("muon_ratio", 0) for e in structural if "muon_ratio" in e]
            if muons:
                result["muon_ratio_final"] = muons[-1]
                result["muon_ratio_max"] = max(muons)
                result["muon_above_1_count"] = sum(1 for m in muons if m > 1.0)
        
        # Loss summary
        if loss_events:
            losses = [e.get("loss", 0) for e in loss_events if "loss" in e]
            if losses:
                result["loss_final"] = losses[-1]
                result["loss_min"] = min(losses)
        
        return result
# ---------------------------------------------------------------------
# === LEGACY API COMPAT (tests/test_core.py expects these names) ===
# Backwards-compatible export (tests + older code expect SummaryGenerator)
# ---------------------------------------------------------------------
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Summary:
    train_ppl: Optional[float] = None
    test_ppl: Optional[float] = None
    train_acc: Optional[float] = None
    test_acc: Optional[float] = None
    gap: Optional[float] = None
    n_events: int = 0
    n_eval: int = 0
    extras: Dict[str, Any] = None  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "train_ppl": self.train_ppl,
            "test_ppl": self.test_ppl,
            "train_acc": self.train_acc,
            "test_acc": self.test_acc,
            "gap": self.gap,
            "n_events": self.n_events,
            "n_eval": self.n_eval,
        }
        if self.extras:
            d["extras"] = self.extras
        return d


class SummaryGenerator:
    """
    Back-compat summary generator.

    - If the telemetry file is vNext schema, delegates to gradience.vnext.TelemetryReader.summarize().
    - Otherwise parses JSONL best-effort and extracts last train/test eval metrics.
    """

    def __init__(self, path: str, strict: bool = False) -> None:
        self.path = path
        self.strict = strict
        self.issues: List[str] = []

    def validate(self) -> List[str]:
        # Very light validation; primarily checks that JSON lines parse.
        self.issues = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except Exception as e:
                        self.issues.append(f"line {i}: invalid JSON ({e})")
                        if self.strict:
                            break
        except FileNotFoundError:
            self.issues.append(f"file not found: {self.path}")
        return self.issues

    def summarize(self) -> Summary:
        # Try vNext first
        try:
            from gradience.vnext.telemetry_reader import TelemetryReader  # type: ignore
            r = TelemetryReader(self.path, strict_schema=False)
            issues = r.validate()
            sig = r.summarize()
            s = Summary(
                train_ppl=getattr(sig, "train_ppl", None),
                test_ppl=getattr(sig, "test_ppl", None),
                train_acc=getattr(sig, "train_accuracy", None) if hasattr(sig, "train_accuracy") else getattr(sig, "train_acc", None),
                test_acc=getattr(sig, "test_accuracy", None) if hasattr(sig, "test_accuracy") else getattr(sig, "test_acc", None),
                gap=getattr(sig, "gap", None),
                n_events=0,
                n_eval=0,
                extras={"telemetry_issues": issues, "source": "vnext"} if issues else {"source": "vnext"},
            )
            return s
        except Exception:
            # Fall back to legacy parsing
            pass

        train_ppl = test_ppl = None
        train_acc = test_acc = None
        n_events = 0
        n_eval = 0

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                n_events += 1

                # Heuristic: vNext eval events
                ev = e.get("event") or e.get("type")
                if ev == "eval":
                    n_eval += 1
                    split = e.get("split")
                    metrics = e.get("metrics") or {}
                    ppl = metrics.get("ppl")
                    acc = metrics.get("accuracy")
                    if split == "train":
                        train_ppl = ppl if ppl is not None else train_ppl
                        train_acc = acc if acc is not None else train_acc
                    elif split == "test":
                        test_ppl = ppl if ppl is not None else test_ppl
                        test_acc = acc if acc is not None else test_acc

                # Older patterns (best-effort)
                if isinstance(e.get("train_ppl"), (int, float)):
                    train_ppl = float(e["train_ppl"])
                if isinstance(e.get("test_ppl"), (int, float)):
                    test_ppl = float(e["test_ppl"])
                if isinstance(e.get("train_acc"), (int, float)):
                    train_acc = float(e["train_acc"])
                if isinstance(e.get("test_acc"), (int, float)):
                    test_acc = float(e["test_acc"])

        gap = None
        if train_ppl and test_ppl and train_ppl > 0:
            gap = test_ppl / train_ppl

        return Summary(
            train_ppl=train_ppl,
            test_ppl=test_ppl,
            train_acc=train_acc,
            test_acc=test_acc,
            gap=gap,
            n_events=n_events,
            n_eval=n_eval,
            extras={"source": "legacy"},
        )

    # Common alias in older code
    def generate(self) -> Dict[str, Any]:
        return self.summarize().to_dict()


# Export for star-import patterns
try:
    __all__  # type: ignore
except NameError:
    __all__ = []  # type: ignore
if isinstance(__all__, list):
    for _name in ("SummaryGenerator", "Summary"):
        if _name not in __all__:
            __all__.append(_name)  # type: ignore
