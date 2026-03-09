"""
Stage state management for Gradience Bench resume functionality.

Tracks completion status of expensive operations to enable smart resume
that avoids re-running completed stages, especially valuable for 90-minute
probe training phases.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class StageStateManager:
    """
    Manages stage completion state for bench protocol resume functionality.

    Tracks which expensive stages have completed successfully to enable
    smart resume that skips completed work, saving significant time and cost.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize stage state manager.

        Args:
            output_dir: Bench output directory where stage_state.json will be stored
        """
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / "stage_state.json"
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load existing state or create empty state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    # Ensure required structure
                    if not isinstance(state, dict):
                        state = {}
                    if "stages" not in state:
                        state["stages"] = {}
                    if "variants" not in state:
                        state["variants"] = {}
                    return state
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load stage state from {self.state_file}: {e}")
                print("Starting with fresh state.")

        return {"created_at": time.time(), "stages": {}, "variants": {}, "metadata": {}}

    def _save_state(self):
        """Save current state to disk."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._state["last_updated"] = time.time()

            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=2)
        except (OSError, TypeError, ValueError) as e:
            print(f"Warning: Could not save stage state to {self.state_file}: {e}")

    def mark_stage_completed(self, stage_name: str, metadata: dict[str, Any] | None = None):
        """
        Mark a stage as completed.

        Args:
            stage_name: Name of the completed stage
            metadata: Optional metadata about the stage completion
        """
        self._state["stages"][stage_name] = {"completed": True, "completed_at": time.time(), "metadata": metadata or {}}
        self._save_state()
        print(f"✅ Stage completed: {stage_name}")

    def mark_variant_completed(self, variant_name: str, metadata: dict[str, Any] | None = None):
        """
        Mark a compression variant as completed.

        Args:
            variant_name: Name of the completed variant
            metadata: Optional metadata about the variant completion
        """
        self._state["variants"][variant_name] = {
            "completed": True,
            "completed_at": time.time(),
            "metadata": metadata or {},
        }
        self._save_state()
        print(f"✅ Variant completed: {variant_name}")

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage has been completed."""
        return bool(self._state["stages"].get(stage_name, {}).get("completed", False))

    def is_variant_completed(self, variant_name: str) -> bool:
        """Check if a variant has been completed."""
        return bool(self._state["variants"].get(variant_name, {}).get("completed", False))

    def get_completed_stages(self) -> set[str]:
        """Get set of all completed stage names."""
        return {name for name, info in self._state["stages"].items() if info.get("completed", False)}

    def get_completed_variants(self) -> set[str]:
        """Get set of all completed variant names."""
        return {name for name, info in self._state["variants"].items() if info.get("completed", False)}

    def should_skip_probe_training(self, probe_rank: int) -> bool:
        """
        Check if probe training should be skipped based on existing artifacts.

        Args:
            probe_rank: Probe rank to check

        Returns:
            True if probe training can be skipped
        """
        stage_name = f"probe_r{probe_rank}_trained"

        # Check stage completion state
        if self.is_stage_completed(stage_name):
            # Verify artifacts still exist
            probe_dir = self.output_dir / f"probe_r{probe_rank}"

            # Check for training telemetry
            run_jsonl = probe_dir / "run.jsonl"
            if not run_jsonl.exists():
                print(f"⚠️  Stage marked complete but missing telemetry: {run_jsonl}")
                return False

            # Check for adapter config
            adapter_config = probe_dir / "adapter_config.json"
            if not adapter_config.exists():
                print(f"⚠️  Stage marked complete but missing adapter config: {adapter_config}")
                return False

            print(f"🔄 Skipping probe training (r={probe_rank}) - already completed")
            return True

        return False

    def should_skip_probe_evaluation(self, probe_rank: int) -> bool:
        """Check if probe evaluation should be skipped."""
        stage_name = f"probe_r{probe_rank}_evaluated"

        if self.is_stage_completed(stage_name):
            probe_dir = self.output_dir / f"probe_r{probe_rank}"
            eval_json = probe_dir / "eval.json"

            if not eval_json.exists():
                print(f"⚠️  Stage marked complete but missing eval.json: {eval_json}")
                return False

            print(f"🔄 Skipping probe evaluation (r={probe_rank}) - already completed")
            return True

        return False

    def should_skip_probe_audit(self, probe_rank: int) -> bool:
        """Check if probe audit should be skipped."""
        stage_name = f"probe_r{probe_rank}_audited"

        if self.is_stage_completed(stage_name):
            probe_dir = self.output_dir / f"probe_r{probe_rank}"
            audit_json = probe_dir / "audit.json"

            if not audit_json.exists():
                print(f"⚠️  Stage marked complete but missing audit.json: {audit_json}")
                return False

            print(f"🔄 Skipping probe audit (r={probe_rank}) - already completed")
            return True

        return False

    def should_skip_config_generation(self) -> bool:
        """Check if compression config generation should be skipped."""
        if self.is_stage_completed("compression_configs_generated"):
            config_file = self.output_dir / "compression_configs.json"

            if not config_file.exists():
                print(f"⚠️  Stage marked complete but missing compression_configs.json: {config_file}")
                return False

            print("🔄 Skipping compression config generation - already completed")
            return True

        return False

    def should_skip_variant_training(self, variant_name: str) -> bool:
        """
        Check if variant training should be skipped.

        Args:
            variant_name: Name of the variant to check

        Returns:
            True if variant training can be skipped
        """
        if self.is_variant_completed(variant_name):
            variant_dir = self.output_dir / variant_name

            # Check for evaluation results
            eval_json = variant_dir / "eval.json"
            if not eval_json.exists():
                print(f"⚠️  Variant marked complete but missing eval.json: {eval_json}")
                return False

            # Check for adapter config (training artifact)
            adapter_config = variant_dir / "adapter_config.json"
            if not adapter_config.exists():
                print(f"⚠️  Variant marked complete but missing adapter config: {adapter_config}")
                return False

            print(f"🔄 Skipping variant training ({variant_name}) - already completed")
            return True

        return False

    def get_resume_summary(self) -> dict[str, Any]:
        """
        Get a summary of what can be resumed.

        Returns:
            Summary of completed stages and variants
        """
        completed_stages = self.get_completed_stages()
        completed_variants = self.get_completed_variants()

        return {
            "total_completed_stages": len(completed_stages),
            "total_completed_variants": len(completed_variants),
            "completed_stages": sorted(completed_stages),
            "completed_variants": sorted(completed_variants),
            "state_file": str(self.state_file),
            "last_updated": self._state.get("last_updated"),
            "created_at": self._state.get("created_at"),
        }

    def clean_invalid_state(self):
        """
        Clean up state entries that reference non-existent artifacts.

        This helps recover from situations where files were manually deleted
        but state still shows them as completed.
        """
        cleaned_stages = []
        cleaned_variants = []

        # Clean stage state
        for stage_name in list(self._state["stages"].keys()):
            if "probe_r" in stage_name:
                # Extract probe rank and check artifacts
                try:
                    if "_trained" in stage_name:
                        rank = stage_name.split("_r")[1].split("_")[0]
                        probe_dir = self.output_dir / f"probe_r{rank}"
                        if not (probe_dir / "run.jsonl").exists():
                            del self._state["stages"][stage_name]
                            cleaned_stages.append(stage_name)
                    elif "_evaluated" in stage_name:
                        rank = stage_name.split("_r")[1].split("_")[0]
                        probe_dir = self.output_dir / f"probe_r{rank}"
                        if not (probe_dir / "eval.json").exists():
                            del self._state["stages"][stage_name]
                            cleaned_stages.append(stage_name)
                    elif "_audited" in stage_name:
                        rank = stage_name.split("_r")[1].split("_")[0]
                        probe_dir = self.output_dir / f"probe_r{rank}"
                        if not (probe_dir / "audit.json").exists():
                            del self._state["stages"][stage_name]
                            cleaned_stages.append(stage_name)
                except (ValueError, IndexError):
                    # If parsing fails, remove the entry
                    del self._state["stages"][stage_name]
                    cleaned_stages.append(stage_name)
            elif stage_name == "compression_configs_generated":
                if not (self.output_dir / "compression_configs.json").exists():
                    del self._state["stages"][stage_name]
                    cleaned_stages.append(stage_name)

        # Clean variant state
        for variant_name in list(self._state["variants"].keys()):
            variant_dir = self.output_dir / variant_name
            if not (variant_dir / "eval.json").exists():
                del self._state["variants"][variant_name]
                cleaned_variants.append(variant_name)

        if cleaned_stages or cleaned_variants:
            self._save_state()
            print("🧹 Cleaned invalid state entries:")
            for stage in cleaned_stages:
                print(f"   - Stage: {stage}")
            for variant in cleaned_variants:
                print(f"   - Variant: {variant}")

    def reset_state(self):
        """Reset all state (for testing or fresh start)."""
        self._state = {"created_at": time.time(), "stages": {}, "variants": {}, "metadata": {}}
        self._save_state()
        print("🔄 Stage state reset")


def create_stage_manager(output_dir: Path) -> StageStateManager:
    """
    Factory function to create a stage state manager.

    Args:
        output_dir: Bench output directory

    Returns:
        Initialized StageStateManager instance
    """
    return StageStateManager(output_dir)
