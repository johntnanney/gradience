"""
Heartbeat logger for keeping SSH sessions alive during long-running bench phases.

Prevents apparent "stalls" during silent phases like evaluation, auditing, and saving
by emitting periodic keepalive messages to both console and log files.
"""

import threading
import time
from pathlib import Path
from typing import Optional, Union, List
import sys
import datetime


class HeartbeatLogger:
    """
    Background heartbeat logger that emits periodic keepalive messages.
    
    Designed to prevent SSH timeouts and provide visibility into long-running
    bench phases that might otherwise appear stuck.
    
    Enhanced with explicit stage boundary logging for debugging.
    """
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._stage_name: str = ""
        self._start_time: float = 0.0
        self._output_dir: Optional[Path] = None
        self._log_file: Optional[Path] = None
        self._interval: int = 30
        self._seed: Optional[int] = None
        self._output_artifacts: List[str] = []
        
    def start_heartbeat(self, 
                       stage_name: str, 
                       output_dir: Optional[Union[str, Path]] = None,
                       interval: int = 30,
                       log_to_file: bool = True,
                       seed: Optional[int] = None) -> None:
        """
        Start the heartbeat logger for a given stage.
        
        Args:
            stage_name: Human-readable name of the current stage
            output_dir: Directory for log file output (optional)
            interval: Seconds between heartbeat messages
            log_to_file: Whether to write heartbeats to a log file
            seed: Random seed for the run (for debugging context)
        """
        # Stop any existing heartbeat
        self.stop_heartbeat()
        
        self._stage_name = stage_name
        self._start_time = time.time()
        self._output_dir = Path(output_dir) if output_dir else None
        self._interval = interval
        self._seed = seed
        self._output_artifacts = []
        
        # Set up log file if requested
        if log_to_file and self._output_dir:
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                self._log_file = self._output_dir / "heartbeat.log"
                
                # Write initial stage start message
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                start_msg = f"[{timestamp}] Starting stage: {stage_name}\n"
                with open(self._log_file, "a") as f:
                    f.write(start_msg)
            except (OSError, PermissionError) as e:
                # If log file setup fails, continue without logging to file
                print(f"[heartbeat] Warning: Could not set up log file at {self._output_dir}: {e}")
                self._log_file = None
        
        # Start background thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        
        # Print explicit stage start message with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seed_info = f" (seed={seed})" if seed is not None else ""
        print(f">>> STAGE START: {stage_name}{seed_info} [{timestamp}]")
        sys.stdout.flush()
        
        if self._output_dir:
            print(f"    Output dir: {self._output_dir}")
        if self._log_file:
            print(f"    Log file: {self._log_file}")
    
    def stop_heartbeat(self, output_artifacts: Optional[List[Union[str, Path]]] = None) -> None:
        """
        Stop the heartbeat logger.
        
        Args:
            output_artifacts: List of output files/directories created during this stage
        """
        if self._stop_event:
            self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if self._stage_name:  # Only print if we had an active heartbeat
            elapsed_seconds = int(time.time() - self._start_time)
            
            # Format duration as HH:MM:SS
            hours = elapsed_seconds // 3600
            minutes = (elapsed_seconds % 3600) // 60
            seconds = elapsed_seconds % 60
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Build artifacts info
            if output_artifacts:
                self._output_artifacts.extend([str(Path(a).name) for a in output_artifacts])
            
            artifacts_info = ""
            if self._output_artifacts:
                if len(self._output_artifacts) == 1:
                    artifacts_info = f" wrote={self._output_artifacts[0]}"
                else:
                    artifacts_info = f" wrote={len(self._output_artifacts)} files"
            
            # Print explicit stage end message with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f">>> STAGE END: {self._stage_name} (took={duration_str}){artifacts_info} [{timestamp}]")
            sys.stdout.flush()
            
            # Write completion message to log file
            if self._log_file:
                end_msg = f"[{timestamp}] STAGE END: {self._stage_name} (took={duration_str}){artifacts_info}\n"
                try:
                    with open(self._log_file, "a") as f:
                        f.write(end_msg)
                        f.flush()
                except (OSError, PermissionError):
                    pass  # Don't crash if log write fails

        # Reset state
        self._thread = None
        self._stop_event = None
        self._stage_name = ""
        self._start_time = 0.0
        self._output_dir = None
        self._log_file = None
        self._seed = None
        self._output_artifacts = []
    
    def _heartbeat_loop(self) -> None:
        """Main heartbeat loop running in background thread."""
        try:
            while not self._stop_event.wait(timeout=self._interval):
                elapsed = int(time.time() - self._start_time)
                
                # Format heartbeat message
                heartbeat_msg = f"[heartbeat] stage={self._stage_name} elapsed={elapsed}s"
                
                if self._output_dir:
                    heartbeat_msg += f" output={self._output_dir}"
                
                # Print to console (with flush for SSH)
                print(heartbeat_msg)
                sys.stdout.flush()
                
                # Write to log file
                if self._log_file:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_msg = f"[{timestamp}] {heartbeat_msg}\n"
                    try:
                        with open(self._log_file, "a") as f:
                            f.write(log_msg)
                            f.flush()
                    except (OSError, PermissionError):
                        pass  # Don't crash heartbeat on log write failures

        except Exception as e:  # Intentionally broad: background thread must not crash
            # Log any unexpected errors but don't crash
            try:
                error_msg = f"[heartbeat] ERROR: {e}\n"
                print(error_msg.strip())
                if self._log_file:
                    with open(self._log_file, "a") as f:
                        f.write(error_msg)
            except Exception:  # Intentionally broad: last-resort error handler in thread
                pass
    
    def update_stage(self, new_stage_name: str) -> None:
        """
        Update the current stage name without stopping/restarting heartbeat.
        
        Useful for sub-stages within a longer operation.
        """
        if self._stop_event and not self._stop_event.is_set():
            old_stage = self._stage_name
            self._stage_name = new_stage_name
            elapsed = int(time.time() - self._start_time)
            
            transition_msg = f"[heartbeat] stage transition: {old_stage} -> {new_stage_name} (elapsed={elapsed}s)"
            print(transition_msg)
            sys.stdout.flush()
            
            if self._log_file:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_msg = f"[{timestamp}] {transition_msg}\n"
                try:
                    with open(self._log_file, "a") as f:
                        f.write(log_msg)
                        f.flush()
                except (OSError, PermissionError):
                    pass


# Global heartbeat instance for easy use
_global_heartbeat = HeartbeatLogger()


def start_heartbeat(stage_name: str, 
                   output_dir: Optional[Union[str, Path]] = None,
                   interval: int = 30,
                   log_to_file: bool = True,
                   seed: Optional[int] = None) -> None:
    """
    Start a heartbeat logger for the given stage.
    
    This is the main API function that should be used in bench protocols.
    
    Args:
        stage_name: Human-readable name of the current stage
        output_dir: Directory for log file output (optional)
        interval: Seconds between heartbeat messages (default: 30)
        log_to_file: Whether to write heartbeats to a log file (default: True)
        seed: Random seed for the run (for debugging context)
    
    Example:
        start_heartbeat("eval_probe", output_dir="./results/experiment_01", interval=30, seed=42)
    """
    _global_heartbeat.start_heartbeat(stage_name, output_dir, interval, log_to_file, seed)


def stop_heartbeat(output_artifacts: Optional[List[Union[str, Path]]] = None) -> None:
    """
    Stop the active heartbeat logger.
    
    Should be called at the end of each stage to provide completion timing.
    
    Args:
        output_artifacts: List of output files/directories created during this stage
    """
    _global_heartbeat.stop_heartbeat(output_artifacts)


def update_stage(new_stage_name: str) -> None:
    """
    Update the current stage name without stopping the heartbeat.
    
    Useful for sub-stages within a longer operation.
    
    Args:
        new_stage_name: New stage name to display in heartbeat messages
    """
    _global_heartbeat.update_stage(new_stage_name)


# Context manager for automatic heartbeat management
class heartbeat_stage:
    """
    Context manager for automatic heartbeat management with stage boundary logging.
    
    Usage:
        with heartbeat_stage("eval_probe", output_dir="/path/to/output", seed=42) as stage:
            # Long-running operation
            trainer.evaluate()
            # Optionally track outputs
            stage.add_artifact("eval.json")
    """
    
    def __init__(self, 
                 stage_name: str, 
                 output_dir: Optional[Union[str, Path]] = None,
                 interval: int = 30,
                 log_to_file: bool = True,
                 seed: Optional[int] = None):
        self.stage_name = stage_name
        self.output_dir = output_dir
        self.interval = interval
        self.log_to_file = log_to_file
        self.seed = seed
        self.artifacts: List[Union[str, Path]] = []
    
    def add_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Add an artifact that will be reported at stage end."""
        self.artifacts.append(artifact_path)
    
    def __enter__(self):
        start_heartbeat(self.stage_name, self.output_dir, self.interval, self.log_to_file, self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        stop_heartbeat(self.artifacts if self.artifacts else None)
        
        # If an exception occurred, log it
        if exc_type is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg = f">>> STAGE FAILED: {self.stage_name} - {exc_type.__name__}: {exc_val} [{timestamp}]"
            print(error_msg)
            sys.stdout.flush()