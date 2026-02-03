"""
Watchdog system for detecting and diagnosing hangs in Gradience Bench.

Monitors critical ML operations that are prone to hanging:
- Model saving/loading (filesystem I/O issues)
- Evaluation/generation (GPU/compute hangs)
- Network-mounted storage operations

Provides comprehensive diagnostics when hangs are detected.
"""

import os
import signal
import threading
import time
import traceback
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import datetime
import psutil
import faulthandler


class HangDiagnostics:
    """Collects and formats diagnostic information when hangs are detected."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Collect system resource information."""
        try:
            # Memory info
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_percent": memory.percent,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 1),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except Exception as e:
            return {"error": f"Failed to collect system info: {e}"}
    
    @staticmethod
    def get_process_info(process_name: str = "python") -> List[Dict[str, Any]]:
        """Get information about running processes matching the name."""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent', 'cpu_percent']):
                try:
                    if process_name.lower() in proc.info['name'].lower():
                        # Get command line for python processes
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if 'run_bench' in cmdline or 'gradience' in cmdline:
                            processes.append({
                                "pid": proc.info['pid'],
                                "name": proc.info['name'],
                                "cmdline": cmdline[:200] + "..." if len(cmdline) > 200 else cmdline,
                                "memory_percent": round(proc.info['memory_percent'], 2),
                                "cpu_percent": round(proc.info['cpu_percent'], 2)
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return processes
        except Exception as e:
            return [{"error": f"Failed to collect process info: {e}"}]
    
    @staticmethod
    def get_gpu_info() -> Optional[str]:
        """Get GPU information if available."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"nvidia-smi failed: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "nvidia-smi timeout"
        except FileNotFoundError:
            return "nvidia-smi not found (no NVIDIA GPU or drivers)"
        except Exception as e:
            return f"GPU info error: {e}"
    
    @staticmethod
    def get_disk_io_info() -> Dict[str, Any]:
        """Get disk I/O statistics."""
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters:
                return {
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes,
                    "read_count": io_counters.read_count,
                    "write_count": io_counters.write_count,
                    "read_time": io_counters.read_time,
                    "write_time": io_counters.write_time
                }
            return {"error": "No disk I/O counters available"}
        except Exception as e:
            return {"error": f"Failed to get disk I/O info: {e}"}
    
    @staticmethod
    def get_log_tail(log_file: Optional[Path], lines: int = 20) -> List[str]:
        """Get the last N lines from a log file."""
        if not log_file or not log_file.exists():
            return ["Log file not found or not specified"]
        
        try:
            with open(log_file, 'r') as f:
                return f.readlines()[-lines:]
        except Exception as e:
            return [f"Failed to read log file: {e}"]
    
    @classmethod
    def dump_all_diagnostics(cls, stage_name: str, timeout_minutes: int, 
                           log_file: Optional[Path] = None, 
                           output_dir: Optional[Path] = None) -> str:
        """Collect and format all diagnostic information."""
        
        divider = "=" * 60
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
{divider}
HANG DETECTED - DIAGNOSTIC DUMP
{divider}
Stage: {stage_name}
Timeout: {timeout_minutes} minutes
Timestamp: {timestamp}
PID: {os.getpid()}

{divider}
SYSTEM RESOURCES
{divider}
"""
        
        # System info
        system_info = cls.get_system_info()
        for key, value in system_info.items():
            report += f"{key}: {value}\n"
        
        # Process info
        report += f"\n{divider}\nRUNNING PROCESSES (gradience/run_bench)\n{divider}\n"
        processes = cls.get_process_info()
        for proc in processes:
            for key, value in proc.items():
                report += f"{key}: {value}\n"
            report += "---\n"
        
        # GPU info
        report += f"\n{divider}\nGPU STATUS\n{divider}\n"
        gpu_info = cls.get_gpu_info()
        report += f"{gpu_info}\n"
        
        # Disk I/O
        report += f"\n{divider}\nDISK I/O\n{divider}\n"
        io_info = cls.get_disk_io_info()
        for key, value in io_info.items():
            report += f"{key}: {value}\n"
        
        # Recent log entries
        if log_file:
            report += f"\n{divider}\nRECENT LOG ENTRIES ({log_file})\n{divider}\n"
            log_lines = cls.get_log_tail(log_file)
            for line in log_lines:
                report += line
        
        # Python stack traces (if available)
        report += f"\n{divider}\nPYTHON STACK TRACES\n{divider}\n"
        try:
            # Get stack traces of all threads
            for thread_id, frame in sys._current_frames().items():
                report += f"\nThread {thread_id}:\n"
                report += ''.join(traceback.format_stack(frame))
        except Exception as e:
            report += f"Failed to get stack traces: {e}\n"
        
        report += f"\n{divider}\nEND DIAGNOSTIC DUMP\n{divider}\n"
        
        # Save diagnostic dump to file if output_dir provided
        if output_dir:
            try:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                dump_file = output_dir / f"hang_diagnostic_{timestamp.replace(' ', '_').replace(':', '-')}.txt"
                with open(dump_file, 'w') as f:
                    f.write(report)
                report += f"\nDiagnostic dump saved to: {dump_file}\n"
            except Exception as e:
                report += f"\nFailed to save diagnostic dump: {e}\n"
        
        return report


class WatchdogTimer:
    """
    Watchdog timer that monitors a stage and triggers diagnostics on timeout.
    
    Uses a separate thread to monitor progress and can interrupt hanging operations.
    """
    
    def __init__(self, stage_name: str, timeout_minutes: int, 
                 log_file: Optional[Path] = None, 
                 output_dir: Optional[Path] = None):
        self.stage_name = stage_name
        self.timeout_seconds = timeout_minutes * 60
        self.log_file = log_file
        self.output_dir = output_dir
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.is_running = False
        self.timer_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start the watchdog timer."""
        with self.lock:
            if self.is_running:
                return
            
            self.is_running = True
            self.start_time = time.time()
            self.last_progress_time = time.time()
            
            self.timer_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self.timer_thread.start()
            
            print(f"🐕 Watchdog started for stage '{self.stage_name}' (timeout: {self.timeout_seconds/60:.1f} min)")
    
    def stop(self):
        """Stop the watchdog timer."""
        with self.lock:
            if not self.is_running:
                return
                
            self.is_running = False
            elapsed = time.time() - self.start_time
            print(f"🐕 Watchdog stopped for stage '{self.stage_name}' (ran for {elapsed/60:.1f} min)")
    
    def reset_progress(self):
        """Reset the progress timer (call when stage makes progress)."""
        with self.lock:
            self.last_progress_time = time.time()
    
    def _watchdog_loop(self):
        """Main watchdog monitoring loop."""
        # Use shorter check interval for short timeouts
        check_interval = min(30, max(5, self.timeout_seconds / 4))
        
        while self.is_running:
            time.sleep(check_interval)
            
            with self.lock:
                if not self.is_running:
                    break
                    
                elapsed = time.time() - self.last_progress_time
                
                if elapsed >= self.timeout_seconds:
                    print(f"\n🚨 HANG DETECTED in stage '{self.stage_name}'!")
                    print(f"⏰ No progress for {elapsed/60:.1f} minutes (timeout: {self.timeout_seconds/60:.1f} min)")
                    
                    # Generate diagnostic dump
                    diagnostic_report = HangDiagnostics.dump_all_diagnostics(
                        self.stage_name, int(self.timeout_seconds/60), 
                        self.log_file, self.output_dir
                    )
                    
                    print(diagnostic_report)
                    
                    # Optional: Send signal to trigger stack trace dump
                    try:
                        if hasattr(signal, 'SIGUSR1'):
                            os.kill(os.getpid(), signal.SIGUSR1)
                    except Exception:
                        pass
                    
                    # Mark as no longer running to avoid repeated dumps
                    self.is_running = False
                    break


class WatchdogContext:
    """Context manager for easy watchdog usage around critical operations."""
    
    def __init__(self, stage_name: str, timeout_minutes: int = 30,
                 log_file: Optional[Path] = None,
                 output_dir: Optional[Path] = None,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize watchdog context.
        
        Args:
            stage_name: Name of the stage being monitored
            timeout_minutes: Minutes without progress before triggering diagnostics
            log_file: Path to log file to include in diagnostics
            output_dir: Directory to save diagnostic dumps
            progress_callback: Optional callback to check for progress
        """
        self.watchdog = WatchdogTimer(stage_name, timeout_minutes, log_file, output_dir)
        self.progress_callback = progress_callback
    
    def __enter__(self):
        self.watchdog.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.watchdog.stop()
    
    def reset_progress(self):
        """Signal that progress has been made."""
        self.watchdog.reset_progress()


def setup_stack_trace_handler():
    """Set up signal handler for on-demand stack traces (Linux/Unix only)."""
    
    def dump_stack_traces(signum, frame):
        """Signal handler to dump stack traces of all threads."""
        print("\n" + "="*60)
        print("STACK TRACE DUMP (triggered by signal)")
        print("="*60)
        
        try:
            faulthandler.dump_traceback()
        except Exception as e:
            print(f"Faulthandler failed: {e}")
            
            # Fallback manual stack trace
            try:
                for thread_id, frame in sys._current_frames().items():
                    print(f"\nThread {thread_id}:")
                    traceback.print_stack(frame)
            except Exception as e2:
                print(f"Manual stack trace failed: {e2}")
        
        print("="*60)
        print("END STACK TRACE DUMP")
        print("="*60)
    
    # Set up signal handler for SIGUSR1 (Unix/Linux only)
    if hasattr(signal, 'SIGUSR1'):
        try:
            signal.signal(signal.SIGUSR1, dump_stack_traces)
            print("📡 Stack trace signal handler installed (kill -USR1 <pid> to trigger)")
        except Exception as e:
            print(f"Warning: Could not install signal handler: {e}")
    
    # Enable faulthandler for better crash diagnostics
    try:
        faulthandler.enable()
        print("🔧 Faulthandler enabled for crash diagnostics")
    except Exception as e:
        print(f"Warning: Could not enable faulthandler: {e}")


# Convenience functions for common use cases
def monitor_evaluation(stage_name: str, timeout_minutes: int = 45, **kwargs):
    """Create watchdog for evaluation stages (typically longer)."""
    return WatchdogContext(stage_name, timeout_minutes, **kwargs)

def monitor_file_operations(stage_name: str, timeout_minutes: int = 15, **kwargs):
    """Create watchdog for file I/O operations (typically shorter)."""
    return WatchdogContext(stage_name, timeout_minutes, **kwargs)

def monitor_training(stage_name: str, timeout_minutes: int = 120, **kwargs):
    """Create watchdog for training operations (very long)."""
    return WatchdogContext(stage_name, timeout_minutes, **kwargs)