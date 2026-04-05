"""
Unit tests for the Gradience Bench watchdog system.

Tests hang detection, diagnostic collection, and monitoring integration.
"""

import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from gradience.bench.monitored_stage import (
    monitor_evaluation,
    monitor_file_operations,
    monitor_training,
    monitored_stage,
    setup_global_monitoring,
)
from gradience.bench.watchdog import HangDiagnostics, WatchdogContext, WatchdogTimer, setup_stack_trace_handler


class TestHangDiagnostics(unittest.TestCase):
    """Test diagnostic information collection."""

    def test_system_info_collection(self):
        """Test system information gathering."""
        info = HangDiagnostics.get_system_info()

        # Should have key system metrics
        self.assertIn("timestamp", info)
        self.assertIn("memory_total_gb", info)
        self.assertIn("memory_available_gb", info)
        self.assertIn("disk_total_gb", info)
        self.assertIn("cpu_percent", info)

        # Values should be reasonable
        if "error" not in info:
            self.assertGreater(info["memory_total_gb"], 0)
            self.assertGreater(info["disk_total_gb"], 0)

    def test_process_info_collection(self):
        """Test process information gathering."""
        processes = HangDiagnostics.get_process_info("python")

        # Should return a list
        self.assertIsInstance(processes, list)

        # If we found processes, they should have expected keys
        for proc in processes:
            if "error" not in proc:
                self.assertIn("pid", proc)
                self.assertIn("name", proc)
                self.assertIn("cmdline", proc)

    def test_gpu_info_collection(self):
        """Test GPU information gathering."""
        gpu_info = HangDiagnostics.get_gpu_info()

        # Should return string (either info or error message)
        self.assertIsInstance(gpu_info, str)

        # Should handle case where nvidia-smi is not available gracefully
        if "not found" not in gpu_info.lower():
            # If nvidia-smi exists, response should be meaningful
            self.assertGreater(len(gpu_info), 0)

    def test_log_tail_reading(self):
        """Test log file tail reading."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            log_path = Path(f.name)
            # Write test log entries
            for i in range(10):
                f.write(f"Log entry {i}\n")

        try:
            # Read last 5 lines
            tail = HangDiagnostics.get_log_tail(log_path, 5)
            self.assertEqual(len(tail), 5)
            self.assertIn("Log entry 9", tail[-1])
            self.assertIn("Log entry 5", tail[0])
        finally:
            log_path.unlink()

    def test_comprehensive_diagnostic_dump(self):
        """Test complete diagnostic dump generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test log file
            log_file = temp_path / "test.log"
            with open(log_file, "w") as f:
                f.write("Test log entry 1\n")
                f.write("Test log entry 2\n")

            # Generate diagnostic dump
            report = HangDiagnostics.dump_all_diagnostics("test_stage", 30, log_file, temp_path)

            # Should be comprehensive
            self.assertIn("HANG DETECTED", report)
            self.assertIn("test_stage", report)
            self.assertIn("SYSTEM RESOURCES", report)
            self.assertIn("RUNNING PROCESSES", report)
            self.assertIn("GPU STATUS", report)
            self.assertIn("PYTHON STACK TRACES", report)
            self.assertIn("Test log entry", report)

            # Should save diagnostic file
            diagnostic_files = list(temp_path.glob("hang_diagnostic_*.txt"))
            self.assertGreater(len(diagnostic_files), 0)


class TestWatchdogTimer(unittest.TestCase):
    """Test watchdog timer functionality."""

    def test_basic_watchdog_lifecycle(self):
        """Test basic watchdog start and stop."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            watchdog = WatchdogTimer(
                "test_stage",
                timeout_minutes=0.05,  # 3 second timeout
                output_dir=temp_path,
            )

            # Start watchdog
            watchdog.start()
            self.assertTrue(watchdog.is_running)

            # Let it run briefly (shouldn't timeout)
            time.sleep(1)

            # Stop watchdog
            watchdog.stop()
            self.assertFalse(watchdog.is_running)

    def test_progress_reset(self):
        """Test that progress resets prevent timeout."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            watchdog = WatchdogTimer(
                "test_stage",
                timeout_minutes=0.05,  # 3 second timeout
                output_dir=temp_path,
            )

            watchdog.start()

            # Sleep almost to timeout, then reset progress
            time.sleep(2)
            watchdog.reset_progress()

            # Should still be running
            time.sleep(1)
            self.assertTrue(watchdog.is_running)

            watchdog.stop()

    @patch("builtins.print")  # Capture print output to avoid spam
    def test_timeout_detection(self, mock_print):
        """Test that timeout is detected and diagnostics are triggered.

        Uses a 3-second timeout (0.05 min).  With check_interval =
        max(1, 3/4) = 1 s the watchdog should detect the hang within
        ~4 seconds.  We wait 8 seconds to give ample room for the
        diagnostic collection (psutil, nvidia-smi) to finish.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            watchdog = WatchdogTimer(
                "test_stage",
                timeout_minutes=0.05,  # 3 second timeout
                output_dir=temp_path,
            )

            watchdog.start()

            # Wait for timeout + diagnostics to complete.  The check
            # interval is 1 s and timeout is 3 s, so detection happens
            # around t ≈ 4 s.  is_running is set False immediately on
            # detection, before the (potentially slow) diagnostic dump.
            time.sleep(8)

            # Should have triggered hang detection
            self.assertFalse(watchdog.is_running)

            # Should have printed diagnostic info
            printed_output = "".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
            self.assertIn("HANG DETECTED", printed_output)
            self.assertIn("test_stage", printed_output)


class TestWatchdogContext(unittest.TestCase):
    """Test watchdog context manager."""

    def test_context_manager_lifecycle(self):
        """Test watchdog context manager basic usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with WatchdogContext("test_context", timeout_minutes=1, output_dir=temp_path) as watchdog:
                self.assertIsNotNone(watchdog)
                time.sleep(1)
                watchdog.reset_progress()

    def test_context_manager_exception_handling(self):
        """Test that context manager handles exceptions properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                with WatchdogContext("test_context", timeout_minutes=1, output_dir=temp_path):
                    time.sleep(0.5)
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected

        # Watchdog should have stopped cleanly despite exception


class TestMonitoredStage(unittest.TestCase):
    """Test monitored stage integration."""

    def test_monitored_stage_basic_usage(self):
        """Test basic monitored stage functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with monitored_stage("test_stage", output_dir=temp_path, timeout_minutes=1, enable_watchdog=False) as stage:
                stage.progress("Making progress")
                stage.add_artifact("test.txt")

                # Create the artifact
                artifact_path = temp_path / "test.txt"
                artifact_path.write_text("test content")

    def test_stage_type_timeouts(self):
        """Test that different stage types have appropriate timeouts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test evaluation stage (should have 45 min timeout)
            eval_stage = monitor_evaluation("test_eval", output_dir=temp_path, enable_watchdog=False)
            self.assertEqual(eval_stage.timeout_minutes, 45)

            # Test training stage (should have 120 min timeout)
            train_stage = monitor_training("test_train", output_dir=temp_path, enable_watchdog=False)
            self.assertEqual(train_stage.timeout_minutes, 120)

            # Test file operations (should have 15 min timeout)
            file_stage = monitor_file_operations("test_file", output_dir=temp_path, enable_watchdog=False)
            self.assertEqual(file_stage.timeout_minutes, 15)

    def test_custom_timeout_override(self):
        """Test that custom timeouts override defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Override evaluation timeout
            stage = monitor_evaluation("test", output_dir=temp_path, timeout_minutes=60, enable_watchdog=False)
            self.assertEqual(stage.timeout_minutes, 60)

    @patch("builtins.print")  # Capture output
    def test_progress_reporting(self, mock_print):
        """Test progress reporting functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with monitored_stage("test_stage", output_dir=temp_path, enable_watchdog=False) as stage:
                stage.progress("Test progress message")

            # Should have printed progress message
            printed_output = "".join(str(call.args[0]) for call in mock_print.call_args_list)
            self.assertIn("Test progress message", printed_output)


class TestSignalHandling(unittest.TestCase):
    """Test signal handling for stack traces."""

    @unittest.skipIf(not hasattr(signal, "SIGUSR1"), "SIGUSR1 not available on this platform")
    def test_signal_handler_installation(self):
        """Test that signal handlers can be installed."""
        # Save original handler
        original_handler = signal.signal(signal.SIGUSR1, signal.SIG_DFL)

        try:
            setup_stack_trace_handler()

            # Handler should have been installed
            current_handler = signal.signal(signal.SIGUSR1, signal.SIG_DFL)
            self.assertNotEqual(current_handler, signal.SIG_DFL)

        finally:
            # Restore original handler
            signal.signal(signal.SIGUSR1, original_handler)

    def test_global_monitoring_setup(self):
        """Test global monitoring setup."""
        # Should not raise exceptions
        setup_global_monitoring()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete monitoring system."""

    @patch("builtins.print")  # Reduce test output
    def test_end_to_end_monitoring(self, mock_print):
        """Test complete monitoring workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Setup global monitoring
            setup_global_monitoring()

            # Run monitored operation
            with monitor_evaluation(
                "integration_test", output_dir=temp_path, timeout_minutes=1, enable_watchdog=False
            ) as stage:
                # Simulate work with progress updates
                stage.progress("Starting work")
                time.sleep(0.5)

                stage.progress("Halfway done")
                time.sleep(0.5)

                # Create some artifacts
                artifact1 = temp_path / "result1.json"
                artifact1.write_text('{"status": "completed"}')
                stage.add_artifact(artifact1)

                artifact2 = temp_path / "result2.txt"
                artifact2.write_text("Processing complete")
                stage.add_artifact(artifact2)

                stage.progress("Work completed")

            # Verify artifacts were created
            self.assertTrue(artifact1.exists())
            self.assertTrue(artifact2.exists())

            # Verify heartbeat log was created
            heartbeat_log = temp_path / "heartbeat.log"
            self.assertTrue(heartbeat_log.exists())

            # Check log content
            log_content = heartbeat_log.read_text()
            self.assertIn("integration_test", log_content)


if __name__ == "__main__":
    unittest.main()
