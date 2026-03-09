"""
Unit tests for the heartbeat system used in bench protocols.

Tests the keepalive heartbeat logger that prevents SSH timeouts during
long-running phases like evaluation, auditing, and report generation.
"""

import tempfile
import time
import unittest
from pathlib import Path

from gradience.bench.heartbeat import heartbeat_stage, start_heartbeat, stop_heartbeat, update_stage


class TestHeartbeatSystem(unittest.TestCase):
    """Test heartbeat system functionality."""

    def test_basic_heartbeat_lifecycle(self):
        """Test basic heartbeat start and stop functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Start heartbeat
            start_heartbeat("test_stage", output_dir=temp_path, interval=1)

            # Let it run briefly
            time.sleep(2.5)

            # Stop heartbeat
            stop_heartbeat()

            # Verify log file exists
            log_file = temp_path / "heartbeat.log"
            self.assertTrue(log_file.exists(), "Heartbeat log file should be created")

            # Verify log content
            with open(log_file) as f:
                content = f.read()
                self.assertIn("Starting stage: test_stage", content)
                self.assertIn("STAGE END: test_stage", content)
                self.assertIn("[heartbeat]", content)

    def test_context_manager(self):
        """Test heartbeat context manager functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with heartbeat_stage("context_test", output_dir=temp_path, interval=1):
                time.sleep(2.5)

            # Verify log file exists
            log_file = temp_path / "heartbeat.log"
            self.assertTrue(log_file.exists(), "Context manager should create log file")

            # Verify completion message
            with open(log_file) as f:
                content = f.read()
                self.assertIn("Starting stage: context_test", content)
                self.assertIn("STAGE END: context_test", content)

    def test_stage_update(self):
        """Test stage name updates without restarting heartbeat."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            start_heartbeat("initial_stage", output_dir=temp_path, interval=1)
            time.sleep(1.5)

            update_stage("updated_stage")
            time.sleep(1.5)

            stop_heartbeat()

            # Verify both stages appear in log
            log_file = temp_path / "heartbeat.log"
            with open(log_file) as f:
                content = f.read()
                self.assertIn("initial_stage", content)
                self.assertIn("updated_stage", content)
                self.assertIn("stage transition", content)

    def test_no_log_file_mode(self):
        """Test heartbeat without log file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Disable log file
            start_heartbeat("no_log_test", output_dir=temp_path, interval=1, log_to_file=False)
            time.sleep(1.5)
            stop_heartbeat()

            # Verify no log file created
            log_file = temp_path / "heartbeat.log"
            self.assertFalse(log_file.exists(), "No log file should be created when disabled")

    def test_error_handling(self):
        """Test that heartbeat handles errors gracefully."""
        # Test with invalid output directory (should not crash heartbeat, just disable logging)
        start_heartbeat("error_test", output_dir="/nonexistent/invalid/path", interval=1)
        time.sleep(1.5)
        stop_heartbeat()
        # Should complete without exceptions - logging will be disabled but heartbeat continues

    def test_multiple_start_stop(self):
        """Test that multiple start/stop cycles work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # First cycle
            start_heartbeat("cycle1", output_dir=temp_path, interval=1)
            time.sleep(1.5)
            stop_heartbeat()

            # Second cycle (should not interfere with first)
            start_heartbeat("cycle2", output_dir=temp_path, interval=1)
            time.sleep(1.5)
            stop_heartbeat()

            # Verify both cycles logged
            log_file = temp_path / "heartbeat.log"
            with open(log_file) as f:
                content = f.read()
                self.assertIn("cycle1", content)
                self.assertIn("cycle2", content)
                # Should have 2 start messages and 2 completion messages
                self.assertEqual(content.count("Starting stage:"), 2)
                self.assertEqual(content.count("STAGE END:"), 2)


if __name__ == "__main__":
    unittest.main()
