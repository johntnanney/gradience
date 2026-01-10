import subprocess
import sys
from pathlib import Path
import unittest

class TestCLISmoke(unittest.TestCase):
    def test_monitor_fixture(self):
        fixture = Path("tests/fixtures/vnext_minimal.jsonl")
        self.assertTrue(fixture.exists(), "Missing fixture JSONL")

        p = subprocess.run(
            [sys.executable, "-m", "gradience", "monitor", str(fixture)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr)

if __name__ == "__main__":
    unittest.main()
