import json
import os
import tempfile
import unittest

from gradience.vnext.telemetry import TelemetryWriter


class TestTelemetryTextGuardrail(unittest.TestCase):
    def test_redacts_long_strings_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "run.jsonl")
            w = TelemetryWriter(p)  # default: redact long strings
            long = "x" * 2000
            w.log("metrics", step=0, kind="test", metrics={"ppl": 1.0}, prompt=long)
            w.close()

            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()
            self.assertNotIn(long, raw, "Long raw string should not appear in telemetry JSONL")

            obj = json.loads(raw.strip())
            self.assertTrue(str(obj["prompt"]).startswith("[REDACTED"), "Expected prompt to be redacted")

    def test_raise_mode_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "run.jsonl")
            w = TelemetryWriter(p, on_text_violation="raise")
            try:
                with self.assertRaises(ValueError):
                    w.log("train_step", step=1, prompt="x" * 2000)
            finally:
                w.close()

    def test_allow_text_allows_long_strings(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "run.jsonl")
            long = "y" * 2000
            w = TelemetryWriter(p, allow_text=True)
            w.log("train_step", step=1, prompt=long)
            w.close()

            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()
            obj = json.loads(raw.strip())
            self.assertEqual(obj["prompt"], long)
