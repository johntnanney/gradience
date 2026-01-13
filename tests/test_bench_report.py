import json
import tempfile
import unittest
from pathlib import Path

from gradience.bench.report import render_markdown, write_report


class TestBenchReport(unittest.TestCase):
    def test_render_markdown_smoke(self):
        report = {
            "bench_version": "0.1",
            "model": "distilbert-base-uncased",
            "task": "glue/sst2",
            "probe": {"rank": 16, "params": 10240, "accuracy": 0.91},
            "compressed": {
                "uniform_median": {
                    "rank": 4,
                    "params": 2560,
                    "accuracy": 0.905,
                    "delta_vs_probe": -0.005,
                    "param_reduction": 0.75,
                    "verdict": "PASS",
                }
            },
            "summary": {"best_compression": "uniform_median"},
        }
        md = render_markdown(report)
        self.assertIn("Gradience Bench v0.1", md)
        self.assertIn("distilbert-base-uncased", md)
        self.assertIn("uniform_median", md)

    def test_write_report_writes_valid_json_and_md(self):
        report = {
            "bench_version": "0.1",
            "model": "distilbert-base-uncased",
            "task": "glue/sst2",
            "probe": {"rank": 16, "params": 10240, "accuracy": 0.91},
            "compressed": {},
            "summary": {},
        }
        with tempfile.TemporaryDirectory() as td:
            json_path, md_path = write_report(td, report)

            self.assertTrue(Path(json_path).exists())
            self.assertTrue(Path(md_path).exists())

            loaded = json.loads(Path(json_path).read_text(encoding="utf-8"))
            self.assertEqual(loaded["bench_version"], "0.1")

            md = Path(md_path).read_text(encoding="utf-8")
            self.assertIn("# Gradience Bench v0.1", md)


if __name__ == "__main__":
    unittest.main()