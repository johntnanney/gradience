import unittest
from pathlib import Path


class TestVNextTelemetryContract(unittest.TestCase):
    """Minimal contract tests for gradience.vnext.telemetry/v1.

    These tests intentionally lock down the *spine* of the schema so the
    vNext tools (`check`, `monitor`, `audit`) remain interoperable.
    """

    def setUp(self) -> None:
        self.fixture_path = (
            Path(__file__).resolve().parent / "fixtures" / "vnext_minimal.jsonl"
        )

    def test_validate_passes_on_golden_fixture(self) -> None:
        from gradience.vnext.telemetry_reader import TelemetryReader

        reader = TelemetryReader(self.fixture_path, strict_schema=True)
        issues = reader.validate()
        self.assertEqual(issues, [], msg=f"Unexpected telemetry issues: {issues}")

    def test_summarize_gap_math(self) -> None:
        from gradience.vnext.telemetry_reader import TelemetryReader

        reader = TelemetryReader(self.fixture_path, strict_schema=True)
        s = reader.summarize()

        self.assertIsNotNone(s.train.ppl)
        self.assertIsNotNone(s.test.ppl)
        # Fixture uses train_ppl=2.0, test_ppl=3.0
        self.assertIsNotNone(s.gap)
        self.assertAlmostEqual(float(s.gap), 1.5, places=7)

    def test_latest_config_roundtrip(self) -> None:
        from gradience.vnext.telemetry_reader import TelemetryReader

        reader = TelemetryReader(self.fixture_path, strict_schema=True)
        cfg = reader.latest_config()
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg.model_name, "toy-model")
        self.assertEqual(cfg.dataset_name, "toy-dataset")
        self.assertEqual(cfg.lora.r, 8)
        self.assertEqual(cfg.optimizer.lr, 0.00005)


if __name__ == "__main__":
    unittest.main()
