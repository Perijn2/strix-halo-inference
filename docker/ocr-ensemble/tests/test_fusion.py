"""Regression tests for OCR ensemble fusion behavior.

Author: Perijn
Summary: Verifies that incomplete validation cannot be accepted and that weighted corroboration uses the best classical witness.
Usage: Run from docker/ocr-ensemble with `python -m unittest discover -s tests -v` after installing requirements.txt.
"""

from __future__ import annotations

import unittest

from app.engines import EngineResult
from app.fusion import ACCEPT_WEIGHTED, REVIEW_ENTROPY, FusionEngine


class FusionEngineTests(unittest.TestCase):
    """Verify review routing rules that protect validated OCR responses."""

    def setUp(self) -> None:
        """Use deterministic entropy so each test isolates its decision rule."""
        self.engine = FusionEngine()
        self.engine._consensus = lambda texts, task_type: 0.0

    def test_failed_engine_forces_review(self) -> None:
        """Reject automatic acceptance when a full-ensemble witness fails."""
        result = self.engine.fuse(
            [
                EngineResult("mineru", "vlm", "confirmed text"),
                EngineResult("ppocrv5", "classical", "confirmed text"),
                EngineResult("tesseract", "classical", "confirmed text"),
                EngineResult("surya", "vlm", error="worker unavailable"),
            ]
        )

        self.assertEqual(REVIEW_ENTROPY, result.verdict)
        self.assertTrue(result.review)
        self.assertIn("one or more validation engines", result.reasons)

    def test_partial_engine_selection_forces_review(self) -> None:
        """Reject diagnostic subsets as insufficient for a validated acceptance."""
        result = self.engine.fuse(
            [
                EngineResult("mineru", "vlm", "confirmed text"),
                EngineResult("ppocrv5", "classical", "confirmed text"),
            ],
            allow_auto_accept=False,
        )

        self.assertEqual(REVIEW_ENTROPY, result.verdict)
        self.assertTrue(result.review)
        self.assertIn("partial engine selection", result.reasons)

    def test_weighted_accept_uses_best_classical_witness(self) -> None:
        """Accept when MinerU matches one readable classical witness."""
        result = self.engine.fuse(
            [
                EngineResult("mineru", "vlm", "alpha bravo charlie"),
                EngineResult("ppocrv5", "classical", "alpha bravo charlie"),
                EngineResult("tesseract", "classical", "alpha bravo charlie three"),
                EngineResult("surya", "vlm", "unrelated layout annotation"),
            ]
        )

        self.assertEqual(ACCEPT_WEIGHTED, result.verdict)
        self.assertFalse(result.review)
        self.assertIn(
            "MinerU corroborated by at least one classical engine", result.reasons
        )


if __name__ == "__main__":
    unittest.main()
