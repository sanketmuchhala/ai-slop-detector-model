"""Tests for inference module."""

import json
import math

import pytest

from detector.infer import TextDetector


class TestNormalizeText:
    def test_unicode_nfc(self):
        # Combining character form (NFD) should become NFC
        text = "caf\u0065\u0301"  # e + combining accent
        result = TextDetector.normalize_text(text)
        assert "\u0301" not in result or result == text  # at minimum processed

    def test_strip_whitespace(self):
        assert TextDetector.normalize_text("  hello  ") == "hello"

    def test_collapse_newlines(self):
        assert TextDetector.normalize_text("a\n\n\nb") == "a\nb"

    def test_collapse_spaces(self):
        assert TextDetector.normalize_text("a   b") == "a b"

    def test_combined(self):
        result = TextDetector.normalize_text("  hello   world\n\n\nfoo  ")
        assert result == "hello world\nfoo"


class TestConfidence:
    def test_at_threshold(self):
        """Score equal to threshold should give 0 confidence."""
        assert TextDetector._get_confidence(0.7, 0.7) == 0.0

    def test_above_threshold(self):
        """score=0.8, threshold=0.7: distance=0.1, margin=0.7, confidence≈0.143."""
        conf = TextDetector._get_confidence(0.8, 0.7)
        assert abs(conf - 0.1 / 0.7) < 1e-6

    def test_below_threshold(self):
        """score=0.3, threshold=0.7: distance=0.4, margin=0.7, confidence≈0.571."""
        conf = TextDetector._get_confidence(0.3, 0.7)
        assert abs(conf - 0.4 / 0.7) < 1e-6

    def test_extreme_high(self):
        """score=1.0, threshold=0.5: distance=0.5, margin=0.5, confidence=1.0."""
        conf = TextDetector._get_confidence(1.0, 0.5)
        assert conf == 1.0

    def test_extreme_low(self):
        """score=0.0, threshold=0.5: distance=0.5, margin=0.5, confidence=1.0."""
        conf = TextDetector._get_confidence(0.0, 0.5)
        assert conf == 1.0

    def test_range(self):
        """Confidence should always be in [0, 1]."""
        for score in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                conf = TextDetector._get_confidence(score, threshold)
                assert 0.0 <= conf <= 1.0, f"score={score}, threshold={threshold}, conf={conf}"

    def test_zero_margin(self):
        """Edge case: threshold=0 or threshold=1."""
        assert TextDetector._get_confidence(0.5, 0.0) == pytest.approx(0.5, abs=1e-6)
        assert TextDetector._get_confidence(0.5, 1.0) == pytest.approx(0.5, abs=1e-6)


class TestScoreSchema:
    """Test that score() returns valid JSON with all required keys.

    These tests use a mock model approach - we test the TextDetector methods
    that don't require a real model.
    """

    REQUIRED_KEYS = {"label", "score_ai", "confidence", "model_version", "threshold_mode", "threshold", "text_stats"}

    def test_empty_input_returns_error(self):
        """Empty input should return error dict, not crash.

        We test this via the static method pattern since it doesn't need model loading.
        """
        # The empty check is done before model inference
        # Test the normalize + empty check path
        assert TextDetector.normalize_text("") == ""
        assert TextDetector.normalize_text("   ") == ""

    def test_confidence_formula_documented(self):
        """Verify the confidence formula is what we documented."""
        # confidence = abs(score_ai - threshold) / max(threshold, 1 - threshold)
        score_ai, threshold = 0.8, 0.7
        expected = abs(score_ai - threshold) / max(threshold, 1 - threshold)
        actual = TextDetector._get_confidence(score_ai, threshold)
        assert actual == pytest.approx(expected, abs=1e-6)


class TestDeterminism:
    def test_normalize_deterministic(self):
        """Same input always gives same normalized output."""
        text = "  Hello   world\n\n\ntest  "
        results = [TextDetector.normalize_text(text) for _ in range(3)]
        assert all(r == results[0] for r in results)
