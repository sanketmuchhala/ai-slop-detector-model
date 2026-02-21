"""Inference / scoring function: text in, JSON out."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class TextDetector:
    """Stateful inference wrapper. Loads model + thresholds once, scores many texts."""

    def __init__(
        self,
        model_dir: str | Path,
        thresholds_path: str | Path,
        device: str | None = None,
        model_version: str = "v0.1.0",
    ):
        self.model_dir = Path(model_dir)
        self.model_version = model_version

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model = self.model.to(self.device)
        self.model.eval()

        with open(thresholds_path) as f:
            self.thresholds = json.load(f)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize input text.

        1. Unicode NFC normalization
        2. Strip leading/trailing whitespace
        3. Collapse multiple newlines to single newline
        4. Collapse multiple spaces to single space
        """
        text = unicodedata.normalize("NFC", text)
        text = text.strip()
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text

    @staticmethod
    def _get_confidence(score_ai: float, threshold: float) -> float:
        """Compute confidence as normalized distance from threshold.

        Formula: abs(score_ai - threshold) / max(threshold, 1 - threshold)

        This normalizes by the available margin on the relevant side,
        keeping the value in [0, 1]. Better than raw distance which
        underestimates near 0.5 and overestimates at extremes.
        """
        margin = max(threshold, 1.0 - threshold)
        if margin == 0:
            return 0.0
        return min(abs(score_ai - threshold) / margin, 1.0)

    def score(
        self,
        text: str,
        threshold_mode: Literal["conservative", "balanced"] = "conservative",
    ) -> dict:
        """Score a single text. Returns the full inference JSON.

        Output schema:
        {
            "label": "ai" | "human",
            "score_ai": float,
            "confidence": float,
            "model_version": str,
            "threshold_mode": str,
            "threshold": float,
            "text_stats": {"chars": int, "tokens": int}
        }
        """
        if not text or not text.strip():
            return {
                "error": "empty_input",
                "label": None,
                "score_ai": None,
                "confidence": None,
                "model_version": self.model_version,
                "threshold_mode": threshold_mode,
                "threshold": None,
                "text_stats": {"chars": 0, "tokens": 0},
            }

        normalized = self.normalize_text(text)
        inputs = self.tokenizer(
            normalized,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            score_ai = float(probs[0, 1].cpu().item())

        threshold_data = self.thresholds.get(threshold_mode, {})
        threshold = threshold_data.get("threshold", 0.5)
        label = "ai" if score_ai >= threshold else "human"
        confidence = self._get_confidence(score_ai, threshold)

        n_tokens = len(inputs["input_ids"][0])

        return {
            "label": label,
            "score_ai": round(score_ai, 6),
            "confidence": round(confidence, 6),
            "model_version": self.model_version,
            "threshold_mode": threshold_mode,
            "threshold": round(threshold, 6),
            "text_stats": {"chars": len(normalized), "tokens": n_tokens},
        }

    def score_batch(
        self,
        texts: list[str],
        threshold_mode: Literal["conservative", "balanced"] = "conservative",
        batch_size: int = 32,
    ) -> list[dict]:
        """Score multiple texts. Returns list of inference JSONs."""
        results = []
        for text in texts:
            results.append(self.score(text, threshold_mode))
        return results
