"""Inference / scoring function: text in, JSON out."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from detector.model import DebertaV3ForSlopDetection


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

        # Load correct model class
        try:
            self.model = DebertaV3ForSlopDetection.from_pretrained(str(self.model_dir))
        except (ValueError, KeyError, EnvironmentError):
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))

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
        # Use return_overflowing_tokens to handle sliding windows during inference
        inputs = self.tokenizer(
            normalized,
            max_length=512,
            truncation=True,
            stride=128,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Remove overflow_to_sample_mapping to pass to model
        inputs.pop("overflow_to_sample_mapping", None)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Soft-voting across sliding windows (average probabilities)
            if isinstance(self.model, DebertaV3ForSlopDetection):
                probs = torch.sigmoid(outputs.logits)
                mean_probs = probs.mean(dim=0)
                score_ai = float(mean_probs[0].cpu().item())
            else:
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                mean_probs = probs.mean(dim=0)
                score_ai = float(mean_probs[1].cpu().item())

        threshold_data = self.thresholds.get(threshold_mode, {})
        threshold = threshold_data.get("threshold", 0.5)
        label = "ai" if score_ai >= threshold else "human"
        confidence = self._get_confidence(score_ai, threshold)

        # Approximate total tokens from first dimension of first window + step size * (num_windows - 1)
        num_windows = inputs["input_ids"].shape[0]
        n_tokens = int(inputs["attention_mask"][0].sum().item())
        if num_windows > 1:
             n_tokens += (num_windows - 1) * (512 - 128) # Approximate

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
        """Score multiple texts in batched forward passes. Much faster than calling score() in a loop."""
        threshold_data = self.thresholds.get(threshold_mode, {})
        threshold = threshold_data.get("threshold", 0.5)
        results: list[dict] = []

        for i in range(0, len(texts), batch_size):
            batch_norm = [self.normalize_text(t) for t in texts[i : i + batch_size]]

            # Separate valid texts from empty ones (preserve original positions)
            valid_idx = [j for j, t in enumerate(batch_norm) if t]
            valid_texts = [batch_norm[j] for j in valid_idx]

            # Map from batch-local index → (score_ai, n_tokens)
            scores_map: dict[int, tuple[float, int]] = {}
            if valid_texts:
                inputs = self.tokenizer(
                    valid_texts,
                    max_length=512,
                    truncation=True,
                    stride=128,
                    return_overflowing_tokens=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                sample_mapping = inputs.pop("overflow_to_sample_mapping")
                inputs = inputs.to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    if isinstance(self.model, DebertaV3ForSlopDetection):
                        probs = torch.sigmoid(outputs.logits)
                    else:
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Aggregate across windows for each valid text
                for k, orig_j in enumerate(valid_idx):
                    # Find all windows that belong to this sample
                    window_indices = [idx for idx, m in enumerate(sample_mapping) if m == k]
                    sample_probs = probs[window_indices]

                    if isinstance(self.model, DebertaV3ForSlopDetection):
                        mean_score_ai = float(sample_probs.mean(dim=0)[0].item())
                    else:
                        mean_score_ai = float(sample_probs.mean(dim=0)[1].item())

                    # Approximate token count
                    num_windows = len(window_indices)
                    n_tokens = int(inputs["attention_mask"][window_indices[0]].sum().item())
                    if num_windows > 1:
                        n_tokens += (num_windows - 1) * (512 - 128)

                    scores_map[orig_j] = (mean_score_ai, n_tokens)

            for j, norm_text in enumerate(batch_norm):
                if j not in scores_map:
                    results.append({
                        "error": "empty_input",
                        "label": None,
                        "score_ai": None,
                        "confidence": None,
                        "model_version": self.model_version,
                        "threshold_mode": threshold_mode,
                        "threshold": None,
                        "text_stats": {"chars": 0, "tokens": 0},
                    })
                else:
                    score_ai, n_tokens = scores_map[j]
                    label = "ai" if score_ai >= threshold else "human"
                    results.append({
                        "label": label,
                        "score_ai": round(score_ai, 6),
                        "confidence": round(self._get_confidence(score_ai, threshold), 6),
                        "model_version": self.model_version,
                        "threshold_mode": threshold_mode,
                        "threshold": round(threshold, 6),
                        "text_stats": {"chars": len(norm_text), "tokens": n_tokens},
                    })

        return results
