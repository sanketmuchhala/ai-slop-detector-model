"""RAID dataset loader using the official RAID Python API."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

RAID_INSTALL_MSG = (
    "Could not import 'raid'. Install it with:\n"
    "  pip install raid-bench\n"
    "Or from source:\n"
    "  pip install git+https://github.com/liamdugan/raid.git"
)


def _check_raid_installed():
    """Check if RAID is installed and provide helpful error if not."""
    try:
        import raid  # noqa: F401

        return True
    except ImportError:
        raise ImportError(RAID_INSTALL_MSG)


def load_raid_dataframe(
    split: Literal["train", "extra", "test"] = "train",
    include_adversarial: bool = True,
) -> pd.DataFrame:
    """Load RAID data via the official API. Returns raw pandas DataFrame.

    For labeled splits (train, extra): model=="human" indicates human text.
    For test split: no labels available, use for leaderboard submission only.
    """
    _check_raid_installed()
    from raid.utils import load_data

    if split == "test":
        return load_data(split="test")

    kwargs = {"split": split}
    if not include_adversarial:
        kwargs["include_adversarial"] = False

    return load_data(**kwargs)


def raid_df_to_labels(df: pd.DataFrame) -> np.ndarray:
    """Derive binary labels from RAID DataFrame. model=="human" -> 0, else -> 1."""
    return (df["model"] != "human").astype(int).values


def build_raid_detector_fn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    batch_size: int = 32,
    device: str | None = None,
) -> Callable[[list[str]], list[float]]:
    """Build a detector function compatible with raid.run_detection.

    Returns a function: list[str] -> list[float]
    Each float is P(AI) in [0, 1] computed via softmax on logits[:, 1].
    """
    if device is None:
        device = str(model.device) if hasattr(model, "device") else "cpu"

    model.eval()

    def detector_fn(texts: list[str]) -> list[float]:
        all_scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                ai_probs = probs[:, 1].cpu().tolist()
                all_scores.extend(ai_probs)

        return all_scores

    return detector_fn


def prepare_raid_for_training(
    split: Literal["train", "extra"] = "train",
    include_adversarial: bool = True,
    seed: int = 42,
):
    """Load a RAID labeled split as a balanced HF Dataset for training.

    AI samples are undersampled to 1:1 human:AI ratio so class balance
    matches the wiki_human_ai training data.

    Returns Dataset with columns: text (str), label (int 0=human 1=AI).
    """
    import datasets as hf_datasets

    df = load_raid_dataframe(split=split, include_adversarial=include_adversarial)

    human_df = df[df["model"] == "human"][["generation"]].copy().rename(columns={"generation": "text"})
    human_df["label"] = 0

    ai_df = df[df["model"] != "human"][["generation"]].copy().rename(columns={"generation": "text"})
    ai_df["label"] = 1

    # Balance: keep all human, subsample AI to match
    n = min(len(human_df), len(ai_df))
    human_df = human_df.sample(n=n, random_state=seed)
    ai_df = ai_df.sample(n=n, random_state=seed)

    combined = pd.concat([human_df, ai_df], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return hf_datasets.Dataset.from_pandas(combined, preserve_index=False)


def run_raid_evaluation(
    detector_fn: Callable[[list[str]], list[float]],
    split: Literal["train", "extra"] = "train",
    include_adversarial: bool = True,
) -> dict:
    """Run full RAID evaluation using the official API.

    Only works on labeled splits (train, extra). Not on test split.
    """
    _check_raid_installed()
    from raid import run_detection, run_evaluation

    df = load_raid_dataframe(split=split, include_adversarial=include_adversarial)
    predictions = run_detection(detector_fn, df)
    results = run_evaluation(predictions, df)
    return results
