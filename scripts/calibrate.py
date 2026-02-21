"""CLI entrypoint for threshold calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from detector.calibrate import calibrate, save_thresholds
from detector.config import load_config
from detector.data.wiki_human_ai import prepare_wiki_dataset
from detector.eval.run_eval import load_checkpoint, score_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate detection thresholds")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--config", type=str, default=None, help="Override config")
    parser.add_argument("--split", type=str, default="validation", help="Data split to calibrate on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or str(Path(args.run_dir) / "config.yaml")
    config = load_config(config_path)
    run_dir = Path(args.run_dir)

    # Load model
    model, tokenizer = load_checkpoint(run_dir / "checkpoint")

    # Load calibration data
    dataset = prepare_wiki_dataset(config.data, config.seed)
    cal_data = dataset[args.split]
    texts = cal_data["text"]
    labels = np.array(cal_data["label"])

    # Score
    scores = score_dataset(model, tokenizer, texts, max_length=config.data.max_length)

    # Calibrate
    thresholds = calibrate(labels, scores, config.calibration, args.split)

    # Save
    save_thresholds(thresholds, run_dir / "thresholds.json")
    print(f"Calibration complete. Thresholds saved to: {run_dir / 'thresholds.json'}")
    print(f"Conservative threshold: {thresholds['conservative']['threshold']:.4f} "
          f"(FPR={thresholds['conservative']['measured_fpr_human']:.4f})")
    print(f"Balanced threshold: {thresholds['balanced']['threshold']:.4f} "
          f"(FPR={thresholds['balanced']['measured_fpr']:.4f})")


if __name__ == "__main__":
    main()
