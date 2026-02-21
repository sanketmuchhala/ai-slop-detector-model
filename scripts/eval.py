"""CLI entrypoint for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from detector.config import load_config
from detector.eval.run_eval import run_full_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AI text detector")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--config", type=str, default=None, help="Override config (default: use run's config.yaml)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["wiki", "raid_labeled", "raid_test_unlabeled", "all"],
        help="Which dataset to evaluate on",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or str(Path(args.run_dir) / "config.yaml")
    config = load_config(config_path)
    run_dir = Path(args.run_dir)

    run_full_eval(config, run_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
