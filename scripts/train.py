"""CLI entrypoint for training."""

from __future__ import annotations

import argparse

import yaml

from detector.config import Config, apply_overrides, load_config
from detector.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI text detector")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values: key=value (dot notation)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.override:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        raw = apply_overrides(raw, args.override)
        config = Config(**raw)

    run_dir = train(config)
    print(f"Artifacts at: {run_dir}")


if __name__ == "__main__":
    main()
