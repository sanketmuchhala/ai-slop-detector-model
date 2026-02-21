"""CLI entrypoint for model export."""

from __future__ import annotations

import argparse
from pathlib import Path

from detector.config import load_config
from detector.export import export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to HF + ONNX")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--config", type=str, default=None, help="Override config")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or str(Path(args.run_dir) / "config.yaml")
    config = load_config(config_path)
    if args.no_onnx:
        config.export.onnx = False

    run_dir = Path(args.run_dir)
    export_dir = export(config, run_dir)
    print(f"Export complete. Artifacts at: {export_dir}")


if __name__ == "__main__":
    main()
