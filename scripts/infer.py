"""CLI entrypoint for inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from detector.infer import TextDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on text")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to exported model directory")
    parser.add_argument("--thresholds", type=str, required=True, help="Path to thresholds.json")
    parser.add_argument("--mode", type=str, default="conservative", choices=["conservative", "balanced"])
    parser.add_argument("--text", type=str, default=None, help="Text to classify")
    parser.add_argument("--input-file", type=str, default=None, help="Input JSONL file (each line has 'text' key)")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSONL file (default: stdout)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = TextDetector(
        model_dir=args.model_dir,
        thresholds_path=args.thresholds,
    )

    # Single text mode
    if args.text:
        result = detector.score(args.text, threshold_mode=args.mode)
        print(json.dumps(result, indent=2))
        return

    # File mode (JSONL input)
    if args.input_file:
        texts = []
        with open(args.input_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(obj["text"])
        results = detector.score_batch(texts, threshold_mode=args.mode)
    else:
        # Stdin mode
        text = sys.stdin.read().strip()
        if not text:
            print(json.dumps({"error": "empty_input"}))
            return
        results = [detector.score(text, threshold_mode=args.mode)]

    # Output
    out = open(args.output_file, "w") if args.output_file else sys.stdout
    for result in results:
        out.write(json.dumps(result) + "\n")
    if args.output_file:
        out.close()


if __name__ == "__main__":
    main()
