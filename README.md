# SlopGuard
LinkedIn AI slop detector. Scores posts as Human vs AI-written and shows a percentage badge directly on the post.

## What this does
- Adds a small label on LinkedIn posts in your feed and on post pages.
- Shows an “AI-likelihood” score as a percentage.
- Lets you tune sensitivity and hide labels if you want.
- Works locally in your browser for fast tagging; optional server mode for heavier models.

## Why
LinkedIn is flooded with template posts and LLM-generated content.
This extension helps you:
- Filter what you read.
- Calibrate trust signals.
- Spot engagement bait faster.

This is a classifier, not a truth machine. Treat scores as a signal.

## How it works
1. The browser extension reads the visible post text on linkedin.com.
2. It runs a detector model and outputs:
   - AI likelihood (0–100)
   - Human likelihood (0–100)
   - Confidence band (low, medium, high)
3. The extension injects a badge into the LinkedIn UI next to the post header.

## Model approach
We train a lightweight text classifier focused on real LinkedIn writing.
Planned pipeline:
- Dataset
  - Human: verified posts, long-form originals, varied styles
  - AI: LLM generations, rewrites, templated posts, prompt variants
- Features
  - Token patterns, repetition, burstiness, structure signals
  - Optional embedding features for robustness
- Training
  - Baseline: logistic regression or small transformer encoder
  - Calibrated probabilities (temperature scaling or isotonic)
- Evaluation
  - Accuracy, ROC-AUC, precision/recall
  - False positive rate on niche domains
  - Stress tests on edited AI text and bilingual posts

## Extension UI
Badge states:
- Human-leaning: “Human” + percent
- AI-leaning: “AI” + percent
- Uncertain: “Unclear” + percent range

Controls:
- Threshold slider (more strict vs more forgiving)
- Toggle badges on/off
- Debug mode to show feature hints and confidence

## Tech stack
- Extension: Manifest V3, TypeScript, DOM injection
- Model:
  - Option A: ONNX model shipped with the extension
  - Option B: Local server inference (FastAPI/Node) for bigger models
- Packaging: pnpm or npm, build scripts for Chrome

## Repo structure (suggested)
- extension/
  - src/
  - manifest.json
  - content-script/
  - options-page/
- model/
  - data/
  - training/
  - eval/
  - export/
- api/ (optional)
  - inference server
- docs/
  - screenshots
  - design notes

## Getting started (planned)
1. Clone repo
2. Install deps
3. Build extension
4. Load unpacked extension in Chrome
5. Open LinkedIn feed and see badges

## Roadmap
- v0: Heuristic baseline + simple UI badge
- v1: Trained model + calibrated score + thresholds
- v2: ONNX in-browser inference
- v3: Per-user tuning and feedback loop (opt-in)
- v4: Language support beyond English

## Ethics and safety
- No auto-actions. This only labels content.
- No selling private data.
- Default to on-device inference.
- Clear “uncertain” state to reduce overconfidence.

## Contributing
PRs welcome:
- DOM injection reliability across LinkedIn layouts
- Dataset curation and evaluation harness
- Model export and runtime performance

## License
MIT
