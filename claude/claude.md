# Claude Code — Contributor Guide

## Project Intent

- Train a binary text classifier that distinguishes AI-written text from human-written text.
- Evaluate rigorously using the RAID benchmark across domains, generator families, and adversarial attacks.
- Export an inference artifact consumed by a separate browser-extension repo for LinkedIn AI detection.

## Guardrails

**No certainty claims.** The model produces probabilities and threshold-based labels. Never describe output as "definitely AI" or "definitely human." Use language like "scored 0.87 probability of AI at the conservative threshold."

**Minimize false positives as a first-class requirement.** Flagging human text as AI is the most damaging failure mode. The Conservative operating point (FPR ≤ 1%) exists for this reason. Any change that increases FPR on the human slice must be justified and documented.

**Never store user text in logs by default.** Eval artifacts store sample IDs and scores, not raw text. Inference logs must not persist input text unless explicitly opted in via a config flag. This applies to training scripts, eval scripts, and any future serving layer.

## Coding Conventions

**Python version:** 3.10+. Use type hints. Use `pathlib` for file paths.

**Core libraries:**

| Purpose | Library |
|---|---|
| Model and tokenizer | `transformers` (Hugging Face) |
| Datasets | `datasets` (Hugging Face) |
| Metrics | `evaluate` (Hugging Face) + `scikit-learn` |
| LoRA / parameter-efficient tuning | `peft` |
| ONNX export | `onnx`, `onnxruntime` |
| Experiment tracking (optional) | `wandb` or `tensorboard` |

**Reproducibility requirements:**

- Set `seed=42` everywhere: `torch.manual_seed`, `transformers.set_seed`, `numpy`, `random`.
- Enable deterministic mode: `torch.use_deterministic_algorithms(True)` where possible.
- All hyperparameters must live in config files (`training/config.yaml` or similar), not hardcoded in scripts.
- Pin all dependency versions in `requirements.txt`.

**Code style:**

- Formatter: `black` (default settings).
- Linter: `ruff`.
- No wildcard imports.
- Docstrings only where the function signature is not self-explanatory.

## Required CLI Commands

These are `Makefile` targets. Implementations are placeholders until code is added.

```bash
make setup      # Create venv, install dependencies, download data
make train      # Run training with config from training/config.yaml
make eval       # Run evaluation suite, produce metrics.json, slices.json, roc.png
make export     # Convert best checkpoint to ONNX, validate equivalence
```

All commands must be runnable from the repo root. All commands must be idempotent (safe to re-run).

## PR Checklist

Every pull request must confirm:

- [ ] **Metrics updated** — If the model or training pipeline changed, `eval/results/metrics.json` and `eval/results/slices.json` reflect the new model.
- [ ] **Eval slices updated** — Slice analysis covers all RAID axes. No axis dropped silently.
- [ ] **No raw text in artifacts** — Grep all output files for text longer than 50 characters. Eval artifacts contain IDs and scores only.
- [ ] **Model card updated** — `export/model_card.md` reflects the current model version, training data, and known limitations.
- [ ] **Config committed** — Any hyperparameter change is reflected in the config file, not just in script arguments.
- [ ] **Tests pass** — `make eval` runs without error on the current checkpoint.
- [ ] **No secrets or credentials** — No API keys, tokens, or paths containing usernames in committed files.
