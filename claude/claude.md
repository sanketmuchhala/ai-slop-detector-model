# Claude Code — Contributor Guide

## Project Intent

- Train a binary text classifier that distinguishes AI-written text from human-written text.
- Evaluate rigorously using the RAID benchmark across domains, generator families, and adversarial attacks.
- Export an inference artifact consumed by a separate browser-extension repo for LinkedIn AI detection.

## Guardrails

**No certainty claims.** The model produces probabilities and threshold-based labels. Never describe output as "definitely AI" or "definitely human." Use language like "scored 0.87 probability of AI at the conservative threshold."

**Minimize false positives as a first-class requirement.** Flagging human text as AI is the most damaging failure mode. The Conservative operating point (FPR <= 1%) exists for this reason. Any change that increases FPR on the human slice must be justified and documented.

**Never store user text in logs by default.** Eval artifacts store sample IDs and scores, not raw text. Error samples store `text_hash` (SHA-256) for deduplication. Inference logs must not persist input text unless explicitly opted in via a config flag.

## Coding Conventions

**Python version:** 3.10+. Use type hints. Use `pathlib` for file paths.

**Core libraries:**

| Purpose | Library |
|---|---|
| Model and tokenizer | `transformers` (Hugging Face) |
| Datasets | `datasets` (Hugging Face) |
| Metrics | `evaluate` (Hugging Face) + `scikit-learn` |
| LoRA / parameter-efficient tuning | `peft` |
| Config validation | `pydantic` v2 |
| ONNX export | `onnx`, `onnxruntime` |
| RAID benchmark | `raid-bench` |

**Reproducibility requirements:**

- Set `seed=42` everywhere: `torch.manual_seed`, `transformers.set_seed`, `numpy`, `random`.
- All hyperparameters live in YAML config files (`configs/`), not hardcoded in scripts.
- Pin all dependency versions in `pyproject.toml`.
- `env.json` saved with each run (torch version, CUDA, git commit).

**Code style:**

- Formatter: `black` (line-length 120).
- Linter: `ruff`.
- No wildcard imports.

## CLI Commands

```bash
make setup            # pip install -e ".[dev]", verify torch+GPU
make train            # CONFIG=configs/baseline.yaml
make eval             # RUN_DIR=runs/latest DATASET=all
make eval-raid-submit # RUN_DIR=runs/latest — generates predictions.json for RAID leaderboard
make calibrate        # RUN_DIR=runs/latest
make export           # RUN_DIR=runs/latest
make infer            # RUN_DIR=runs/latest MODE=conservative TEXT="..."
make train-lora       # shortcut for CONFIG=configs/lora.yaml
make test             # pytest tests/ -v
make test-fast        # pytest tests/ -v -m "not slow"
make lint             # ruff check
make format           # black
make smoke            # full pipeline with smoke config (fast CI check)
make pipeline         # train -> eval -> calibrate -> export (chained)
```

All commands runnable from repo root. All commands idempotent (safe to re-run).

## PR Checklist

Every pull request must confirm:

- [ ] **Metrics updated** — If the model or training pipeline changed, `metrics.json` and `slices.json` reflect the new model.
- [ ] **Eval slices updated** — Slice analysis covers all RAID axes. No axis dropped silently.
- [ ] **No raw text in artifacts** — Grep all output files for text longer than 50 characters. Eval artifacts contain IDs and scores only.
- [ ] **Model card updated** — `MODEL_CARD.md` reflects the current model version, training data, and known limitations.
- [ ] **Config committed** — Any hyperparameter change is reflected in the config file, not just in script arguments.
- [ ] **Tests pass** — `make test-fast` passes without error.
- [ ] **No secrets or credentials** — No API keys, tokens, or paths containing usernames in committed files.

## Artifact Structure

Each training run produces `runs/<run_id>/` containing:

```
config.yaml, env.json, checkpoint/, metrics.json, slices.json,
roc.csv, roc.png, thresholds.json, confusion_*.json,
error_samples.json, export/model_hf/, export/MODEL_CARD.md
```

`thresholds.json` must include `measured_fpr_human` for the conservative threshold. Verify: `measured_fpr_human <= 0.01`.
