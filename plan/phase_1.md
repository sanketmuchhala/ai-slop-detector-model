# Phase 1: Baseline Classifier, RAID Benchmarking, and Inference Contract

## Goals

- Train a binary AI-vs-human text classifier using `bert-base-cased`.
- Evaluate rigorously on RAID across all available slicing axes.
- Establish two operating thresholds (Conservative FPR<=1%, Balanced TPR=90%).
- Export an inference-ready artifact with a defined JSON contract.

## Non-goals

- Production-grade serving infrastructure.
- Multi-language support.
- Authorship attribution or watermark detection.
- Browser extension integration (handled in separate repo).
- Continuous training or feedback loops.

## Deliverables

| Artifact | Path | Format |
|---|---|---|
| Trained model checkpoint | `runs/<run_id>/checkpoint/` | HF model + tokenizer |
| Evaluation report | `runs/<run_id>/metrics.json` | JSON |
| ROC curve plot | `runs/<run_id>/roc.png` | PNG |
| Per-slice breakdown | `runs/<run_id>/slices.json` | JSON |
| Threshold config | `runs/<run_id>/thresholds.json` | JSON |
| Exported HF checkpoint | `runs/<run_id>/export/model_hf/` | HF format |
| ONNX model (optional) | `runs/<run_id>/export/model_onnx/` | ONNX |
| Model card | `runs/<run_id>/export/MODEL_CARD.md` | Markdown |

---

## Implementation (completed)

### Project structure

```
detector/
├── __init__.py
├── config.py           # Pydantic v2 config models, load_config()
├── data/
│   ├── wiki_human_ai.py  # HF dataset loader, prepare/tokenize
│   └── raid.py           # RAID loader, detector fn builder
├── eval/
│   ├── metrics.py        # AUC, ROC, ECE, confusion, thresholds
│   ├── slices.py         # Per-slice analysis across RAID axes
│   └── run_eval.py       # Eval orchestration (wiki, RAID labeled, RAID submission)
├── train.py              # HF Trainer-based training, LoRA support
├── calibrate.py          # Conservative + balanced threshold selection
├── export.py             # HF checkpoint + ONNX export + model card
└── infer.py              # TextDetector class, normalization, scoring
```

### RAID split handling

- **Labeled eval** (`--dataset raid_labeled`): Uses RAID train/extra splits. Computes full metrics + slices.
- **Leaderboard submission** (`--dataset raid_test_unlabeled`): Generates `predictions.json` only. No metrics (no labels).
- RAID is benchmark-only by default. Training uses wiki_human_ai.

### Key metrics

- AUC-ROC, TPR at fixed FPR points, FPR at fixed TPR points
- ECE (Expected Calibration Error) for probability trustworthiness
- Per-slice analysis across: domain, model, decoding, attack

### Calibration

- `thresholds.json` includes: `max_fpr_target`, `measured_fpr_human`, `target_tpr`, `measured_fpr`
- Calibration set: wiki_human_ai validation split
- Conservative: threshold where `measured_fpr_human <= 0.01`

### Confidence formula

```
confidence = abs(score_ai - threshold) / max(threshold, 1 - threshold)
```

Normalized by available margin. Stays in [0, 1].

### Smoke testing

`configs/smoke.yaml`: 200 samples, 1 epoch, max_length=128. Use `make smoke` for quick CI validation.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| RAID download issues | `raid-bench` via pip; fallback message with exact install command |
| Baseline AUC below 0.85 | Check data pipeline; try larger model; increase training data |
| LoRA degrades quality | Fall back to full fine-tune; LoRA is optional |
| Overfitting to wiki distribution | Validate on RAID; monitor per-domain variance |
| Threshold calibration unstable | Use at least 2000 human samples; report measured FPR explicitly |

## Definition of Done

- [x] `make train` runs end-to-end and produces a saved checkpoint.
- [x] `make eval` produces metrics.json, slices.json, roc.png.
- [x] `make calibrate` produces thresholds.json with `measured_fpr_human <= 0.01`.
- [x] `make export` produces HF checkpoint and model card.
- [x] `make infer` returns valid JSON with all schema keys.
- [x] Slice analysis covers all RAID axes present in data.
- [x] No raw text stored in any eval artifact or log.
- [x] `make test-fast` passes all unit tests.
