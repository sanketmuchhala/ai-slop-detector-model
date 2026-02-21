# AI Slop Detector — Model Repo

## What this is

Binary text classifier that scores whether text is AI-written or human-written. Outputs calibrated probabilities and threshold-based labels. Trained on `gouwsxander/wikipedia-human-ai`, benchmarked against RAID. The exported artifact (HF checkpoint + optional ONNX) is consumed by a separate browser-extension repo for LinkedIn AI detection.

## Scope

**In scope (Phase 1)**

- Text-only detection (no images, metadata, or user history).
- LinkedIn as the primary deployment surface.
- Multi-domain evaluation using RAID (news, Wikipedia, abstracts, reviews, etc.).
- Exported inference artifact with a defined JSON contract.
- Conservative operating mode: FPR <= 1% on human text.

**Out of scope (Phase 1)**

- Authorship certainty claims — the model produces probabilities, not verdicts.
- Identity detection or attribution to a specific person or LLM.
- Watermark detection or embedding.

## Quick start

```bash
make setup                                    # install deps
make train CONFIG=configs/smoke.yaml          # fast smoke test (1 epoch, 200 samples)
make train CONFIG=configs/baseline.yaml       # full baseline training
make eval RUN_DIR=runs/<run_id> DATASET=wiki  # evaluate on wiki test set
make eval RUN_DIR=runs/<run_id> DATASET=raid_labeled  # RAID benchmark
make calibrate RUN_DIR=runs/<run_id>          # compute thresholds
make export RUN_DIR=runs/<run_id>             # export HF + ONNX
make infer RUN_DIR=runs/<run_id> TEXT="Your text here"  # inference
```

## Architecture

- **detector/data/** — Ingest wikipedia-human-ai (training) and RAID (benchmark). Unified schema: text, label, domain, generator, attack.
- **detector/train.py** — Fine-tune `bert-base-cased` with HF Trainer. Supports full fine-tune and LoRA (via PEFT).
- **detector/eval/** — ROC, AUC, ECE, threshold sweeps, per-slice analysis across RAID axes (domain, generator family, decoding strategy, adversarial attack).
- **detector/calibrate.py** — Select thresholds: Conservative (FPR <= 1%) and Balanced (TPR = 90%).
- **detector/export.py** — Save HF checkpoint, optional ONNX export, model card generation.
- **detector/infer.py** — Deterministic scoring: text in, JSON out. Handles normalization, tokenization, threshold application.

RAID is the **primary evaluation benchmark**, not the training source (by default). Training data is `gouwsxander/wikipedia-human-ai`. Optional `train_on_raid: true` flag for experiments.

## Data

### Training: wikipedia-human-ai
[`gouwsxander/wikipedia-human-ai`](https://huggingface.co/datasets/gouwsxander/wikipedia-human-ai) — 9,970 Wikipedia paragraphs paired with AI rewrites. Split 80/10/10 train/val/test with label stratification.

### Evaluation: RAID
The RAID dataset ([paper](https://arxiv.org/abs/2405.07940), [repo](https://github.com/liamdugan/raid)) covers 8+ domains, 11+ generator models, multiple decoding strategies, and 11+ adversarial attacks. Two evaluation modes:

- **Labeled eval**: RAID train/extra splits (have labels) — computes FPR, TPR, AUC, slice analysis.
- **Leaderboard submission**: RAID test split (no labels) — generates `predictions.json` only.

## Metrics and evaluation

| Metric | Description |
|---|---|
| AUC-ROC | Overall ranking quality |
| TPR at fixed FPR | TPR at FPR = 0.1%, 0.5%, 1%, 2%, 5% |
| FPR at fixed TPR | FPR at TPR = 80%, 90%, 95% |
| ECE | Expected Calibration Error (probability trustworthiness) |

### Evaluation slices

| Slice | Grouping key |
|---|---|
| Overall | — |
| By domain | `domain` |
| By generator model | `model` |
| By decoding strategy | `decoding` |
| By adversarial attack | `attack` |

### Operating points

| Mode | Definition |
|---|---|
| **Conservative** | Threshold where FPR <= 1% on human slice |
| **Balanced** | Threshold where TPR = 90%; report FPR |

## Inference JSON contract

```json
{
  "label": "ai",
  "score_ai": 0.87,
  "confidence": 0.24,
  "model_version": "v0.1.0",
  "threshold_mode": "conservative",
  "threshold": 0.73,
  "text_stats": {"chars": 342, "tokens": 89}
}
```

Confidence formula: `abs(score_ai - threshold) / max(threshold, 1 - threshold)`.

## Repo layout

```
ai-slop-detector-model/
├── detector/           # Core Python package
│   ├── data/           #   Data loaders (wiki, RAID)
│   ├── eval/           #   Metrics, slices, eval orchestration
│   ├── config.py       #   Pydantic config models
│   ├── train.py        #   Training logic
│   ├── calibrate.py    #   Threshold calibration
│   ├── export.py       #   Model export (HF + ONNX)
│   └── infer.py        #   Inference / scoring
├── scripts/            # CLI entrypoints
├── configs/            # YAML configs (baseline, lora, smoke)
├── tests/              # pytest tests
├── runs/               # Training artifacts (gitignored)
├── plan/               # Execution plans
├── claude/             # Claude Code contributor guide
├── Makefile
├── pyproject.toml
└── README.md
```

## Configs

| Config | Purpose |
|---|---|
| `configs/baseline.yaml` | Full fine-tune, lr=2e-5, 5 epochs, max_length=512 |
| `configs/lora.yaml` | LoRA (r=16, alpha=16), lr=5e-5, 5 epochs |
| `configs/smoke.yaml` | CI/testing: 200 samples, 1 epoch, max_length=128 |

## Phase plan

See [plan/phase_1.md](plan/phase_1.md) for the execution plan.
See [claude/claude.md](claude/claude.md) for the contributor guide.

## References

- [RAID paper](https://arxiv.org/abs/2405.07940)
- [RAID repo](https://github.com/liamdugan/raid)
- [Reference implementation](https://github.com/gouwsxander/slop-detector)
- [HF model](https://huggingface.co/gouwsxander/slop-detector-bert)
- [HF dataset](https://huggingface.co/datasets/gouwsxander/wikipedia-human-ai)
