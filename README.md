# AI Slop Detector — Model Repo

## What this is

This repository trains a binary text classifier that scores whether a given passage is AI-written or human-written. The model outputs a calibrated probability and a discrete label at a chosen operating threshold. The primary use-case is LinkedIn post classification, but evaluation spans multiple domains via the RAID benchmark. The trained artifact (PyTorch + ONNX) is consumed by a separate browser-extension repo that runs inference on LinkedIn pages.

## Scope

**In scope (Phase 1)**

- Text-only detection (no images, metadata, or user history).
- LinkedIn as the primary deployment surface.
- Multi-domain evaluation using RAID (news, Wikipedia, abstracts, reviews, etc.).
- Exported inference artifact (ONNX) with a defined JSON contract.

**Out of scope (Phase 1)**

- Authorship certainty claims — the model produces probabilities, not verdicts.
- Identity detection or attribution to a specific person or LLM.
- Watermark detection or embedding.

## Architecture

- **Data** — Ingest and split RAID dataset; optionally augment with `wikipedia-human-ai`. Produce stratified train/val/test sets.
- **Train** — Fine-tune `bert-base-cased` with a binary classification head. Standard full fine-tune first; LoRA (via PEFT) as an optimization lever to reduce compute and memory.
- **Eval** — Compute ROC, AUC, threshold sweeps, and per-slice metrics across RAID axes (domain, generator family, decoding strategy, adversarial attack).
- **Export** — Save best checkpoint as PyTorch `.bin` and convert to ONNX with fixed input shapes.
- **Inference** — Deterministic scoring function: text in, JSON out. Handles tokenization, normalization, and threshold application.

RAID is used as the **primary evaluation benchmark**, not necessarily as the only training source. Training data may combine RAID with other curated sets.

LoRA is an **optimization lever** — the baseline is a full fine-tune; LoRA is introduced in Phase 1C to compare parameter-efficient tuning against the baseline in both accuracy and resource usage.

## Data

### RAID

The RAID dataset ([paper](https://arxiv.org/abs/2405.07940), [repo](https://github.com/liamdugan/raid)) covers:

- **Domains**: news, Wikipedia, books, abstracts, reviews, Reddit, recipes, poetry, and more.
- **Generator models**: GPT-4, GPT-3.5, LLaMA, Mistral, Cohere, and others.
- **Decoding strategies**: greedy, sampling, top-k, nucleus.
- **Adversarial attacks**: paraphrasing, homoglyph substitution, whitespace perturbation, and others.

### Local dataset option

[`gouwsxander/wikipedia-human-ai`](https://huggingface.co/datasets/gouwsxander/wikipedia-human-ai) — Wikipedia passages paired with AI rewrites. Useful as a warm-start fine-tuning set or supplementary data.

### Label schema

| Field | Type | Required | Notes |
|---|---|---|---|
| `label` | `human` \| `ai` | yes | Binary target |
| `domain` | string | eval only | For stratified eval slicing |
| `generator_family` | string | eval only | Model or model family that generated the text |
| `attack_type` | string | eval only | Adversarial perturbation type, if any |

## Metrics and evaluation

### Definitions

- **TPR (True Positive Rate)** — fraction of AI texts correctly identified as AI. Also called recall or sensitivity.
- **FPR (False Positive Rate)** — fraction of human texts incorrectly flagged as AI. The quantity we most want to minimize.

### ROC and threshold sweeps

Produce a full ROC curve over the `[0, 1]` score range. Report AUC. Sweep thresholds at 0.01 increments and log (threshold, TPR, FPR) triples.

### Required evaluation slices

| Slice | Grouping key | Purpose |
|---|---|---|
| Overall | — | Aggregate performance |
| By domain | `domain` | Detect domain-specific weaknesses |
| By generator model/family | `generator_family` | Robustness across LLM sources |
| By decoding strategy | `decoding_strategy` | Sensitivity to generation method |
| By adversarial attack | `attack_type` | Resilience to evasion attempts |

### Operating points

| Mode | Definition | Use-case |
|---|---|---|
| **Conservative** | Choose threshold where FPR ≤ 1% on the human slice | Production default — minimize false accusations |
| **Balanced** | Maximize AUC; report FPR at TPR = 90% | Development analysis and model comparison |

## Repo layout

```
ai-slop-detector-model/
├── data/              # raw and processed datasets (gitignored)
├── training/          # training scripts, configs, hyperparameter sweeps
├── eval/              # evaluation scripts, slice analysis, plots
├── export/            # ONNX conversion, model packaging
├── inference/         # inference contract, scoring function, normalization
├── plan/              # execution plans (exists now)
├── claude/            # Claude Code contributor guide (exists now)
└── README.md
```

Only `plan/`, `claude/`, and `README.md` exist right now. Other directories will be created as code is added.

## Phase plan

See [`plan/phase_1.md`](plan/phase_1.md) for the Phase 1 execution plan.
