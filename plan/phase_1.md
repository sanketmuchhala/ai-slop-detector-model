# Phase 1: Baseline Classifier, RAID Benchmarking, and Inference Contract

## Goals

- Train a binary AI-vs-human text classifier using `bert-base-cased`.
- Evaluate rigorously on RAID across all available slicing axes.
- Establish two operating thresholds (Conservative, Balanced).
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
| Trained model checkpoint | `training/checkpoints/best/` | PyTorch `.bin` + `config.json` |
| LoRA adapter (if applicable) | `training/checkpoints/lora/` | PEFT adapter files |
| Evaluation report | `eval/results/metrics.json` | JSON |
| ROC curve plot | `eval/results/roc.png` | PNG |
| Per-slice breakdown | `eval/results/slices.json` | JSON |
| Threshold config | `eval/results/thresholds.json` | JSON (conservative + balanced) |
| ONNX model | `export/model.onnx` | ONNX |
| Inference module | `inference/predict.py` | Python |
| Model card | `export/model_card.md` | Markdown |

---

## Phase 1A: Baseline Pipeline

**Size: Medium**

### Steps

1. **Environment setup**
   - Create `requirements.txt` with pinned versions: `transformers`, `datasets`, `evaluate`, `peft`, `torch`, `scikit-learn`, `onnx`, `onnxruntime`.
   - Add a `Makefile` with targets: `setup`, `train`, `eval`, `export`.
   - Verify GPU availability and CUDA version.

2. **Data ingestion — RAID**
   - Clone or install RAID tools from https://github.com/liamdugan/raid.
   - Download the RAID dataset via the provided scripts or API.
   - Parse into a unified format: `{"text": str, "label": int, "domain": str, "generator_family": str, "decoding_strategy": str, "attack_type": str | null}`.
   - Write to `data/raid/` as Parquet or Arrow files.

3. **Data ingestion — wikipedia-human-ai (optional warm start)**
   - Load via `datasets.load_dataset("gouwsxander/wikipedia-human-ai")`.
   - Map to the same schema (label, text; domain = "wikipedia", generator_family from metadata).
   - Write to `data/wiki-hai/`.

4. **Train/val/test split**
   - Split RAID with domain-stratified sampling: 80/10/10.
   - Ensure no text overlap across splits.
   - Save split indices to `data/splits.json` for reproducibility.

5. **Baseline model**
   - Load `bert-base-cased` from Hugging Face.
   - Add a linear classification head (`BertForSequenceClassification`, `num_labels=2`).
   - Tokenizer: `BertTokenizerFast`, `max_length=512`, padding to longest in batch, truncation enabled.

6. **Training loop**
   - Use `Trainer` from Transformers or a manual PyTorch loop.
   - Initial hyperparameters:
     - Learning rate: `2e-5`
     - Batch size: `16` (gradient accumulation to simulate `32` if needed)
     - Max epochs: `5`
     - Weight decay: `0.01`
     - Warmup: 10% of total steps
     - Seed: `42`
   - Log to Weights & Biases or TensorBoard (optional).
   - Save best checkpoint by validation AUC.

7. **Quick validation**
   - After training, compute val AUC and val loss.
   - Sanity-check: AUC > 0.85 on val set or investigate data/model issues.

### Outputs

- `training/checkpoints/best/` — model weights, config, tokenizer.
- `data/raid/` and `data/wiki-hai/` — processed data.
- `data/splits.json` — split indices.
- `training/train_log.json` — loss curves and val metrics per epoch.

### Acceptance criteria

- Training completes without error.
- Val AUC ≥ 0.85.
- All data files are reproducible from a clean clone + `make setup`.

---

## Phase 1B: RAID Benchmarking and Analysis

**Size: Medium**

### Steps

1. **Full evaluation script**
   - Load best checkpoint from 1A.
   - Run inference on the RAID test split.
   - Compute for each sample: predicted probability of AI (`score_ai`).

2. **Aggregate metrics**
   - AUC (overall).
   - ROC curve: sweep thresholds from 0.0 to 1.0 in 0.01 steps.
   - At each threshold, record (threshold, TPR, FPR, precision, F1).
   - Save to `eval/results/metrics.json`.

3. **Slice analysis**
   - Group test predictions by: domain, generator_family, decoding_strategy, attack_type.
   - For each group, compute: AUC, TPR@FPR=1%, FPR@TPR=90%.
   - Save to `eval/results/slices.json`.

4. **ROC plot**
   - Plot ROC curve with AUC in legend.
   - Mark the Conservative operating point (FPR = 1%) and Balanced point (TPR = 90%).
   - Save to `eval/results/roc.png`.

5. **Error analysis**
   - Identify the 50 highest-confidence false positives (human texts scored as AI).
   - Identify the 50 highest-confidence false negatives (AI texts scored as human).
   - Save to `eval/results/error_samples.json` — store only sample IDs and scores, **not raw text**.

### Outputs

- `eval/results/metrics.json`
- `eval/results/slices.json`
- `eval/results/roc.png`
- `eval/results/error_samples.json`

### Acceptance criteria

- All RAID test samples scored.
- Slice analysis covers every axis present in the data.
- ROC plot renders correctly with both operating points marked.
- No raw text stored in any eval artifact.

---

## Phase 1C: LoRA and Calibration

**Size: Medium**

### Steps

1. **LoRA fine-tune**
   - Apply LoRA via `peft` to the attention layers of `bert-base-cased`.
   - LoRA config: `r=8`, `lora_alpha=16`, `lora_dropout=0.1`, target modules: `query`, `value`.
   - Train with same hyperparameters as 1A baseline.
   - Save LoRA adapter to `training/checkpoints/lora/`.

2. **Compare baseline vs LoRA**
   - Run the same eval suite from 1B on the LoRA model.
   - Record: AUC delta, parameter count, training time, peak GPU memory.
   - Decide which model to carry forward based on AUC within 1% of baseline and resource savings.

3. **Hyperparameter sweep**
   - Sweep over:
     - Learning rate: `[1e-5, 2e-5, 5e-5]`
     - Batch size: `[16, 32]`
     - Max length: `[256, 512]`
     - Weight decay: `[0.0, 0.01, 0.1]`
     - Epochs: `[3, 5]`
   - Use val AUC as the selection criterion.
   - Log all runs to a sweep results file.

4. **Threshold calibration**
   - On the **validation set**, find the threshold where FPR ≤ 1% on human samples → Conservative threshold.
   - Find the threshold where TPR = 90% → Balanced threshold. Record corresponding FPR.
   - Record calibration dataset (val split), method (direct threshold search), and date.
   - Save to `eval/results/thresholds.json`:
     ```json
     {
       "conservative": {"threshold": 0.XX, "fpr": 0.XX, "tpr": 0.XX, "calibration_set": "val"},
       "balanced": {"threshold": 0.XX, "fpr": 0.XX, "tpr": 0.XX, "calibration_set": "val"}
     }
     ```

5. **Optional: temperature scaling**
   - If raw probabilities are poorly calibrated (ECE > 0.05), fit temperature scaling on the val set.
   - Record temperature parameter in `thresholds.json`.

### Outputs

- `training/checkpoints/lora/` — LoRA adapter.
- `eval/results/lora_comparison.json` — baseline vs LoRA metrics.
- `eval/results/sweep_results.json` — hyperparameter sweep log.
- `eval/results/thresholds.json` — calibrated thresholds.

### Acceptance criteria

- LoRA model trained and evaluated on the same test set as baseline.
- Comparison table produced with AUC, param count, training time, memory.
- Conservative threshold achieves FPR ≤ 1% on human val samples.
- Balanced threshold achieves TPR ≥ 90% on AI val samples.

---

## Phase 1D: Export and Inference Contract

**Size: Small**

### Steps

1. **ONNX export**
   - Export the selected model (baseline or LoRA-merged) to ONNX.
   - Input: `input_ids` (int64, shape `[1, 512]`), `attention_mask` (int64, shape `[1, 512]`).
   - Output: `logits` (float32, shape `[1, 2]`).
   - Validate ONNX output matches PyTorch output within `atol=1e-5` on 100 test samples.
   - Save to `export/model.onnx`.

2. **Inference module**
   - Create `inference/predict.py` with a `score(text: str) -> dict` function.
   - Text normalization: strip leading/trailing whitespace, collapse multiple newlines to one, normalize unicode to NFC.
   - Tokenize, run model, apply softmax, apply threshold.
   - Deterministic: no dropout at inference, same input always produces same output.

3. **Output JSON schema**
   ```json
   {
     "label": "ai" | "human",
     "score_ai": 0.0-1.0,
     "confidence": "low" | "medium" | "high",
     "model_version": "v0.1.0",
     "threshold_mode": "conservative" | "balanced"
   }
   ```
   - `confidence` derived from distance of `score_ai` to the active threshold:
     - `|score_ai - threshold| < 0.1` → `low`
     - `0.1 ≤ |score_ai - threshold| < 0.3` → `medium`
     - `|score_ai - threshold| ≥ 0.3` → `high`

4. **Model card**
   - Write `export/model_card.md` documenting: base model, training data, eval results, known limitations, intended use.

### Outputs

- `export/model.onnx`
- `inference/predict.py`
- `export/model_card.md`

### Acceptance criteria

- ONNX model passes numerical equivalence check against PyTorch.
- `inference/predict.py` produces valid JSON for any non-empty string input.
- Empty string input returns a defined error response, not a crash.
- Determinism verified: same input → same output across 3 runs.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| RAID data format changes or download issues | Medium | High | Pin RAID repo commit hash; cache downloaded data in `data/` |
| Baseline AUC below 0.85 | Low | Medium | Check data pipeline for leaks; try `bert-large-cased`; increase training data |
| LoRA degrades quality significantly | Low | Medium | Fall back to full fine-tune; LoRA is optional |
| ONNX conversion fails for custom architecture | Low | Low | Using standard `BertForSequenceClassification` — well-tested export path |
| Overfitting to RAID distribution | Medium | High | Validate on held-out wikipedia-human-ai; monitor per-domain variance |
| Threshold calibration unstable with small val set | Medium | Medium | Use at least 2000 human samples for calibration; bootstrap confidence intervals |

## Definition of Done

Phase 1 is complete when all of the following are true:

- [ ] `make train` runs end-to-end and produces a saved checkpoint.
- [ ] `make eval` produces `metrics.json`, `slices.json`, `roc.png`, and `thresholds.json`.
- [ ] `make export` produces `model.onnx` that passes equivalence checks.
- [ ] Conservative threshold achieves FPR ≤ 1% on human val set.
- [ ] Slice analysis covers all RAID axes present in data.
- [ ] `inference/predict.py` returns valid JSON and is deterministic.
- [ ] No raw text is stored in any eval artifact or log.
- [ ] Model card exists at `export/model_card.md`.
