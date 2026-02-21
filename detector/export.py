"""Export trained model to HF checkpoint format + optional ONNX."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from detector.config import Config


def merge_lora_weights(model) -> PreTrainedModel:
    """Merge LoRA adapter weights into the base model."""
    from peft import PeftModel

    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    return model


def export_hf_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
) -> None:
    """Save model + tokenizer in HF format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def export_onnx(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_path: Path,
    opset_version: int = 14,
    max_length: int = 512,
) -> None:
    """Export model to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    dummy_input = tokenizer(
        "This is a test sentence for ONNX export.",
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )


def validate_onnx_equivalence(
    pytorch_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    onnx_path: Path,
    test_texts: list[str],
    atol: float = 1e-4,
    max_length: int = 512,
) -> bool:
    """Validate ONNX model output matches PyTorch output."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device

    for text in test_texts:
        inputs = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        # PyTorch
        with torch.no_grad():
            pt_logits = pytorch_model(**inputs).logits.cpu().numpy()

        # ONNX
        ort_inputs = {
            "input_ids": inputs["input_ids"].cpu().numpy(),
            "attention_mask": inputs["attention_mask"].cpu().numpy(),
        }
        ort_logits = session.run(None, ort_inputs)[0]

        if not np.allclose(pt_logits, ort_logits, atol=atol):
            raise AssertionError(
                f"ONNX mismatch for text '{text[:50]}...': "
                f"PyTorch={pt_logits}, ONNX={ort_logits}, diff={np.abs(pt_logits - ort_logits).max()}"
            )

    return True


def generate_model_card(
    config: Config,
    metrics: dict | None,
    thresholds: dict | None,
    output_path: Path,
) -> None:
    """Generate MODEL_CARD.md."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    auc_str = "N/A"
    ece_str = "N/A"
    if metrics:
        wiki = metrics.get("wiki", {})
        auc_str = str(wiki.get("auc", "N/A"))
        ece_str = str(wiki.get("ece", "N/A"))

    conservative_str = "N/A"
    balanced_str = "N/A"
    if thresholds:
        c = thresholds.get("conservative", {})
        conservative_str = f"threshold={c.get('threshold', 'N/A')}, FPR={c.get('measured_fpr_human', 'N/A')}"
        b = thresholds.get("balanced", {})
        balanced_str = f"threshold={b.get('threshold', 'N/A')}, FPR={b.get('measured_fpr', 'N/A')}"

    card = dedent(f"""\
    # AI Slop Detector — Model Card

    ## Model Details
    - **Base model**: {config.model.name}
    - **Task**: Binary text classification (AI-generated vs human-written)
    - **LoRA**: {'Yes' if config.model.use_lora else 'No'}
    - **Version**: {config.export.model_version}

    ## Training Data
    - **Primary**: gouwsxander/wikipedia-human-ai (Wikipedia paragraphs + AI rewrites)
    - **Labels**: 0 = human, 1 = AI-generated

    ## Evaluation Results
    - **Wiki test AUC**: {auc_str}
    - **Wiki test ECE**: {ece_str}
    - **Conservative mode**: {conservative_str}
    - **Balanced mode**: {balanced_str}

    ## Intended Use
    - Score text as AI-generated vs human-written
    - Primary deployment: LinkedIn post classification via browser extension
    - Outputs calibrated probabilities, not certainty claims

    ## Limitations
    - Trained primarily on Wikipedia-style text; performance may degrade on other domains
    - Single language (English)
    - Does not detect specific AI model or authorship
    - Scores should be interpreted as probabilities, not verdicts
    {"- ECE > 0.05 indicates probability scale may not be well-calibrated" if ece_str != "N/A" and metrics and metrics.get("wiki", {}).get("ece", 0) > 0.05 else ""}

    ## Ethical Considerations
    - False positives (human text flagged as AI) are the most damaging failure mode
    - Conservative mode targets FPR <= 1% on human text
    - No raw text is stored in model artifacts or logs
    """)

    output_path.write_text(card)


def export(config: Config, run_dir: Path) -> Path:
    """Full export pipeline."""
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoint"

    from detector.eval.run_eval import load_checkpoint

    model, tokenizer = load_checkpoint(checkpoint_dir)

    # Merge LoRA if applicable
    model = merge_lora_weights(model)

    # Export HF checkpoint
    hf_dir = run_dir / "export" / "model_hf"
    export_hf_checkpoint(model, tokenizer, hf_dir)
    print(f"HF checkpoint saved to: {hf_dir}")

    # Optional ONNX export
    if config.export.onnx:
        try:
            onnx_path = run_dir / "export" / "model_onnx" / "model.onnx"
            export_onnx(model, tokenizer, onnx_path, config.export.onnx_opset, config.data.max_length)
            print(f"ONNX model saved to: {onnx_path}")

            # Validate
            test_texts = ["This is a test.", "Another example text for validation."]
            validate_onnx_equivalence(model, tokenizer, onnx_path, test_texts, max_length=config.data.max_length)
            print("ONNX equivalence check passed.")
        except ImportError:
            print("ONNX export skipped: onnx/onnxruntime not installed. Install with: pip install ai-slop-detector[onnx]")
        except Exception as e:
            print(f"ONNX export failed: {e}")

    # Generate model card
    metrics = None
    thresholds = None
    metrics_path = run_dir / "metrics.json"
    thresholds_path = run_dir / "thresholds.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            thresholds = json.load(f)

    generate_model_card(config, metrics, thresholds, run_dir / "export" / "MODEL_CARD.md")
    print(f"Model card saved to: {run_dir / 'export' / 'MODEL_CARD.md'}")

    return run_dir / "export"
