# Google Colab Training Guide

Run each section below as its own Colab cell. Use a **T4 GPU runtime** (Runtime → Change runtime type → T4 GPU).

---

## Cell 1 — Mount Drive and get the repo

```python
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess

REPO = '/content/drive/MyDrive/ai-slop-detector-model'

if not os.path.exists(REPO):
    subprocess.run(['git', 'clone',
                    'https://github.com/sanketmuchhala/ai-slop-detector-model.git',
                    REPO], check=True)
else:
    # Pull latest changes (includes all the fixes)
    subprocess.run(['git', '-C', REPO, 'pull', 'origin', 'main'], check=True)

%cd {REPO}
!pwd
```

---

## Cell 2 — Install dependencies

```python
!make setup
```

Expected output ends with something like:
```
PyTorch 2.x.x, CUDA: True
Transformers 4.x.x
Setup complete.
```

If CUDA is `False`, stop and switch to a GPU runtime.

---

## Cell 3 — Confirm GPU

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Cell 4 — Train

Pick **one** of the two options below.

### Option A — RoBERTa (recommended, best accuracy)

Trains `roberta-base` on wiki + RAID train + RAID extra splits.  
Expected time: **~2–3 hours** on T4.

```python
!make train CONFIG=configs/roberta.yaml
```

### Option B — BERT baseline (faster, cheaper)

Trains `bert-base-cased` on wiki + RAID train split.  
Expected time: **~1–1.5 hours** on T4.

```python
!make train CONFIG=configs/baseline.yaml
```

Training prints `eval_auc` at the end of each epoch. You want to see it climbing toward 0.90+. If it's stuck at ~0.50, stop and check the RAID install warning in the output.

---

## Cell 5 — Find your run directory (run this after training)

```python
import glob, os
from pathlib import Path

runs = sorted(glob.glob('runs/*/'), key=os.path.getctime, reverse=True)
if not runs:
    raise RuntimeError("No runs found — did training complete?")

RUN_DIR = runs[0].rstrip('/')
print(f"Latest run: {RUN_DIR}")
```

All subsequent cells use `RUN_DIR` automatically. No more hardcoding.

---

## Cell 6 — Calibrate thresholds

Finds the conservative threshold (≤1% FPR on human text) and the balanced threshold (90% TPR).

```python
!python scripts/calibrate.py --run-dir {RUN_DIR}
```

Output will show something like:
```
Conservative threshold: 0.7821  (FPR=0.0098)
Balanced threshold:     0.3142  (FPR=0.0830)
```

---

## Cell 7 — Export model

Copies the best checkpoint to `{RUN_DIR}/export/model_hf/` and generates a model card.

```python
!python scripts/export.py --run-dir {RUN_DIR}
```

---

## Cell 8 — Test inference

```python
import subprocess, json

def test(text, mode="balanced"):
    result = subprocess.run(
        ['python', 'scripts/infer.py',
         '--model-dir', f'{RUN_DIR}/export/model_hf',
         '--thresholds', f'{RUN_DIR}/thresholds.json',
         '--mode', mode,
         '--text', text],
        capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)

# Obvious AI slop — should be "ai"
ai_text = ("In a monumental leap for mankind, the historic Apollo 11 endeavor "
           "successfully executed humanity's inaugural lunar landing.")
print("AI text →", test(ai_text))

# Plain human writing — should be "human"
human_text = "Neil Armstrong landed on the moon in July 1969."
print("Human text →", test(human_text))
```

A working model will correctly label both. If both come back `"human"`, the training AUC was too low — check Cell 4 output.

---

## Cell 9 — Evaluate on RAID (optional, takes ~20 min)

```python
!python scripts/eval.py --run-dir {RUN_DIR} --dataset raid_labeled
```

Look for `"auc"` in the output. Anything above **0.85** on RAID is solid. Above **0.90** is very good.

---

## Full pipeline in one shot (alternative to Cells 4–7)

If you trust the run will complete without errors:

```python
# RoBERTa
!make pipeline CONFIG=configs/roberta.yaml DATASET=wiki

# Then grab the run dir and calibrate (pipeline doesn't do this automatically)
import glob, os
RUN_DIR = sorted(glob.glob('runs/*/'), key=os.path.getctime, reverse=True)[0].rstrip('/')
!python scripts/calibrate.py --run-dir {RUN_DIR}
!python scripts/export.py --run-dir {RUN_DIR}
print(f"\nDone. Artifacts at: {RUN_DIR}/export/")
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA: False` | Runtime → Change runtime type → T4 GPU |
| `Warning: raid-bench not installed` during training | Run `!pip install git+https://github.com/liamdugan/raid.git` then retry Cell 4 |
| `eval_auc` stuck at 0.50 all epochs | RAID data didn't load (check for the warning above). Model is training on wiki only. |
| Both test texts classified as `"human"` | AUC was low; try lowering `conservative_max_fpr` in the config or re-train with RAID data confirmed loaded |
| `No runs found` in Cell 5 | Training didn't complete — scroll up in Cell 4 output for the error |
| Colab session disconnected mid-training | Re-run Cell 1 (repo is on Drive, so no re-clone needed), skip to Cell 5 if a checkpoint exists |
