PYTHON ?= python
CONFIG ?= configs/baseline.yaml
RUN_DIR ?= runs/latest
DATASET ?= all
MODE ?= conservative
TEXT ?= "Hello world"

.PHONY: setup train eval eval-raid-submit calibrate export infer train-lora \
        test test-fast lint format clean pipeline smoke help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv, install dependencies, verify GPU
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install --no-deps git+https://github.com/liamdugan/raid.git || echo "Warning: raid-bench install failed. RAID eval will not be available."
	$(PYTHON) -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
	$(PYTHON) -c "import transformers; print(f'Transformers {transformers.__version__}')"
	@echo "Setup complete."

train:  ## Train model (CONFIG=configs/baseline.yaml)
	$(PYTHON) scripts/train.py --config $(CONFIG)

eval:  ## Run evaluation (RUN_DIR=runs/latest DATASET=all)
	$(PYTHON) scripts/eval.py --run-dir $(RUN_DIR) --dataset $(DATASET)

eval-raid-submit:  ## Generate predictions.json for RAID leaderboard (RUN_DIR=runs/latest)
	$(PYTHON) scripts/eval.py --run-dir $(RUN_DIR) --dataset raid_test_unlabeled

calibrate:  ## Calibrate thresholds on validation set (RUN_DIR=runs/latest)
	$(PYTHON) scripts/calibrate.py --run-dir $(RUN_DIR)

export:  ## Export model to HF + ONNX (RUN_DIR=runs/latest)
	$(PYTHON) scripts/export.py --run-dir $(RUN_DIR)

infer:  ## Run inference (RUN_DIR=runs/latest MODE=conservative TEXT="...")
	$(PYTHON) scripts/infer.py --model-dir $(RUN_DIR)/export/model_hf --thresholds $(RUN_DIR)/thresholds.json --mode $(MODE) --text "$(TEXT)"

train-lora:  ## Train with LoRA config
	$(PYTHON) scripts/train.py --config configs/lora.yaml

test:  ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:  ## Run tests excluding slow markers
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow"

lint:  ## Run ruff linter
	$(PYTHON) -m ruff check detector/ scripts/ tests/

format:  ## Run black formatter
	$(PYTHON) -m black detector/ scripts/ tests/

clean:  ## Remove generated artifacts (not runs/)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/

pipeline:  ## Run full pipeline: train -> eval -> calibrate -> export
	$(MAKE) train CONFIG=$(CONFIG)
	$(MAKE) eval RUN_DIR=$$(ls -td runs/*/ | head -1) DATASET=$(DATASET)
	$(MAKE) calibrate RUN_DIR=$$(ls -td runs/*/ | head -1)
	$(MAKE) export RUN_DIR=$$(ls -td runs/*/ | head -1)

smoke:  ## Full pipeline with smoke config (fast CI check)
	$(MAKE) pipeline CONFIG=configs/smoke.yaml DATASET=wiki
