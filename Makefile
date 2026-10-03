.PHONY: install install-colab download validate captions prepare check-manifest package-data smoke-test train \
        generate samples baseline evaluate app test lint

PY ?= python
SEEDS ?= 42 43 44

# ---- setup ---------------------------------------------------------------
install:
	pip install -r requirements.txt
	pip install -e .

install-colab:
	pip install -r requirements-colab.txt
	pip install -e . --no-deps

# ---- Milestone 1: dataset (local machine) ----------------------------------
download:            ## re-fetch the exact raw images listed in metadata.jsonl
	$(PY) scripts/download_dataset.py --from-metadata

validate:
	$(PY) scripts/validate_dataset.py

captions:            ## BLIP + BLIP-VQA template captions for accepted images
	$(PY) scripts/generate_captions.py

prepare:
	$(PY) scripts/prepare_dataset.py

check-manifest:
	$(PY) scripts/validate_dataset.py --manifest

package-data:        ## dist/ravi_varma_dataset.zip for Colab
	$(PY) scripts/package_dataset.py

# ---- Milestone 2: training (CUDA GPU / Colab) -------------------------------
smoke-test:
	$(PY) scripts/train_lora.py --config configs/lora_sd15.yaml --smoke-test

train:
	$(PY) scripts/train_lora.py --config configs/lora_sd15.yaml

# ---- generation + evaluation ---------------------------------------------------
generate:            ## make generate PROMPT="<rvvarma>, ..."
	$(PY) scripts/generate.py --prompt "$(PROMPT)" --seed 42

samples:
	$(PY) scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds $(SEEDS) --output-dir outputs/generated/lora

baseline:
	$(PY) scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds $(SEEDS) --no-lora --output-dir outputs/generated/baseline

evaluate:
	$(PY) scripts/evaluate.py --input outputs/generated/lora --baseline outputs/generated/baseline --output-dir outputs/evaluations/lora_vs_base

# ---- Milestone 4 preview / dev -------------------------------------------------
app:
	$(PY) app/app.py

test:
	pytest -v

lint:
	$(PY) -m pyflakes src scripts app tests || true
