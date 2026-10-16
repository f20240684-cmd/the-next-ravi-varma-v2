#!/usr/bin/env bash
# One-shot GPU run for Colab / any CUDA box, from the project root:
#   bash scripts/colab_train.sh            # smoke test + full training + samples + evaluation
#   SKIP_SMOKE=1 bash scripts/colab_train.sh
# Mirrors notebooks/03_lora_training.ipynb; expects the prepared dataset
# (data/processed, data/reference) to be present already.
set -euo pipefail

echo "=== 1. GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "=== 2. Dependencies (keeps the preinstalled torch) ==="
pip install -q -r requirements-colab.txt
pip install -q -e . --no-deps

echo "=== 3. torchao compatibility ==="
probe='import diffusers, peft, accelerate; from diffusers import StableDiffusionPipeline; from peft import LoraConfig'
if ! python -c "$probe" 2>/tmp/probe.err; then
  if grep -q torchao /tmp/probe.err; then
    echo "torchao mismatch -> pinning torchao==0.18.0"
    pip install -q --upgrade --no-deps "torchao==0.18.0"
    python -c "$probe" 2>/tmp/probe.err || { echo "still failing -> removing torchao (unused)"; pip uninstall -y -q torchao; }
  fi
  python -c "$probe"
fi
python -c "import torch, diffusers, peft, transformers, accelerate; print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| peft', peft.__version__, '| transformers', transformers.__version__, '| accelerate', accelerate.__version__)"

echo "=== 4. Dataset check ==="
python scripts/validate_dataset.py --manifest

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "=== 5. Smoke test ==="
  python scripts/train_lora.py --config configs/lora_sd15.yaml --smoke-test
fi

# TAG=v3 writes to separate folders, so unzipping the results locally never
# overwrites an earlier LoRA (checkpoints/lora/ravi_varma) and its images.
TAG="${TAG:-}"
SUFFIX="${TAG:+_$TAG}"
CK="checkpoints/lora/ravi_varma$SUFFIX"
GEN="outputs/generated/lora$SUFFIX"

echo "=== 6. Full training -> $CK ==="
python scripts/train_lora.py --config configs/lora_sd15.yaml --output-dir "$CK"

echo "=== 7. Sample images with the trained LoRA ==="
python scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds 42 43 44 \
  --skip-existing --lora-path "$CK" --output-dir "$GEN"
for step in 750 1000 1250; do
  python scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds 42 \
    --skip-existing --lora-path "$CK/checkpoint-$step" --output-dir "outputs/generated/ckpt${SUFFIX}_$step"
done
echo "=== 7b. Same prompts with hires fix (1.5x) + face fix (needs the T4's 16 GB) ==="
python scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds 42 \
  --skip-existing --lora-path "$CK" --hires-scale 1.5 --face-fix --output-dir "${GEN}_hires_face"

echo "=== 8. Evaluation (paired against the base-model baseline) ==="
BASELINE_ARGS=()
if ls outputs/generated/baseline/*.png >/dev/null 2>&1; then BASELINE_ARGS=(--baseline outputs/generated/baseline); fi
python scripts/evaluate.py --input "$GEN" "${BASELINE_ARGS[@]}" \
  --output-dir "outputs/evaluations/lora${SUFFIX}_vs_base" --device cuda

echo "=== 9. Package results ==="
rm -f ravi_varma_results.zip
zip -q -r ravi_varma_results.zip \
  "$CK"/*.safetensors "$CK"/*.yaml "$CK"/*.json "$CK"/*.csv \
  "$CK"/checkpoint-*/pytorch_lora_weights.safetensors \
  outputs/logs/validation_images outputs/logs/train_loss.csv \
  "$GEN" "${GEN}_hires_face" outputs/generated/ckpt"${SUFFIX}"_* "outputs/evaluations/lora${SUFFIX}_vs_base"
ls -la ravi_varma_results.zip
echo "DONE"
