# `checkpoints/lora/`

Trained LoRA weights are written here by `scripts/train_lora.py` (on a CUDA GPU / Colab):

```
checkpoints/lora/ravi_varma/
├── pytorch_lora_weights.safetensors   # final adapter (~13 MB), loadable with pipe.load_lora_weights()
├── training_config.yaml               # exact config used
├── training_summary.json              # steps, dataset size, device, final loss, checkpoint list
├── loss_history.csv                   # mean loss + LR per optimizer step
└── checkpoint-<step>/                 # every 250 steps: resumable accelerator state
    └── pytorch_lora_weights.safetensors   # + a loadable adapter for checkpoint selection
```

Weights are **not** committed to git (see `.gitignore`); keep them in Google Drive or the Hugging Face Hub.
After training on Colab, download `ravi_varma_results.zip` from the notebook and run
`unzip -o ravi_varma_results.zip` in the project root.

`scripts/generate.py --lora-path` accepts the run directory, a `checkpoint-<step>` directory or a
`.safetensors` file. If no weights are found, generation still runs but with the **base** model only, and
every output's metadata says `"lora_loaded": false` — it never pretends a LoRA is loaded.
