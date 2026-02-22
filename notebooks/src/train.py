
# On cpu
# pip install mlx-lm
# This script uses Low-Rank Adaptation (LoRA), which allows you to "teach" the model your PNB glossary without needing a massive GPU.

import os
from mlx_lm import lora

# 1. Define your paths
model_path = "./models/smollm2-1.7b-local"
data_path = "./data"  # This folder should contain train.jsonl
output_path = "./pnb_adapters"

# 2. Set Training Arguments
# On an M2, iters 500-1000 is usually enough for a glossary
training_args = [
    "--model", model_path,
    "--train",
    "--data", data_path,
    "--iters", "600",
    "--batch-size", "1",
    "--steps-per-report", "10",
    "--steps-per-eval", "50",
    "--resume-adapter-file", output_path,
    "--learning-rate", "1e-5"
]

if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists(data_path):
        print(f"Error: Put your train.jsonl in {data_path}")
    else:
        # Run the MLX training
        lora.run(training_args)

##################################################################################################

# On gpu