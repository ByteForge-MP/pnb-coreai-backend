import os
from huggingface_hub import snapshot_download

# 🔥 ACTIVATE TURBO DOWNLOAD
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Define your local storage folder
LOCAL_MODEL_PATH = "./models/smollm2-1.7b-local"

print("📡 Connecting to Hugging Face to download SmolLM2...")

# This downloads the full repo (weights, tokenizer, config)
snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    local_dir=LOCAL_MODEL_PATH,
    local_dir_use_symlinks=False, # Important: saves real files, not links
    revision="main",
    # 🎯 THIS LINE FORCES THE DOWNLOAD OF THE RIGHT FILE
    allow_patterns=["*.safetensors", "*.json", "*.txt"]
)

print(f"✅ Download complete! All files are in: {LOCAL_MODEL_PATH}")
print("You can now safely disconnect from the internet.")
