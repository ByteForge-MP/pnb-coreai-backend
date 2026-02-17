from huggingface_hub import login

# 1. Paste your token between the quotes below
# Get it from: https://huggingface.co/settings/tokens
my_token = "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"

print("🔐 Attempting to login to Hugging Face...")
login(token=my_token)

print("✅ Success! Your machine is now authenticated.")