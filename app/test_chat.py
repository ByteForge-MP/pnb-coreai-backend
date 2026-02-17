import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "smollm2-1.7b-local")

if not os.path.exists(model_path):
    print(f"⚠️ Path error! Could not find model at: {model_path}")

print(f"Loading from: {model_path}")

# 1. Use "mps" for Mac M2 GPU instead of "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Loading model onto {device.upper()}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16, # Uses half-memory, much faster on Mac
    low_cpu_mem_usage=True
).to(device)

# 2. Add a Streamer so you see the AI "typing" live
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def chat_local(user_query):
    messages = [{"role": "user", "content": user_query}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    print("AI is thinking...", end="", flush=True)
    
    # generate() now uses the streamer
    model.generate(
        **inputs, 
        max_new_tokens=200, 
        temperature=0.7,
        do_sample=True,
        streamer=streamer, # 👈 This shows words one-by-one
        pad_token_id=tokenizer.eos_token_id
    )

# --- TEST ---
print("\n" + "-"*30)
query = "How to become GM to CGM?"
chat_local(query)