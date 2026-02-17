from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "models/my_model" 

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16
)

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)