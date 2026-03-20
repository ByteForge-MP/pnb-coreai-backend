from transformers import AutoTokenizer, AutoModelForCausalLM

from app.device import get_best_device, get_model_dtype

MODEL_PATH = "models/my_model" 
DEVICE = get_best_device()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=get_model_dtype(DEVICE)
)

model.eval()
model.to(DEVICE)
