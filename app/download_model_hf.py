from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# This will download EVERYTHING and cache locally
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

print("Download complete")


#######
# mkdir -p ./models/smollm2-instruct
# cp -r ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/* ./models/smollm2-instruct/
