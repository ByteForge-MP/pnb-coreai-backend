from fastapi import FastAPI
from app.api.controller import router as chat_router
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# raw_path = os.path.join(BASE_DIR, "..", "models", "smollm2-1.7b-local")
raw_path = os.path.join(BASE_DIR, "models", "smollm2-1.7b-local")
model_path = os.path.normpath(raw_path)

if not os.path.exists(model_path):
    print(f"Path error! Could not find model at: {model_path}")

print(f"Loading from: {model_path}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Runs once on server start ---
    print("Initializing PNB Core AI...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    try:
        app.state.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        app.state.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Crucial for M2 performance
            low_cpu_mem_usage=True,
            local_files_only=True
        ).to(device)
        app.state.device = device
        print("Model loaded successfully into GPU/Memory.")
    except Exception as e:
        print(f"Failed to load model: {e}")

    yield
    
    print("Shutting down... releasing resources.")
    if hasattr(app.state, "model"):
        del app.state.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="PNB_COREAI_BACKEND",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "AI Backend is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)