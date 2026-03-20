import os
import argparse
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.api.controller import router as chat_router
from app.device import get_best_device, get_model_dtype

parser = argparse.ArgumentParser()

parser.add_argument(
    "--ollama",
    type=str,
    default="false",
    help="Run backend in ollama mode (true/false)"
)

args, _ = parser.parse_known_args()

USE_OLLAMA = args.ollama.lower() == "true"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(BASE_DIR, "models", "smollm2-1.7b-local")
model_path = os.path.normpath(raw_path)
HF_MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
LOCAL_MODEL_PATH = model_path

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Initializing PNB Core AI...")

    device = get_best_device()
    model_dtype = get_model_dtype(device)
    print("Device detected:", device)

    if USE_OLLAMA:
        print("Running in OLLAMA MODE.")
        print("Skipping local model loading.")
        yield
        print("Server shutting down (Ollama mode).")
        return

    try:

        if os.path.exists(LOCAL_MODEL_PATH):

            print(f"Loading model from local path: {LOCAL_MODEL_PATH}")

            tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_MODEL_PATH,
                local_files_only=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
                device_map="auto" if device == "cuda" else None
            )

        # -----------------------------
        # Otherwise download from HF
        # -----------------------------

        else:

            print("Local model not found. Downloading from HuggingFace...")

            tokenizer = AutoTokenizer.from_pretrained(
                HF_MODEL_NAME
            )

            model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if device == "cuda" else None
            )

        if device != "cuda":
            model.to(device)

        # -----------------------------
        # Store in FastAPI state
        # -----------------------------

        app.state.tokenizer = tokenizer
        app.state.model = model
        app.state.device = device

        print("Model loaded successfully into memory.")

    except Exception as e:

        print("Model loading failed:", e)

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

app.state.use_ollama = USE_OLLAMA

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
    return {
        "message": "AI Backend is running",
        "ollama_mode": USE_OLLAMA
    }

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
