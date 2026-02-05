import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Official OpenAI Client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Gemini Client (OpenAI-Compatible Mode)
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)