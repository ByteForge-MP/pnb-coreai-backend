from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o"  # Default model