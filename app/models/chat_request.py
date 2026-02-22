from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str
    model: str
    time: str
    