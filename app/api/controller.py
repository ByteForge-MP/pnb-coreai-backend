from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from app.models.chat_request import ChatRequest
from app.service.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.api_route("/stream", methods=["POST", "OPTIONS"])
async def stream_chat(request: Request):
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return {}

    # Parse POST body and validate with Pydantic
    body = await request.json()
    chat_request = ChatRequest(**body)

    # Call your streaming service
    return StreamingResponse(
        chat_service.get_streaming_response(request.app, chat_request.prompt, chat_request.model, chat_request.time),
        media_type="text/event-stream"
    )
