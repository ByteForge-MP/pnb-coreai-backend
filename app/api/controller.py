from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from app.service.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.post("/stream")
async def stream_chat(
    request: Request,
    prompt: str = Form(...),
    model: str = Form(...),
    time: str = Form(...),
    file: UploadFile = File(None)
):
    return StreamingResponse(
        chat_service.get_streaming_response(
            app=request.app,
            prompt=prompt,
            model=model,
            time=time,
            file=file
        ),
        media_type="text/event-stream"
    )