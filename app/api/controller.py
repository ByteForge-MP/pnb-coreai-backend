import logging
import uuid

from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from app.service.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()
logger = logging.getLogger("chat_logger")

@router.post("/stream")
async def stream_chat(
    request: Request,
    prompt: str = Form(...),
    model: str = Form(...),
    time: str = Form(...),
    file: UploadFile = File(None)
):
    request_id = str(uuid.uuid4())[:8]

    logger.info(
        "[%s] Incoming /stream request | model=%s | prompt_chars=%s | file=%s",
        request_id,
        model,
        len(prompt),
        file.filename if file else None,
    )

    return StreamingResponse(
        chat_service.get_streaming_response(
            app=request.app,
            prompt=prompt,
            model=model,
            time=time,
            file=file,
            request_id=request_id,
        ),
        media_type="text/event-stream"
    )
