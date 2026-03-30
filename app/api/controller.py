import logging
import uuid

from fastapi import APIRouter, Request, Form, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from app.auth import LoginRequest, get_current_user, login_user
from app.service.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()
logger = logging.getLogger("chat_logger")


@router.post("/auth/login")
async def auth_login(payload: LoginRequest):
    return login_user(payload)


@router.post("/stream")
async def stream_chat(
    request: Request,
    prompt: str = Form(...),
    model: str = Form(...),
    time: str = Form(...),
    file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user),
):
    request_id = str(uuid.uuid4())[:8]
    chat_service.validate_upload(file)

    logger.info(
        "[%s] Incoming /stream request | user=%s | model=%s | prompt_chars=%s | file=%s",
        request_id,
        current_user.get("username"),
        model,
        len(prompt),
        file.filename if file else None,
    )

    return StreamingResponse(
        chat_service.get_streaming_response(
            app=request.app,
            request=request,
            prompt=prompt,
            model=model,
            time=time,
            file=file,
            request_id=request_id,
        ),
        media_type="text/event-stream"
    )
