import json
import os
import asyncio
import queue
import torch
import fitz
import httpx
import pandas as pd
import logging
import time as time_module

from io import BytesIO
from threading import Event, Thread

from fastapi import HTTPException, status
from app.ai.openai_client import get_gemini_client, get_openai_client
from app.retriever import retrieve, embed_query
from app.search_service import search_searxng
from app.dynamic_rag import build_dynamic_embeddings
from app.query_router import route_query
from app.reranker import rerank
from app.context_compressor import compress_context
from app.memory_store import add_memory, get_memory


logger = logging.getLogger("chat_logger")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

DEFAULT_OLLAMA_URLS = (
    "http://ollama:11434/api/generate",
    "http://host.docker.internal:11434/api/generate",
    "http://localhost:11434/api/generate",
)


PNB_SYSTEM_PROMPT = """
You are an assistant for a bank's internal use.

Rules:
- Use the provided knowledge.
- If information is missing say you don't know.
- Be concise and factual.
- If retrieved knowledge is present, do not say you lack real-time access.
- If the retrieved knowledge is ambiguous or does not explicitly answer the question, say that clearly instead of guessing.

Conversation Style:
- At the end of helpful answers, ask ONE short follow-up question.
- Do not ask generic greetings.
- Do not ask a question if the answer is already complete.
"""


class ChatService:

    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".aac", ".ogg", ".flac"}
    DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".xls"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

    FACT_PREFIXES = (
        "who is",
        "what is",
        "when is",
        "where is",
        "which is",
        "tell me the",
        "name the",
    )

    FACT_KEYWORDS = (
        "md",
        "managing director",
        "ceo",
        "chairman",
        "prime minister",
        "president",
        "founder",
        "owner",
        "head office",
        "headquarter",
    )

    def _build_search_event(self, image=None, sources=None, search_error=None):
        payload = {}

        if image:
            payload["image"] = image

        if sources:
            payload["sources"] = sources

        if search_error:
            payload["search_error"] = search_error

        return payload

    def _get_ollama_urls(self):
        configured_url = os.getenv("OLLAMA_URL", "").strip()
        candidates = []

        if configured_url:
            candidates.append(configured_url)

        candidates.extend(DEFAULT_OLLAMA_URLS)

        normalized = []

        for url in candidates:
            final_url = url.rstrip("/")

            if not final_url.endswith("/api/generate"):
                final_url = f"{final_url}/api/generate"

            if final_url not in normalized:
                normalized.append(final_url)

        return normalized

    def _prompt_preview(self, prompt, limit=120):
        cleaned = " ".join(prompt.split())
        return cleaned[:limit]

    async def _client_disconnected(self, request, request_id):
        if request is None:
            return False

        if await request.is_disconnected():
            logger.info("[%s] Stream cancelled by client disconnect", request_id)
            return True

        return False

    def _get_file_extension(self, file):
        filename = (getattr(file, "filename", "") or "").lower()
        _, extension = os.path.splitext(filename)
        return extension

    def _is_audio_file(self, file):
        content_type = (getattr(file, "content_type", "") or "").lower()
        extension = self._get_file_extension(file)
        return content_type.startswith("audio/") or extension in self.AUDIO_EXTENSIONS

    def _is_supported_document_or_image(self, file):
        content_type = (getattr(file, "content_type", "") or "").lower()
        extension = self._get_file_extension(file)

        if content_type.startswith("image/") or extension in self.IMAGE_EXTENSIONS:
            return True

        if extension in self.DOCUMENT_EXTENSIONS:
            return True

        return content_type in {
            "application/pdf",
            "text/plain",
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        }

    def validate_upload(self, file):
        if file is None:
            return

        if self._is_audio_file(file) or self._is_supported_document_or_image(file):
            return

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type or file.filename or 'unknown'}",
        )

    async def _transcribe_audio_file(self, file):
        audio_bytes = await file.read()
        logger.info(
            "Transcribing audio upload | filename=%s | content_type=%s | bytes=%s",
            file.filename,
            file.content_type,
            len(audio_bytes),
        )

        # Placeholder transcription path.
        # Replace this block with Whisper / faster-whisper / OpenAI audio transcription later.
        if not audio_bytes:
            return ""

        return (
            f"[Audio transcription placeholder] Uploaded audio file '{file.filename}' was received "
            "successfully. Replace _transcribe_audio_file() with Whisper or another ASR library "
            "to turn speech into real text."
        )

    def _is_factoid_query(self, prompt):
        normalized = " ".join(prompt.lower().split())

        if normalized.startswith(self.FACT_PREFIXES):
            return True

        return any(keyword in normalized for keyword in self.FACT_KEYWORDS)

    def _build_web_chunks(self, prompt, search_docs, request_id):
        documents = search_docs.get("documents", []) if search_docs else []
        summary_text = search_docs.get("summary", "") if search_docs else ""

        if not documents and not summary_text:
            return []

        if self._is_factoid_query(prompt):
            fact_chunks = []

            for doc in documents[:8]:
                text = str(doc.get("text", "")).strip()

                if text:
                    fact_chunks.append(text)

            logger.info(
                "[%s] Using direct factoid web chunks | documents=%s | chunks=%s",
                request_id,
                len(documents),
                len(fact_chunks),
            )
            return fact_chunks

        candidate_docs = []

        for doc in documents:
            text = str(doc.get("text", "")).strip()

            if text:
                candidate_docs.append(text)

        if summary_text:
            candidate_docs.append(summary_text)

        if not candidate_docs:
            return []

        try:
            dynamic_index, dynamic_chunks = build_dynamic_embeddings(candidate_docs)
        except Exception as exc:
            logger.warning("[%s] Dynamic web embeddings unavailable, falling back to raw docs: %s", request_id, exc)
            return candidate_docs[:5]

        query_embedding = embed_query(prompt)

        if dynamic_index is None or not dynamic_chunks or query_embedding is None:
            logger.warning("[%s] Web reranking unavailable, falling back to raw docs", request_id)
            return candidate_docs[:5]

        _, indices = dynamic_index.search(query_embedding, min(5, len(dynamic_chunks)))

        selected = []
        seen = set()

        for idx in indices[0]:
            if idx < 0 or idx >= len(dynamic_chunks):
                continue

            chunk = dynamic_chunks[idx]

            if chunk and chunk not in seen:
                seen.add(chunk)
                selected.append(chunk)

        logger.info(
            "[%s] Using reranked web chunks | candidate_docs=%s | selected_chunks=%s",
            request_id,
            len(candidate_docs),
            len(selected),
        )
        return selected


# ---------------------------------------------------------
# MAIN STREAM ROUTER
# ---------------------------------------------------------

    async def get_streaming_response(self, app, request, prompt, model, time, file=None, request_id="unknown"):
        started_at = time_module.monotonic()

        logger.info(
            "[%s] Stream started | model=%s | ollama_mode=%s | prompt_preview=%r",
            request_id,
            model,
            getattr(app.state, "use_ollama", False),
            self._prompt_preview(prompt),
        )

        try:
            if await self._client_disconnected(request, request_id):
                logger.info("[%s] Stream stopped before provider execution", request_id)
                return

            if getattr(app.state, "use_ollama", False):
                logger.info("[%s] Routing request to Ollama stream", request_id)

                async for chunk in self._stream_ollama(request, prompt, time, file, request_id):
                    yield chunk

                    if await self._client_disconnected(request, request_id):
                        logger.info("[%s] Stream stopped during Ollama streaming", request_id)
                        return

                logger.info(
                    "[%s] Stream completed via Ollama in %.2fs",
                    request_id,
                    time_module.monotonic() - started_at,
                )
                yield "data: [DONE]\n\n"
                return

            if model in {"gpt-4o", "gpt-3.5-turbo"}:
                logger.info("[%s] Routing request to OpenAI stream", request_id)

                async for chunk in self._stream_openai(request, prompt, model, request_id):
                    yield chunk

                    if await self._client_disconnected(request, request_id):
                        logger.info("[%s] Stream stopped during OpenAI streaming", request_id)
                        return

            elif model == "gemini-3-flash-preview":
                logger.info("[%s] Routing request to Gemini stream", request_id)

                async for chunk in self._stream_gemini(request, prompt, model, request_id):
                    yield chunk

                    if await self._client_disconnected(request, request_id):
                        logger.info("[%s] Stream stopped during Gemini streaming", request_id)
                        return

            elif model == "pnb-local-model":
                logger.info("[%s] Routing request to local model stream", request_id)

                async for chunk in self._stream_local(app, request, prompt, time, file, request_id):
                    yield chunk

                    if await self._client_disconnected(request, request_id):
                        logger.info("[%s] Stream stopped during local model streaming", request_id)
                        return

            else:
                logger.info("[%s] Unknown model %s, defaulting to local stream", request_id, model)

                async for chunk in self._stream_local(app, request, prompt, time, file, request_id):
                    yield chunk

                    if await self._client_disconnected(request, request_id):
                        logger.info("[%s] Stream stopped during fallback local model streaming", request_id)
                        return

        except asyncio.CancelledError:
            logger.info("[%s] Streaming task cancelled", request_id)
            raise

        except Exception as e:

            logger.exception("[%s] Streaming failed: %s", request_id, str(e))

            yield f"data: {json.dumps({'error': 'Switch Case Failed', 'details': str(e)})}\n\n"

        logger.info(
            "[%s] Stream finished in %.2fs",
            request_id,
            time_module.monotonic() - started_at,
        )
        if not await self._client_disconnected(request, request_id):
            yield "data: [DONE]\n\n"
        else:
            logger.info("[%s] Stream closed without [DONE] because client had already disconnected", request_id)


# ---------------------------------------------------------
# OPENAI STREAM
# ---------------------------------------------------------

    async def _stream_openai(self, request, prompt, model, request_id):

        system_prompt, context, image, sources, search_error = await self._build_context(prompt, None, request_id)
        openai_client = get_openai_client()
        logger.info("[%s] OpenAI request started | model=%s", request_id, model)

        search_event = self._build_search_event(image=image, sources=sources, search_error=search_error)

        if search_event:
            logger.info("[%s] Sending search metadata to frontend", request_id)
            yield f"data: {json.dumps(search_event)}\n\n"

        full_response = ""
        chunk_count = 0
        was_cancelled = False
        stream = None

        try:
            stream = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt + context},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )

            async for chunk in stream:
                if await self._client_disconnected(request, request_id):
                    logger.info("[%s] OpenAI upstream stream stopped after client disconnect", request_id)
                    return

                content = chunk.choices[0].delta.content

                if content:
                    chunk_count += 1

                    if chunk_count == 1:
                        logger.info("[%s] OpenAI first chunk received", request_id)

                    full_response += content

                    yield f"data: {json.dumps({'text': content, 'provider': 'openai'})}\n\n"
        except asyncio.CancelledError:
            logger.info("[%s] OpenAI stream cancelled", request_id)
            raise
        finally:
            if stream is not None and hasattr(stream, "aclose"):
                await stream.aclose()

        add_memory(prompt, full_response)
        logger.info("[%s] OpenAI stream completed | chunks=%s", request_id, chunk_count)


# ---------------------------------------------------------
# GEMINI STREAM
# ---------------------------------------------------------

    async def _stream_gemini(self, request, prompt, model, request_id):

        system_prompt, context, image, sources, search_error = await self._build_context(prompt, None, request_id)
        gemini_client = get_gemini_client()
        logger.info("[%s] Gemini request started | model=%s", request_id, model)

        search_event = self._build_search_event(image=image, sources=sources, search_error=search_error)

        if search_event:
            logger.info("[%s] Sending search metadata to frontend", request_id)
            yield f"data: {json.dumps(search_event)}\n\n"

        full_response = ""
        chunk_count = 0
        stream = None

        try:
            stream = await gemini_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt + context},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )

            async for chunk in stream:
                if await self._client_disconnected(request, request_id):
                    logger.info("[%s] Gemini upstream stream stopped after client disconnect", request_id)
                    return

                content = chunk.choices[0].delta.content

                if content:
                    chunk_count += 1

                    if chunk_count == 1:
                        logger.info("[%s] Gemini first chunk received", request_id)

                    full_response += content

                    yield f"data: {json.dumps({'text': content, 'provider': 'gemini'})}\n\n"
        except asyncio.CancelledError:
            logger.info("[%s] Gemini stream cancelled", request_id)
            raise
        finally:
            if stream is not None and hasattr(stream, "aclose"):
                await stream.aclose()

        add_memory(prompt, full_response)
        logger.info("[%s] Gemini stream completed | chunks=%s", request_id, chunk_count)


# ---------------------------------------------------------
# CONTEXT BUILDER (RAG)
# ---------------------------------------------------------

    async def _build_context(self, prompt, file, request_id):
        logger.info("[%s] Building context | has_file=%s", request_id, file is not None)

        if file:
            if self._is_audio_file(file):
                logger.info(
                    "[%s] Processing uploaded audio | filename=%s | content_type=%s",
                    request_id,
                    file.filename,
                    file.content_type,
                )

                transcript = await self._transcribe_audio_file(file)

                context = f"""
Uploaded Audio Transcript:
{transcript}
"""

                system_prompt = """
You are assisting with a user-uploaded audio transcript.
Use the transcript and the user's prompt together to answer.
If the transcript is unclear or incomplete, say that clearly.
"""

                compressed = compress_context(context)
                logger.info(
                    "[%s] Audio transcript built | transcript_chars=%s | compressed_chars=%s",
                    request_id,
                    len(transcript),
                    len(compressed),
                )
                return system_prompt, compressed, None, [], None

            logger.info(
                "[%s] Extracting uploaded file | filename=%s | content_type=%s",
                request_id,
                file.filename,
                file.content_type,
            )

            file_content = await self._extract_file(file)

            context = f"""
User Uploaded Document:
{file_content}
"""

            system_prompt = """
You are analyzing a user uploaded document.
Answer ONLY using the document content.
"""

            compressed = compress_context(context)
            logger.info(
                "[%s] File context built | extracted_chars=%s | compressed_chars=%s",
                request_id,
                len(file_content),
                len(compressed),
            )
            return system_prompt, compressed, None, [], None


        route = route_query(file)
        logger.info("[%s] Query routed to %s | has_file=%s", request_id, route, file is not None)

        static_chunks = []
        web_chunks = []
        image = None
        sources = []
        search_error = None


        # -------------------------
        # KNOWLEDGE BASE
        # -------------------------

        if route == "kb":

            retrieved_chunks = retrieve(prompt)
            logger.info("[%s] Retrieved %s knowledge base chunks", request_id, len(retrieved_chunks))

            static_chunks = [chunk["text"] for chunk in retrieved_chunks]


        # -------------------------
        # WEB SEARCH
        # -------------------------

        else:

            search_docs = search_searxng(prompt, request_id=request_id)
            summary_text = search_docs.get("summary", "") if search_docs else ""
            image = search_docs.get("image") if search_docs else None
            sources = search_docs.get("sources", []) if search_docs else []
            search_error = search_docs.get("error") if search_docs else None
            document_count = len(search_docs.get("documents", [])) if search_docs else 0
            logger.info(
                "[%s] Web search returned summary_chars=%s | documents=%s | has_image=%s | sources=%s | error=%s",
                request_id,
                len(summary_text),
                document_count,
                image is not None,
                len(sources),
                search_error,
            )

            web_chunks = self._build_web_chunks(prompt, search_docs, request_id)


        all_docs = static_chunks + web_chunks
        logger.info("[%s] Total context chunks before rerank: %s", request_id, len(all_docs))

        if not all_docs:
            logger.warning("[%s] No knowledge retrieved for prompt", request_id)
            fallback_context = "No knowledge retrieved."

            if search_error:
                fallback_context = f"Web search failed: {search_error}."

            return PNB_SYSTEM_PROMPT, fallback_context, image, sources, search_error


        best_docs = rerank(prompt, all_docs, top_k=5)
        logger.info("[%s] Reranked context to %s chunks", request_id, len(best_docs))

        context_block = "\n".join(best_docs)

        memory = get_memory()

        context = f"""
Conversation History:
{memory}

Retrieved Knowledge:
{context_block}
"""

        compressed = compress_context(context)
        logger.info("[%s] Context compression complete | compressed_chars=%s", request_id, len(compressed))
        return PNB_SYSTEM_PROMPT, compressed, image, sources, search_error


# ---------------------------------------------------------
# OLLAMA STREAM
# ---------------------------------------------------------

    async def _stream_ollama(self, request, prompt, time, file, request_id):

        system_prompt, context, image, sources, search_error = await self._build_context(prompt, file, request_id)

        final_prompt = f"""
{system_prompt}

{context}

Current time: {time}

User Question:
{prompt}

Answer clearly using the context.
After answering ask ONE relevant follow-up question.
"""

        errors = []

        timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            for ollama_url in self._get_ollama_urls():
                try:
                    if await self._client_disconnected(request, request_id):
                        return

                    logger.info("[%s] Connecting to Ollama at %s", request_id, ollama_url)

                    async with client.stream(
                        "POST",
                        ollama_url,
                        json={
                            "model": "mistral",
                            "prompt": final_prompt,
                            "stream": True
                        }
                    ) as response:
                        logger.info("[%s] Ollama response status: %s", request_id, response.status_code)
                        response.raise_for_status()

                        full_response = ""
                        chunk_count = 0

                        search_event = self._build_search_event(image=image, sources=sources, search_error=search_error)

                        if search_event:
                            logger.info("[%s] Sending search metadata to frontend", request_id)
                            yield f"data: {json.dumps(search_event)}\n\n"

                        async for line in response.aiter_lines():
                            if await self._client_disconnected(request, request_id):
                                logger.info("[%s] Closing Ollama upstream stream after client disconnect", request_id)
                                logger.info("[%s] Ollama processing stopped", request_id)
                                return

                            if not line:
                                continue

                            data = json.loads(line)

                            if "response" in data:
                                chunk_count += 1

                                if chunk_count == 1:
                                    logger.info("[%s] Ollama first chunk received", request_id)

                                text = data["response"]

                                full_response += text

                                yield f"data: {json.dumps({'text': text, 'provider': 'ollama'})}\n\n"

                        add_memory(prompt, full_response)
                        logger.info("[%s] Ollama stream completed | chunks=%s", request_id, chunk_count)
                        return

                except asyncio.CancelledError:
                    logger.info("[%s] Ollama stream cancelled", request_id)
                    raise

                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.HTTPStatusError) as exc:
                    logger.warning("[%s] Ollama request failed for %s: %s", request_id, ollama_url, exc)
                    errors.append(f"{ollama_url} -> {exc}")

        raise RuntimeError(
            "Could not connect to Ollama. Set OLLAMA_URL to the reachable endpoint. "
            f"Tried: {'; '.join(errors) if errors else 'no endpoints'}"
        )


# ---------------------------------------------------------
# LOCAL MODEL STREAM
# ---------------------------------------------------------

    async def _stream_local(self, app, request, prompt, time, file=None, request_id="unknown"):
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class DisconnectStoppingCriteria(StoppingCriteria):
            def __init__(self, stop_event):
                self.stop_event = stop_event

            def __call__(self, input_ids, scores, **kwargs):
                return self.stop_event.is_set()

        local_model = app.state.model
        tokenizer = app.state.tokenizer
        device = getattr(app.state, "device", "cpu")
        stop_event = Event()
        logger.info("[%s] Local model stream started | device=%s", request_id, device)

        system_prompt, context, image, sources, search_error = await self._build_context(prompt, file, request_id)

        system_message = f"""
{system_prompt}

{context}

Current time: {time}
"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            input_text,
            return_tensors="pt"
        ).to(device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.25,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([DisconnectStoppingCriteria(stop_event)]),
        )

        thread = Thread(
            target=local_model.generate,
            kwargs=generation_kwargs
        )

        thread.start()
        logger.info("[%s] Local generation thread started", request_id)

        full_response = ""
        chunk_count = 0

        search_event = self._build_search_event(image=image, sources=sources, search_error=search_error)

        if search_event:
            logger.info("[%s] Sending search metadata to frontend", request_id)
            yield f"data: {json.dumps(search_event)}\n\n"

        streamer_iter = iter(streamer)

        try:
            while True:
                if await self._client_disconnected(request, request_id):
                    stop_event.set()
                    logger.info("[%s] Stopping local generation after client disconnect", request_id)
                    was_cancelled = True
                    break

                try:
                    new_text = next(streamer_iter)
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    await asyncio.sleep(0)
                    continue
                except StopIteration:
                    break

                if new_text:
                    chunk_count += 1

                    if chunk_count == 1:
                        logger.info("[%s] Local model first chunk received", request_id)

                    full_response += new_text

                    yield f"data: {json.dumps({'text': new_text, 'provider': 'pnb-local'})}\n\n"
        except asyncio.CancelledError:
            stop_event.set()
            logger.info("[%s] Local model stream cancelled", request_id)
            raise
        finally:
            stop_event.set()
            thread.join(timeout=1)

        if not was_cancelled:
            add_memory(prompt, full_response)
            logger.info("[%s] Local model stream completed | chunks=%s", request_id, chunk_count)


# ---------------------------------------------------------
# FILE EXTRACTION
# ---------------------------------------------------------

    async def _extract_file(self, file):

        content_type = (file.content_type or "").lower()
        extension = self._get_file_extension(file)

        if content_type == "application/pdf" or extension == ".pdf":

            pdf_bytes = await file.read()

            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            logger.info("Extracting PDF | bytes=%s | pages=%s", len(pdf_bytes), len(pdf_document))

            text = ""

            for page in pdf_document:
                text += page.get_text()

            return text


        elif content_type in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel"
        ] or extension in {".xlsx", ".xls"}:

            excel_bytes = await file.read()

            df = pd.read_excel(BytesIO(excel_bytes))
            logger.info("Extracting spreadsheet | bytes=%s | rows=%s | cols=%s", len(excel_bytes), len(df), len(df.columns))

            schema = {
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict()
            }

            data_json = df.to_json(orient="records")

            return f"""
Excel Schema:
{json.dumps(schema)}

Excel Data:
{data_json}
"""


        elif content_type in {"text/plain", "text/csv"} or extension in {".txt", ".csv"}:

            text_bytes = await file.read()
            text = text_bytes.decode("utf-8", errors="ignore")
            logger.info("Extracting text file | filename=%s | bytes=%s", file.filename, len(text_bytes))
            return text

        elif content_type.startswith("image/") or extension in self.IMAGE_EXTENSIONS:
            logger.info("Image upload received | filename=%s | content_type=%s", file.filename, file.content_type)
            return f"Uploaded image file: {file.filename}"

        else:

            return ""
