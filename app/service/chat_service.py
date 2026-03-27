import json
import os
import torch
import fitz
import httpx
import pandas as pd
import logging
import time as time_module

from io import BytesIO
from threading import Thread

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

Conversation Style:
- At the end of helpful answers, ask ONE short follow-up question.
- Do not ask generic greetings.
- Do not ask a question if the answer is already complete.
"""


class ChatService:

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


# ---------------------------------------------------------
# MAIN STREAM ROUTER
# ---------------------------------------------------------

    async def get_streaming_response(self, app, prompt, model, time, file=None, request_id="unknown"):
        started_at = time_module.monotonic()

        logger.info(
            "[%s] Stream started | model=%s | ollama_mode=%s | prompt_preview=%r",
            request_id,
            model,
            getattr(app.state, "use_ollama", False),
            self._prompt_preview(prompt),
        )

        try:

            if getattr(app.state, "use_ollama", False):
                logger.info("[%s] Routing request to Ollama stream", request_id)

                async for chunk in self._stream_ollama(prompt, time, file, request_id):
                    yield chunk

                logger.info(
                    "[%s] Stream completed via Ollama in %.2fs",
                    request_id,
                    time_module.monotonic() - started_at,
                )
                yield "data: [DONE]\n\n"
                return

            if model in {"gpt-4o", "gpt-3.5-turbo"}:
                logger.info("[%s] Routing request to OpenAI stream", request_id)

                async for chunk in self._stream_openai(prompt, model, request_id):
                    yield chunk

            elif model == "gemini-3-flash-preview":
                logger.info("[%s] Routing request to Gemini stream", request_id)

                async for chunk in self._stream_gemini(prompt, model, request_id):
                    yield chunk

            elif model == "pnb-local-model":
                logger.info("[%s] Routing request to local model stream", request_id)

                async for chunk in self._stream_local(app, prompt, time, file, request_id):
                    yield chunk

            else:
                logger.info("[%s] Unknown model %s, defaulting to local stream", request_id, model)

                async for chunk in self._stream_local(app, prompt, time, file, request_id):
                    yield chunk

        except Exception as e:

            logger.exception("[%s] Streaming failed: %s", request_id, str(e))

            yield f"data: {json.dumps({'error': 'Switch Case Failed', 'details': str(e)})}\n\n"

        logger.info(
            "[%s] Stream finished in %.2fs",
            request_id,
            time_module.monotonic() - started_at,
        )
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------
# OPENAI STREAM
# ---------------------------------------------------------

    async def _stream_openai(self, prompt, model, request_id):

        system_prompt, context, image, sources, search_error = await self._build_context(prompt, None, request_id)
        openai_client = get_openai_client()
        logger.info("[%s] OpenAI request started | model=%s", request_id, model)

        search_event = self._build_search_event(image=image, sources=sources, search_error=search_error)

        if search_event:
            logger.info("[%s] Sending search metadata to frontend", request_id)
            yield f"data: {json.dumps(search_event)}\n\n"

        stream = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt + context},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        full_response = ""
        chunk_count = 0

        async for chunk in stream:

            content = chunk.choices[0].delta.content

            if content:
                chunk_count += 1

                if chunk_count == 1:
                    logger.info("[%s] OpenAI first chunk received", request_id)

                full_response += content

                yield f"data: {json.dumps({'text': content, 'provider': 'openai'})}\n\n"

        add_memory(prompt, full_response)
        logger.info("[%s] OpenAI stream completed | chunks=%s", request_id, chunk_count)


# ---------------------------------------------------------
# GEMINI STREAM
# ---------------------------------------------------------

    async def _stream_gemini(self, prompt, model, request_id):

        system_prompt, context, image, sources, search_error = await self._build_context(prompt, None, request_id)
        gemini_client = get_gemini_client()
        logger.info("[%s] Gemini request started | model=%s", request_id, model)

        search_event = self._build_search_event(image=image, sources=sources, search_error=search_error)

        if search_event:
            logger.info("[%s] Sending search metadata to frontend", request_id)
            yield f"data: {json.dumps(search_event)}\n\n"

        stream = await gemini_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt + context},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        full_response = ""
        chunk_count = 0

        async for chunk in stream:

            content = chunk.choices[0].delta.content

            if content:
                chunk_count += 1

                if chunk_count == 1:
                    logger.info("[%s] Gemini first chunk received", request_id)

                full_response += content

                yield f"data: {json.dumps({'text': content, 'provider': 'gemini'})}\n\n"

        add_memory(prompt, full_response)
        logger.info("[%s] Gemini stream completed | chunks=%s", request_id, chunk_count)


# ---------------------------------------------------------
# CONTEXT BUILDER (RAG)
# ---------------------------------------------------------

    async def _build_context(self, prompt, file, request_id):
        logger.info("[%s] Building context | has_file=%s", request_id, file is not None)

        if file:
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
            logger.info(
                "[%s] Web search returned summary_chars=%s | has_image=%s | sources=%s | error=%s",
                request_id,
                len(summary_text),
                image is not None,
                len(sources),
                search_error,
            )

            if summary_text:

                dynamic_index, dynamic_chunks = build_dynamic_embeddings([summary_text])

                query_embedding = embed_query(prompt)

                D, I = dynamic_index.search(query_embedding, 5)

                for idx in I[0]:

                    chunk = dynamic_chunks[idx]

                    if isinstance(chunk, dict):

                        web_chunks.append(chunk.get("text", ""))

                        if not image and chunk.get("image"):
                            image = chunk["image"]

                    else:

                        web_chunks.append(chunk)


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

    async def _stream_ollama(self, prompt, time, file, request_id):

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

    async def _stream_local(self, app, prompt, time, file=None, request_id="unknown"):
        from transformers import TextIteratorStreamer

        local_model = app.state.model
        tokenizer = app.state.tokenizer
        device = getattr(app.state, "device", "cpu")
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
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
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

        for new_text in streamer:

            if new_text:
                chunk_count += 1

                if chunk_count == 1:
                    logger.info("[%s] Local model first chunk received", request_id)

                full_response += new_text

                yield f"data: {json.dumps({'text': new_text, 'provider': 'pnb-local'})}\n\n"

        add_memory(prompt, full_response)
        logger.info("[%s] Local model stream completed | chunks=%s", request_id, chunk_count)


# ---------------------------------------------------------
# FILE EXTRACTION
# ---------------------------------------------------------

    async def _extract_file(self, file):

        content_type = file.content_type

        if content_type == "application/pdf":

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
        ]:

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


        else:

            return ""
