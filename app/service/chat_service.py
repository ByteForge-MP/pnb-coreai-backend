import json
import torch
import fitz
import httpx
import pandas as pd
import logging
import time as time_module

from io import BytesIO
from threading import Thread
from transformers import TextIteratorStreamer

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

# OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_URL = "http://ollama:11434/api/generate"


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

            match model:

                case "gpt-4o" | "gpt-3.5-turbo":
                    logger.info("[%s] Routing request to OpenAI stream", request_id)

                    async for chunk in self._stream_openai(prompt, model, request_id):
                        yield chunk

                case "gemini-3-flash-preview":
                    logger.info("[%s] Routing request to Gemini stream", request_id)

                    async for chunk in self._stream_gemini(prompt, model, request_id):
                        yield chunk

                case "pnb-local-model":
                    logger.info("[%s] Routing request to local model stream", request_id)

                    async for chunk in self._stream_local(app, prompt, time, file, request_id):
                        yield chunk

                case _:
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

        system_prompt, context, image = await self._build_context(prompt, None, request_id)
        openai_client = get_openai_client()
        logger.info("[%s] OpenAI request started | model=%s", request_id, model)

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

        system_prompt, context, image = await self._build_context(prompt, None, request_id)
        gemini_client = get_gemini_client()
        logger.info("[%s] Gemini request started | model=%s", request_id, model)

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
            return system_prompt, compressed, None


        route = route_query(prompt)
        logger.info("[%s] Query routed to %s", request_id, route)

        static_chunks = []
        web_chunks = []
        image = None


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
            logger.info("[%s] Web search returned %s documents", request_id, len(search_docs) if search_docs else 0)

            if search_docs:

                dynamic_index, dynamic_chunks = build_dynamic_embeddings(search_docs)

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

            return PNB_SYSTEM_PROMPT, "No knowledge retrieved.", image


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
        return PNB_SYSTEM_PROMPT, compressed, image


# ---------------------------------------------------------
# OLLAMA STREAM
# ---------------------------------------------------------

    async def _stream_ollama(self, prompt, time, file, request_id):

        system_prompt, context, image = await self._build_context(prompt, file, request_id)

        final_prompt = f"""
{system_prompt}

{context}

Current time: {time}

User Question:
{prompt}

Answer clearly using the context.
After answering ask ONE relevant follow-up question.
"""

        async with httpx.AsyncClient(timeout=None) as client:
            logger.info("[%s] Connecting to Ollama at %s", request_id, OLLAMA_URL)

            async with client.stream(
                "POST",
                OLLAMA_URL,
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

                # send image once
                if image:
                    logger.info("[%s] Sending image metadata to frontend", request_id)
                    yield f"data: {json.dumps({'image': image})}\n\n"

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


# ---------------------------------------------------------
# LOCAL MODEL STREAM
# ---------------------------------------------------------

    async def _stream_local(self, app, prompt, time, file=None, request_id="unknown"):

        local_model = app.state.model
        tokenizer = app.state.tokenizer
        device = getattr(app.state, "device", "cpu")
        logger.info("[%s] Local model stream started | device=%s", request_id, device)

        system_prompt, context, image = await self._build_context(prompt, file, request_id)

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
