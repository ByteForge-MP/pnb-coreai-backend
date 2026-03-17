import json
import torch
import fitz
import httpx
import pandas as pd
import logging

from io import BytesIO
from threading import Thread
from transformers import TextIteratorStreamer

from app.ai.openai_client import openai_client, gemini_client
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

OLLAMA_URL = "http://localhost:11434/api/generate"


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


# ---------------------------------------------------------
# MAIN STREAM ROUTER
# ---------------------------------------------------------

    async def get_streaming_response(self, app, prompt, model, time, file=None):

        try:

            if getattr(app.state, "use_ollama", False):

                async for chunk in self._stream_ollama(prompt, time, file):
                    yield chunk

                yield "data: [DONE]\n\n"
                return

            match model:

                case "gpt-4o" | "gpt-3.5-turbo":

                    async for chunk in self._stream_openai(prompt, model):
                        yield chunk

                case "gemini-3-flash-preview":

                    async for chunk in self._stream_gemini(prompt, model):
                        yield chunk

                case "pnb-local-model":

                    async for chunk in self._stream_local(app, prompt, time, file):
                        yield chunk

                case _:

                    async for chunk in self._stream_local(app, prompt, time, file):
                        yield chunk

        except Exception as e:

            logger.error(str(e))

            yield f"data: {json.dumps({'error': 'Switch Case Failed', 'details': str(e)})}\n\n"

        yield "data: [DONE]\n\n"


# ---------------------------------------------------------
# OPENAI STREAM
# ---------------------------------------------------------

    async def _stream_openai(self, prompt, model):

        system_prompt, context, image = await self._build_context(prompt, None)

        stream = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt + context},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        full_response = ""

        async for chunk in stream:

            content = chunk.choices[0].delta.content

            if content:

                full_response += content

                yield f"data: {json.dumps({'text': content, 'provider': 'openai'})}\n\n"

        add_memory(prompt, full_response)


# ---------------------------------------------------------
# GEMINI STREAM
# ---------------------------------------------------------

    async def _stream_gemini(self, prompt, model):

        system_prompt, context, image = await self._build_context(prompt, None)

        stream = await gemini_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt + context},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        full_response = ""

        async for chunk in stream:

            content = chunk.choices[0].delta.content

            if content:

                full_response += content

                yield f"data: {json.dumps({'text': content, 'provider': 'gemini'})}\n\n"

        add_memory(prompt, full_response)


# ---------------------------------------------------------
# CONTEXT BUILDER (RAG)
# ---------------------------------------------------------

    async def _build_context(self, prompt, file):

        if file:

            file_content = await self._extract_file(file)

            context = f"""
User Uploaded Document:
{file_content}
"""

            system_prompt = """
You are analyzing a user uploaded document.
Answer ONLY using the document content.
"""

            return system_prompt, compress_context(context), None


        route = route_query(prompt)

        static_chunks = []
        web_chunks = []
        image = None


        # -------------------------
        # KNOWLEDGE BASE
        # -------------------------

        if route == "kb":

            retrieved_chunks = retrieve(prompt)

            static_chunks = [chunk["text"] for chunk in retrieved_chunks]


        # -------------------------
        # WEB SEARCH
        # -------------------------

        else:

            search_docs = search_searxng(prompt)

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

        if not all_docs:

            return PNB_SYSTEM_PROMPT, "No knowledge retrieved.", image


        best_docs = rerank(prompt, all_docs, top_k=5)

        context_block = "\n".join(best_docs)

        memory = get_memory()

        context = f"""
Conversation History:
{memory}

Retrieved Knowledge:
{context_block}
"""

        return PNB_SYSTEM_PROMPT, compress_context(context), image


# ---------------------------------------------------------
# OLLAMA STREAM
# ---------------------------------------------------------

    async def _stream_ollama(self, prompt, time, file):

        system_prompt, context, image = await self._build_context(prompt, file)

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

            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": "mistral",
                    "prompt": final_prompt,
                    "stream": True
                }
            ) as response:

                full_response = ""

                # send image once
                if image:
                    yield f"data: {json.dumps({'image': image})}\n\n"

                async for line in response.aiter_lines():

                    if not line:
                        continue

                    data = json.loads(line)

                    if "response" in data:

                        text = data["response"]

                        full_response += text

                        yield f"data: {json.dumps({'text': text, 'provider': 'ollama'})}\n\n"

                add_memory(prompt, full_response)


# ---------------------------------------------------------
# LOCAL MODEL STREAM
# ---------------------------------------------------------

    async def _stream_local(self, app, prompt, time, file=None):

        local_model = app.state.model
        tokenizer = app.state.tokenizer
        device = getattr(app.state, "device", "cpu")

        system_prompt, context, image = await self._build_context(prompt, file)

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

        full_response = ""

        for new_text in streamer:

            if new_text:

                full_response += new_text

                yield f"data: {json.dumps({'text': new_text, 'provider': 'pnb-local'})}\n\n"

        add_memory(prompt, full_response)


# ---------------------------------------------------------
# FILE EXTRACTION
# ---------------------------------------------------------

    async def _extract_file(self, file):

        content_type = file.content_type

        if content_type == "application/pdf":

            pdf_bytes = await file.read()

            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

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