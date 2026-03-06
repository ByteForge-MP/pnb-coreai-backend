import json
import torch
import fitz
import httpx
import pandas as pd

from io import BytesIO
from threading import Thread
from transformers import TextIteratorStreamer

from app.ai.openai_client import openai_client, gemini_client
from app.retriever import retrieve

import logging

logger = logging.getLogger("chat_logger")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

OLLAMA_URL = "http://localhost:11434/api/generate"

PNB_SYSTEM_PROMPT = """
Rules you must follow:
- You are an assistant for a bank's internal use, helping employees answer questions based on the bank's knowledge and documents.
- Always use the provided knowledge and documents to answer questions.
- Do not use any external information.
- If the answer is not in the provided knowledge or documents, say:
  "Sorry, I don't know the answer to that question based on the information I have."
- Be concise and use only relevant information.
"""

MAX_CONTEXT_CHARS = 3500


class ChatService:

    async def get_streaming_response(
        self,
        app,
        prompt: str,
        model: str,
        time: str,
        file=None
    ):

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

    # -------------------------------------------------
    # OPENAI STREAM
    # -------------------------------------------------

    async def _stream_openai(self, prompt, model):

        stream = await openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        async for chunk in stream:

            content = chunk.choices[0].delta.content

            if content:
                yield f"data: {json.dumps({'text': content, 'provider': 'openai'})}\n\n"

    # -------------------------------------------------
    # GEMINI STREAM
    # -------------------------------------------------

    async def _stream_gemini(self, prompt, model):

        stream = await gemini_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        async for chunk in stream:

            content = chunk.choices[0].delta.content

            if content:
                yield f"data: {json.dumps({'text': content, 'provider': 'gemini'})}\n\n"

    # -------------------------------------------------
    # BUILD CONTEXT
    # -------------------------------------------------

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
Do not use external knowledge.
"""

        else:

            retrieved_chunks = retrieve(prompt)

            context_block = "\n".join([chunk["text"] for chunk in retrieved_chunks])

            context = f"""
Internal Banking Knowledge:
{context_block}
"""

            system_prompt = PNB_SYSTEM_PROMPT

        context = context[:MAX_CONTEXT_CHARS]

        return system_prompt, context

    # -------------------------------------------------
    # OLLAMA STREAM
    # -------------------------------------------------

    async def _stream_ollama(self, prompt, time, file):

        system_prompt, context = await self._build_context(prompt, file)

        final_prompt = f"""
{system_prompt}

{context}

Current time: {time}

User Question:
{prompt}
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

                async for line in response.aiter_lines():

                    if not line:
                        continue

                    data = json.loads(line)

                    if "response" in data:

                        yield f"data: {json.dumps({'text': data['response'], 'provider': 'ollama'})}\n\n"

    # -------------------------------------------------
    # LOCAL MODEL STREAM
    # -------------------------------------------------

    async def _stream_local(self, app, prompt: str, time: str, file=None):

        local_model = app.state.model
        tokenizer = app.state.tokenizer
        device = getattr(app.state, "device", "cpu")

        system_prompt, context = await self._build_context(prompt, file)

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
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        thread = Thread(
            target=local_model.generate,
            kwargs=generation_kwargs
        )

        thread.start()

        for new_text in streamer:

            if new_text:
                yield f"data: {json.dumps({'text': new_text, 'provider': 'pnb-local'})}\n\n"

    # -------------------------------------------------
    # FILE EXTRACTION
    # -------------------------------------------------

    async def _extract_file(self, file):

        content_type = file.content_type

        # -----------------------------
        # PDF
        # -----------------------------

        if content_type == "application/pdf":

            pdf_bytes = await file.read()

            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

            text = ""

            for page in pdf_document:
                text += page.get_text()

            return text

        # -----------------------------
        # EXCEL
        # -----------------------------

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

        # -----------------------------
        # OTHER
        # -----------------------------

        else:
            return ""