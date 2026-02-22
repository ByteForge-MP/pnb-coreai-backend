import json
from threading import Thread
from transformers import TextIteratorStreamer
from app.ai.openai_client import openai_client, gemini_client
from app.retriever import retrieve

PNB_SYSTEM_PROMPT = """
You are a highly qualified digital assistant of Punjab National Bank of India.

Rules you must follow:
- Respond professionally and formally.
- Give accurate banking and compliance-oriented answers.
- If unsure, say verification is required instead of guessing.
- Do not provide unsafe financial advice.
- Structure responses clearly.
- Never reveal internal knowledge sources or retrieval process.
"""

MAX_CONTEXT_CHARS = 3500   # prevents prompt overflow


class ChatService:

    async def get_streaming_response(self, app, prompt: str, model: str, time: str):
        try:
            match model:

                case "gpt-4o" | "gpt-3.5-turbo":
                    async for chunk in self._stream_openai(prompt, model):
                        yield chunk

                case "gemini-3-flash-preview":
                    async for chunk in self._stream_gemini(prompt, model):
                        yield chunk

                case "pnb-local-model":
                    async for chunk in self._stream_local(app, prompt, model, time):
                        yield chunk

                case _:
                    async for chunk in self._stream_local(app, prompt, model, time):
                        yield chunk

        except Exception as e:
            yield f"data: {json.dumps({'error': 'Switch Case Failed', 'details': str(e)})}\n\n"

        yield "data: [DONE]\n\n"


    # ---------- OPENAI ----------
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


    # ---------- GEMINI ----------
    async def _stream_gemini(self, prompt, model):
        fallback_stream = await gemini_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in fallback_stream:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {json.dumps({'text': content, 'provider': 'gemini'})}\n\n"


    # ---------- LOCAL MODEL WITH RAG ----------
    async def _stream_local(self, app, prompt: str, model_name: str, time: str):

        local_model = app.state.model
        tokenizer = app.state.tokenizer
        device = getattr(app.state, "device", "cpu")

        # 🔹 STEP 1: Retrieve relevant knowledge silently
        retrieved_chunks = retrieve(prompt)

        context_block = "\n".join(retrieved_chunks)

        # 🔹 STEP 2: Limit context size (very important in production)
        context_block = context_block[:MAX_CONTEXT_CHARS]

        # 🔹 STEP 3: Build system prompt with hidden RAG context
        system_message = f"""
{PNB_SYSTEM_PROMPT}

Internal Banking Knowledge:
{context_block}

Current time: {time}
"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        # 🔹 STEP 4: Tokenize
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # 🔹 STEP 5: Setup streamer
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=500,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # 🔹 STEP 6: Generate in background thread
        thread = Thread(target=local_model.generate, kwargs=generation_kwargs)
        thread.start()

        # 🔹 STEP 7: Stream tokens
        for new_text in streamer:
            if new_text:
                yield f"data: {json.dumps({'text': new_text, 'provider': 'pnb-local'})}\n\n"