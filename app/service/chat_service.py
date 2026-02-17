import json
from threading import Thread
from transformers import TextIteratorStreamer
from app.ai.openai_client import openai_client, gemini_client

class ChatService:
    async def get_streaming_response(self, app, prompt: str, model: str, time: str):
        """
        Switch case to serve different models. 
        'app' is required to access the local model stored in app.state.
        """
        try:
            match model:
                case "gpt-4o" | "gpt-3.5-turbo":
                    async for chunk in self._stream_openai(prompt, model):
                        yield chunk

                case "gemini-3-flash-preview":
                    async for chunk in self._stream_gemini(prompt, model):
                        yield chunk

                case "pnb-local-model":
                    # FIX: Pass all 4 required arguments: app, prompt, model, time
                    async for chunk in self._stream_local(app, prompt, model, time):
                        yield chunk

                case _:
                    # Default to local model with all arguments
                    async for chunk in self._stream_local(app, prompt, model, time):
                        yield chunk

        except Exception as e:
            yield f"data: {json.dumps({'error': 'Switch Case Failed', 'details': str(e)})}\n\n"
        
        yield "data: [DONE]\n\n"

    # --- Helper Methods ---

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

    async def _stream_local(self, app, prompt: str, model_name: str, time: str):
        # 1. Grab model and tokenizer from the app state (loaded during lifespan)
        local_model = app.state.model
        tokenizer = app.state.tokenizer
        device = getattr(app.state, "device", "mps") 

        # 2. Prepare the PNB-themed prompt
        messages = [
            {"role": "system", "content": f"You are a helpful PNB Assistant. It is currently {time}."},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # 3. Setup the Streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 4. Run generation in a background thread
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
        thread = Thread(target=local_model.generate, kwargs=generation_kwargs)
        thread.start()

        # 5. Yield chunks to frontend
        # yield f"data: {json.dumps({'text': f'Good {time}! ', 'provider': 'pnb-local'})}\n\n"

        for new_text in streamer:
            if new_text:
                yield f"data: {json.dumps({'text': new_text, 'provider': 'pnb-local'})}\n\n"