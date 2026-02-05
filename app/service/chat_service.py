import json
import asyncio
from app.ai.openai_client import openai_client, gemini_client

class ChatService:
    async def get_streaming_response(self, prompt: str, model: str):
        """
        Attempts to stream from OpenAI, falls back to Gemini on failure.
        """
        try:
            # --- TRY OPENAI ---
            stream = await openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'text': content, 'provider': 'openai'})}\n\n"

        except Exception as e:
            # --- FALLBACK TO GEMINI ---
            print(f"OpenAI Failed: {e}. Switching to Gemini fallback...")
            try:
                # Using Gemini 3 Flash (Free Tier)
                fallback_stream = await gemini_client.chat.completions.create(
                    model="gemini-3-flash-preview", 
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                async for chunk in fallback_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {json.dumps({'text': content, 'provider': 'gemini'})}\n\n"
            except Exception as gemini_err:
                yield f"data: {json.dumps({'error': 'All providers failed', 'details': str(gemini_err)})}\n\n"

        yield "data: [DONE]\n\n"