import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


def _get_required_env(var_name: str) -> str:
    value = os.getenv(var_name)

    if not value:
        raise RuntimeError(
            f"{var_name} is not set. Provide it in the environment before using this provider."
        )

    return value


def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=_get_required_env("OPENAI_API_KEY"))


def get_gemini_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_get_required_env("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
