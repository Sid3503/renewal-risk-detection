"""
LLM client — OpenAI GPT-4o-mini via LangChain, with retry and Pydantic validation.

All LLM calls go through this module.
Callers pass a prompt and a Pydantic schema — they get back a validated object or None.
"""
import os
import logging
import time
from typing import TypeVar, Type
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_SYSTEM_BASE = (
    "You are a structured data extraction assistant. "
    "You receive text and return ONLY valid JSON matching the schema provided. "
    "No markdown, no explanation, no preamble. Just JSON."
)


def call_llm(
    user_prompt: str,
    output_schema: Type[T],
    system_prompt: str = _SYSTEM_BASE,
    max_retries: int = 3,
    temperature: float = 0.1,
) -> T | None:
    """
    Call GPT-4o-mini via LangChain and parse the response as output_schema.
    Returns None on persistent failure — callers must handle None.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError:
        logger.error("langchain-openai not installed. Run: pip install langchain-openai")
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        return None

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=api_key,
    ).with_structured_output(output_schema)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(max_retries):
        try:
            result = model.invoke(messages)
            return result

        except Exception as e:
            is_last = attempt == max_retries - 1
            log_fn = logger.error if is_last else logger.warning
            log_fn("LLM call error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if not is_last:
                time.sleep(2.0 * (attempt + 1))

    return None
