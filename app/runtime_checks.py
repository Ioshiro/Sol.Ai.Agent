from __future__ import annotations

import logging

import httpx

from app.config import AppConfig

logger = logging.getLogger(__name__)


async def check_openai(config: AppConfig) -> None:
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required.")

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{config.openai_base_url}/models",
            headers={"Authorization": f"Bearer {config.openai_api_key}"},
        )
        response.raise_for_status()
        logger.info("OpenAI STT/TTS endpoint ready at %s", config.openai_base_url)


async def check_llm_service(config: AppConfig) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{config.llm_service_base_url}/models",
            headers={"Authorization": f"Bearer {config.llm_service_api_key}"},
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        model_ids = {item.get("id") for item in models}
        if config.llm_service_model not in model_ids:
            raise RuntimeError(
                f"LLM service model '{config.llm_service_model}' not found. Available models: {', '.join(sorted(m for m in model_ids if m))}"
            )
        logger.info("LLM service ready with model '%s'", config.llm_service_model)
