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
    if not config.llm_service_base_url:
        raise ValueError("LLM_SERVICE_BASE_URL is required.")

    if not config.llm_service_api_key:
        raise ValueError("LLM_SERVICE_API_KEY is required.")

    if not config.llm_service_model:
        raise ValueError("LLM_SERVICE_MODEL is required.")

    # Runpod/OpenAI-compatible endpoints do not consistently expose /models, so
    # we only validate configuration here and let the first chat completion call
    # prove the upstream endpoint shape.
    logger.info(
        "LLM service configured with endpoint %s and model '%s'",
        config.llm_service_base_url,
        config.llm_service_model,
    )
