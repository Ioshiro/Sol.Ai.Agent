from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    openai_base_url: str
    openai_api_key: str
    openai_stt_model: str
    openai_tts_model: str
    openai_tts_voice: str
    llm_service_base_url: str
    llm_service_model: str
    llm_service_api_key: str
    assistant_language: str
    log_level: str

    @classmethod
    def load(cls, *, require_llm_model: bool = True) -> "AppConfig":
        load_dotenv()
        llm_service_model = os.getenv("LLM_SERVICE_MODEL", "").strip()
        if require_llm_model and not llm_service_model:
            raise ValueError("LLM_SERVICE_MODEL is required. Set it in your .env file.")

        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_stt_model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
            openai_tts_model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
            llm_service_base_url=os.getenv("LLM_SERVICE_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/"),
            llm_service_model=llm_service_model,
            llm_service_api_key=os.getenv("LLM_SERVICE_API_KEY", "lm-studio").strip(),
            assistant_language=os.getenv("ASSISTANT_LANGUAGE", "it"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
