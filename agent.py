from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import openai, silero

from app.config import AppConfig
from app.logging_utils import configure_logging
from app.runtime_checks import check_llm_service, check_openai

logger = logging.getLogger(__name__)


@dataclass
class TurnLatencyTracker:
    user_stopped_speaking_at: float | None = None
    user_final_transcript_at: float | None = None
    awaiting_tts_start: bool = False


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


class LocalVoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Sei un assistente vocale. Rispondi in italiano, in modo chiaro e sintetico. "
                "Se l'utente chiede aiuto pratico, proponi passi concreti e mantieni uno stile colloquiale."
            ),
        )


async def entrypoint(ctx: JobContext) -> None:
    config = AppConfig.load()
    configure_logging(config.log_level)
    await asyncio.gather(check_openai(config), check_llm_service(config))

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(model=config.openai_stt_model, base_url=config.openai_base_url, api_key=config.openai_api_key),
        llm=openai.LLM(
            model=config.llm_service_model,
            base_url=config.llm_service_base_url,
            api_key=config.llm_service_api_key,
        ),
        tts=openai.TTS(
            model=config.openai_tts_model,
            voice=config.openai_tts_voice,
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
        ),
    )

    tracker = TurnLatencyTracker()

    @session.on("user_state_changed")
    def _on_user_state_changed(event) -> None:
        if event.old_state == "speaking" and event.new_state == "listening":
            tracker.user_stopped_speaking_at = event.created_at
            tracker.awaiting_tts_start = True

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event) -> None:
        if event.is_final:
            tracker.user_final_transcript_at = event.created_at

    async def _attach_tts_start_logger() -> None:
        while session.output.audio is None:
            await asyncio.sleep(0.05)

        @session.output.audio.on("playback_started")
        def _on_playback_started(event) -> None:
            if not tracker.awaiting_tts_start:
                return

            tracker.awaiting_tts_start = False
            stopped_delta = (
                event.created_at - tracker.user_stopped_speaking_at
                if tracker.user_stopped_speaking_at is not None
                else None
            )
            transcript_delta = (
                event.created_at - tracker.user_final_transcript_at
                if tracker.user_final_transcript_at is not None
                else None
            )

            if stopped_delta is not None:
                logger.info(
                    "Latency user-stop -> TTS-start: %.3fs%s",
                    stopped_delta,
                    (
                        f" (final transcript -> TTS-start: {transcript_delta:.3f}s)"
                        if transcript_delta is not None
                        else ""
                    ),
                )

    asyncio.create_task(_attach_tts_start_logger())

    await session.start(agent=LocalVoiceAssistant(), room=ctx.room)
    session.say("Ciao, come posso aiutarti?")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            ws_url="ws://localhost:7880",
            api_key="devkey",
            api_secret="secret",
        )
    )
