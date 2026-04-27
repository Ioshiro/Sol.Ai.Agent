from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import openai, silero

from app.config import AppConfig
from app.logging_utils import configure_logging
from app.observability import (
    VoiceTraceRecorder,
    configure_langfuse_tracing,
    propagate_voice_attributes,
    shutdown_langfuse,
    start_voice_trace,
)
from app.runtime_checks import check_llm_service, check_openai

logger = logging.getLogger(__name__)


class LocalVoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Sei un assistente vocale. Rispondi in italiano, in modo chiaro e sintetico. "
                "Se l'utente chiede aiuto pratico, proponi passi concreti e mantieni uno stile colloquiale."
            ),
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    config = AppConfig.load()
    configure_logging(config.log_level)

    langfuse = configure_langfuse_tracing(agent_name="local-console-agent")
    trace_name = f"console-voice-session-{ctx.room.name}"
    root_input = {
        "mode": "console",
        "room": ctx.room.name,
        "assistant_language": config.assistant_language,
        "openai_stt_model": config.openai_stt_model,
        "openai_tts_model": config.openai_tts_model,
        "llm_model": config.llm_service_model,
    }
    root_context = start_voice_trace(
        langfuse,
        trace_name=trace_name,
        session_id=ctx.room.name,
        call_kind="console",
        agent_name="local-console-agent",
        user_id=None,
        input_payload=root_input,
    )
    root_context_manager = root_context if root_context is not None else nullcontext()
    attr_context = (
        propagate_voice_attributes(session_id=ctx.room.name, trace_name=trace_name, user_id=None)
        if langfuse is not None
        else nullcontext()
    )

    close_event = asyncio.Event()
    close_reason = "unknown"

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

    try:
        with root_context_manager as root_span, attr_context:
            recorder = VoiceTraceRecorder(
                enabled=root_span is not None,
                root_span=root_span,
                trace_name=trace_name,
                session_id=ctx.room.name,
                call_kind="console",
                agent_name="local-console-agent",
                user_id=None,
            )

            @session.on("close")
            def _on_session_closed(event) -> None:
                nonlocal close_reason
                close_reason = str(getattr(event, "reason", "unknown"))
                close_event.set()

            @session.on("user_state_changed")
            def _on_user_state_changed(event) -> None:
                recorder.on_user_state_changed(event)

            @session.on("user_input_transcribed")
            def _on_user_input_transcribed(event) -> None:
                recorder.on_user_input_transcribed(event)

            @session.on("speech_created")
            def _on_speech_created(event) -> None:
                recorder.on_speech_created(event)

            async def _attach_tts_start_logger() -> None:
                while session.output.audio is None:
                    await asyncio.sleep(0.05)

                @session.output.audio.on("playback_started")
                def _on_playback_started(event) -> None:
                    recorder.on_playback_started(event)

            asyncio.create_task(_attach_tts_start_logger())

            await session.start(agent=LocalVoiceAssistant(), room=ctx.room)
            session.say("Ciao, come posso aiutarti?")

            await close_event.wait()
            recorder.finalize(output={"close_reason": close_reason})
    finally:
        shutdown_langfuse(langfuse)


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
