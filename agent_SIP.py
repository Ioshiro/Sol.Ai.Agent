from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import nullcontext

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    BackgroundAudioPlayer,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.voice.room_io import RoomOptions
from livekit.plugins import openai, silero

from livekit.agents.tts import StreamAdapter
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
from app.sip_tts_debug_chime import (
    llm_stream_debug_chime_audio_config,
    sip_tts_debug_chime_enabled,
    start_debug_chime_player,
    stt_llm_debug_chime_audio_config,
)

logger = logging.getLogger(__name__)

SIP_PARTICIPANT_KINDS = [rtc.ParticipantKind.PARTICIPANT_KIND_SIP]


class LocalSipAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Sei un assistente vocale telefonico. Rispondi in italiano in modo chiaro, breve e naturale. "
                "Considera che l'utente e' in chiamata, quindi evita risposte troppo lunghe e conferma spesso i passaggi importanti."
            )
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    config = AppConfig.load()
    configure_logging(config.log_level)
    langfuse = configure_langfuse_tracing(agent_name=os.getenv("LIVEKIT_AGENT_NAME", "solai-sip-agent"))

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    caller = await _ensure_sip_participant(ctx)
    logger.info(
        "SIP caller connected: identity=%s name=%s kind=%s",
        caller.identity,
        caller.name,
        rtc.ParticipantKind.Name(caller.kind),
    )

    trace_name = f"sip-voice-session-{ctx.room.name}"
    root_input = {
        "mode": "sip",
        "room": ctx.room.name,
        "assistant_language": config.assistant_language,
        "caller_identity": caller.identity,
        "caller_name": caller.name,
        "openai_stt_model": config.openai_stt_model,
        "openai_tts_model": config.openai_tts_model,
        "llm_model": config.llm_service_model,
    }
    root_context = start_voice_trace(
        langfuse,
        trace_name=trace_name,
        session_id=ctx.room.name,
        call_kind="sip",
        agent_name=os.getenv("LIVEKIT_AGENT_NAME", "solai-sip-agent"),
        user_id=caller.identity,
        input_payload=root_input,
    )
    root_context_manager = root_context if root_context is not None else nullcontext()
    attr_context = (
        propagate_voice_attributes(session_id=ctx.room.name, trace_name=trace_name, user_id=caller.identity)
        if langfuse is not None
        else nullcontext()
    )

    close_event = asyncio.Event()
    close_reason = "unknown"
    sip_debug_chime: BackgroundAudioPlayer | None = None

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(model=config.openai_stt_model, language=config.assistant_language, base_url=config.openai_base_url, api_key=config.openai_api_key),
        llm=openai.LLM(
            model=config.llm_service_model,
            base_url=config.llm_service_base_url,
            api_key=config.llm_service_api_key,
        ),
        tts=StreamAdapter(
            tts=openai.TTS(
                model=config.openai_tts_model,
                voice=config.openai_tts_voice,
                base_url=config.openai_base_url,
                api_key=config.openai_api_key,
                response_format="pcm",
            )
        ),
    )

    try:
        with root_context_manager as root_span, attr_context:
            recorder = VoiceTraceRecorder(
                enabled=root_span is not None,
                root_span=root_span,
                trace_name=trace_name,
                session_id=ctx.room.name,
                call_kind="sip",
                agent_name=os.getenv("LIVEKIT_AGENT_NAME", "solai-sip-agent"),
                user_id=caller.identity,
            )
            thinking_chime_last_at: list[float] = [0.0]  # debounce tool-loop speaking→thinking

            @session.on("close")
            def _on_session_closed(event) -> None:
                nonlocal close_reason
                close_reason = str(getattr(event, "reason", "unknown"))
                close_event.set()

            @session.on("user_state_changed")
            def _on_user_state_changed(event) -> None:
                recorder.on_user_state_changed(event)

            @session.on("agent_state_changed")
            def _on_agent_state_changed(event) -> None:
                if sip_debug_chime is None:
                    return
                old_state = getattr(event, "old_state", None) or getattr(event, "oldState", None)
                new_state = getattr(event, "new_state", None) or getattr(event, "newState", None)
                if new_state != "thinking":
                    return
                if old_state == "initializing":
                    return
                now = time.monotonic()
                if now - thinking_chime_last_at[0] < 0.18:
                    return
                thinking_chime_last_at[0] = now
                sip_debug_chime.play(llm_stream_debug_chime_audio_config())

            @session.on("user_input_transcribed")
            def _on_user_input_transcribed(event) -> None:
                recorder.on_user_input_transcribed(event)
                if sip_debug_chime is None:
                    return
                is_final = bool(getattr(event, "is_final", False) or getattr(event, "isFinal", False))
                if is_final:
                    sip_debug_chime.play(stt_llm_debug_chime_audio_config())

            @session.on("speech_created")
            def _on_speech_created(event) -> None:
                recorder.on_speech_created(event)

            @session.on("conversation_item_added")
            def _on_conversation_item_added(event) -> None:
                recorder.on_conversation_item_added(event)

            @session.llm.on("metrics_collected")
            def _on_llm_metrics_collected(metrics) -> None:
                recorder.on_llm_metrics_collected(metrics)
            async def _attach_tts_start_logger() -> None:
                while session.output.audio is None:
                    await asyncio.sleep(0.05)

                @session.output.audio.on("playback_started")
                def _on_playback_started(event) -> None:
                    recorder.on_playback_started(event)

            asyncio.create_task(_attach_tts_start_logger())

            if sip_tts_debug_chime_enabled():
                sip_debug_chime = await start_debug_chime_player(room=ctx.room)
                logger.info(
                    "SIP voice debug chimes → phone: double beep on final STT; "
                    "high tone when agent enters thinking (LLM reply pipeline / text stream forward)"
                )

            await session.start(
                agent=LocalSipAssistant(),
                room=ctx.room,
                room_options=RoomOptions(
                    participant_kinds=SIP_PARTICIPANT_KINDS,
                    close_on_disconnect=True,
                ),
            )
            session.say("Buongiorno, come posso aiutarti?")

            await close_event.wait()
            recorder.finalize(output={"close_reason": close_reason})
    finally:
        if sip_debug_chime is not None:
            await sip_debug_chime.aclose()
        shutdown_langfuse(langfuse)


async def _ensure_sip_participant(ctx: JobContext) -> rtc.RemoteParticipant:
    outbound_target = os.getenv("SIP_OUTBOUND_TARGET", "").strip()
    outbound_trunk = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK", "").strip()

    if outbound_target:
        if not outbound_trunk:
            raise ValueError(
                "SIP_OUTBOUND_TARGET is set but LIVEKIT_SIP_OUTBOUND_TRUNK is missing."
            )

        participant_identity = os.getenv("SIP_OUTBOUND_PARTICIPANT_IDENTITY", "asterisk-caller")
        logger.info(
            "Dialing outbound SIP target '%s' using LiveKit trunk '%s'",
            outbound_target,
            outbound_trunk,
        )
        await ctx.add_sip_participant(
            call_to=outbound_target,
            trunk_id=outbound_trunk,
            participant_identity=participant_identity,
            participant_name="Asterisk SIP caller",
        )
        return await ctx.wait_for_participant(
            identity=participant_identity,
            kind=SIP_PARTICIPANT_KINDS,
        )

    logger.info("Waiting for inbound SIP participant from Asterisk/LiveKit SIP")
    return await ctx.wait_for_participant(kind=SIP_PARTICIPANT_KINDS)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", "solai-sip-agent"),
            ws_url="ws://localhost:7880",
            api_key="devkey",
            api_secret="secret",
        )
    )
