from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents.voice.room_io import RoomOptions
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from app.config import AppConfig
from app.logging_utils import configure_logging
from app.runtime_checks import check_llm_service, check_openai

logger = logging.getLogger(__name__)

SIP_PARTICIPANT_KINDS = [rtc.ParticipantKind.PARTICIPANT_KIND_SIP]


@dataclass
class TurnLatencyTracker:
    user_stopped_speaking_at: float | None = None
    user_final_transcript_at: float | None = None
    awaiting_tts_start: bool = False


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


class LocalSipAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Sei un assistente vocale telefonico. Rispondi in italiano in modo chiaro, breve e naturale. "
                "Considera che l'utente e' in chiamata, quindi evita risposte troppo lunghe e conferma spesso i passaggi importanti."
            )
        )


async def entrypoint(ctx: JobContext) -> None:
    config = AppConfig.load()
    configure_logging(config.log_level)
    await asyncio.gather(check_openai(config), check_llm_service(config))
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    caller = await _ensure_sip_participant(ctx)
    logger.info(
        "SIP caller connected: identity=%s name=%s kind=%s",
        caller.identity,
        caller.name,
        rtc.ParticipantKind.Name(caller.kind),
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
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

    _install_latency_logging(session)

    await session.start(
        agent=LocalSipAssistant(),
        room=ctx.room,
        room_options=RoomOptions(
            participant_kinds=SIP_PARTICIPANT_KINDS,
            close_on_disconnect=True,
        ),
    )
    session.say("Buongiorno, come posso aiutarti?")


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


def _install_latency_logging(session: AgentSession) -> None:
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
                    "SIP latency user-stop -> TTS-start: %.3fs%s",
                    stopped_delta,
                    (
                        f" (final transcript -> TTS-start: {transcript_delta:.3f}s)"
                        if transcript_delta is not None
                        else ""
                    ),
                )

    asyncio.create_task(_attach_tts_start_logger())


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", "local-sip-agent"),
            ws_url="ws://localhost:7880",
            api_key="devkey",
            api_secret="secret",
        )
    )
