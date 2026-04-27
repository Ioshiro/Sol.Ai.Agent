from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from langfuse import Langfuse, get_client, propagate_attributes
from openinference.instrumentation.openai import OpenAIInstrumentor

logger = logging.getLogger(__name__)

_LANGFUSE_INSTRUMENTED = False


@dataclass(slots=True)
class CallTurnMetrics:
    turn_index: int
    stt_started_at: float | None = None
    transcript_final_at: float | None = None
    speech_created_at: float | None = None
    playback_started_at: float | None = None
    user_transcript: str | None = None
    assistant_speech_source: str | None = None
    assistant_user_initiated: bool | None = None

    def stt_duration_ms(self) -> float | None:
        if self.stt_started_at is None or self.transcript_final_at is None:
            return None
        return max(0.0, (self.transcript_final_at - self.stt_started_at) * 1000.0)

    def llm_duration_ms(self) -> float | None:
        if self.transcript_final_at is None or self.speech_created_at is None:
            return None
        return max(0.0, (self.speech_created_at - self.transcript_final_at) * 1000.0)

    def tts_duration_ms(self) -> float | None:
        if self.speech_created_at is None or self.playback_started_at is None:
            return None
        return max(0.0, (self.playback_started_at - self.speech_created_at) * 1000.0)


class VoiceTraceRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        root_span: Any | None,
        trace_name: str,
        session_id: str,
        call_kind: str,
        agent_name: str,
        user_id: str | None,
    ) -> None:
        self.enabled = enabled and root_span is not None
        self.root_span = root_span
        self.trace_name = trace_name
        self.session_id = session_id
        self.call_kind = call_kind
        self.agent_name = agent_name
        self.user_id = user_id
        self.turn_index = 0
        self.current_turn: CallTurnMetrics | None = None
        self._last_summary: dict[str, Any] = {}

    def _event_value(self, event: Any, *names: str) -> Any:
        for name in names:
            if hasattr(event, name):
                return getattr(event, name)
        return None

    def _update_root_metadata(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        self._last_summary.update({k: v for k, v in metadata.items() if v is not None})
        self.root_span.update(metadata=self._last_summary.copy())

    def begin_turn(self, *, started_at: float, source: str) -> None:
        if not self.enabled:
            return

        self.turn_index += 1
        self.current_turn = CallTurnMetrics(turn_index=self.turn_index, stt_started_at=started_at)
        self._update_root_metadata(
            turn_index=self.turn_index,
            active_turn_source=source,
            active_turn_started_at=started_at,
        )

    def on_user_state_changed(self, event: Any) -> None:
        if not self.enabled:
            return

        old_state = self._event_value(event, "old_state", "oldState")
        new_state = self._event_value(event, "new_state", "newState")
        created_at = self._event_value(event, "created_at", "createdAt")

        if old_state == "speaking" and new_state == "listening" and created_at is not None:
            self.begin_turn(started_at=float(created_at), source="user_state_changed")
            logger.info("Langfuse turn %s STT start at %.3f", self.turn_index, float(created_at))

    def on_user_input_transcribed(self, event: Any) -> None:
        if not self.enabled or self.current_turn is None:
            return

        is_final = bool(self._event_value(event, "is_final", "isFinal"))
        if not is_final:
            return

        transcript = self._event_value(event, "transcript") or ""
        language = self._event_value(event, "language")
        created_at = self._event_value(event, "created_at", "createdAt")
        if created_at is not None:
            self.current_turn.transcript_final_at = float(created_at)
        self.current_turn.user_transcript = transcript

        self._update_root_metadata(
            last_transcript=transcript,
            last_transcript_language=language,
            last_stt_duration_ms=self.current_turn.stt_duration_ms(),
        )
        logger.info(
            "Langfuse turn %s STT final at %.3f transcript='%s'",
            self.turn_index,
            float(created_at) if created_at is not None else time.perf_counter(),
            transcript,
        )

    def on_speech_created(self, event: Any) -> None:
        if not self.enabled or self.current_turn is None:
            return

        created_at = self._event_value(event, "created_at", "createdAt")
        if created_at is not None:
            self.current_turn.speech_created_at = float(created_at)

        speech_source = self._event_value(event, "source")
        user_initiated = self._event_value(event, "user_initiated", "userInitiated")
        self.current_turn.assistant_speech_source = str(speech_source) if speech_source is not None else None
        self.current_turn.assistant_user_initiated = bool(user_initiated) if user_initiated is not None else None

        self._update_root_metadata(
            last_llm_duration_ms=self.current_turn.llm_duration_ms(),
            assistant_speech_source=self.current_turn.assistant_speech_source,
            assistant_user_initiated=self.current_turn.assistant_user_initiated,
        )
        logger.info(
            "Langfuse turn %s LLM completed at %.3f (speech created)",
            self.turn_index,
            float(created_at) if created_at is not None else time.perf_counter(),
        )

    def on_playback_started(self, event: Any) -> None:
        if not self.enabled or self.current_turn is None:
            return

        created_at = self._event_value(event, "created_at", "createdAt")
        if created_at is not None:
            self.current_turn.playback_started_at = float(created_at)

        self._update_root_metadata(
            last_tts_duration_ms=self.current_turn.tts_duration_ms(),
            last_turn_index=self.turn_index,
        )
        logger.info(
            "Langfuse turn %s TTS playback started at %.3f",
            self.turn_index,
            float(created_at) if created_at is not None else time.perf_counter(),
        )

    def finalize(self, *, output: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        summary: dict[str, Any] = {
            "turn_count": self.turn_index,
            "call_kind": self.call_kind,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "user_id": self.user_id,
        }
        if output:
            summary.update(output)
        if self._last_summary:
            summary.update(self._last_summary)
        self.root_span.update(output=summary)



def configure_langfuse_tracing(*, agent_name: str) -> Langfuse | None:
    global _LANGFUSE_INSTRUMENTED

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        logger.info("Langfuse tracing disabled: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are missing")
        return None

    client = get_client()

    if not _LANGFUSE_INSTRUMENTED:
        OpenAIInstrumentor().instrument()
        _LANGFUSE_INSTRUMENTED = True
        logger.info("OpenAI SDK instrumentation enabled for Langfuse (%s)", agent_name)

    return client



def start_voice_trace(
    client: Langfuse | None,
    *,
    trace_name: str,
    session_id: str,
    call_kind: str,
    agent_name: str,
    user_id: str | None,
    input_payload: dict[str, Any],
) -> Any:
    if client is None:
        return None

    return client.start_as_current_observation(
        as_type="span",
        name=trace_name,
        input=input_payload,
    )



def propagate_voice_attributes(*, session_id: str, trace_name: str, user_id: str | None):
    attributes: dict[str, str] = {
        "session_id": session_id,
        "trace_name": trace_name,
    }
    if user_id:
        attributes["user_id"] = user_id
    return propagate_attributes(**attributes)



def shutdown_langfuse(client: Langfuse | None) -> None:
    if client is None:
        return

    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown()
    else:
        flush = getattr(client, "flush", None)
        if callable(flush):
            flush()
