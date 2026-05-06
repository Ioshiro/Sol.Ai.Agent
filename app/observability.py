from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from langfuse import Langfuse, get_client, propagate_attributes

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CallTurnMetrics:
    turn_index: int
    stt_started_at: float | None = None
    transcript_final_at: float | None = None
    speech_created_at: float | None = None
    playback_started_at: float | None = None
    user_transcript: str | None = None
    user_language: str | None = None
    assistant_speech_source: str | None = None
    assistant_text: str | None = None
    assistant_user_initiated: bool | None = None
    turn_cm: Any | None = None
    turn_span: Any | None = None
    stt_cm: Any | None = None
    stt_span: Any | None = None
    llm_cm: Any | None = None
    llm_span: Any | None = None
    tts_cm: Any | None = None
    tts_span: Any | None = None

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

    def as_summary(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "stt_started_at": self.stt_started_at,
            "transcript_final_at": self.transcript_final_at,
            "speech_created_at": self.speech_created_at,
            "playback_started_at": self.playback_started_at,
            "user_transcript": self.user_transcript,
            "user_language": self.user_language,
            "assistant_speech_source": self.assistant_speech_source,
            "assistant_text": self.assistant_text,
            "assistant_user_initiated": self.assistant_user_initiated,
            "stt_duration_ms": self.stt_duration_ms(),
            "llm_duration_ms": self.llm_duration_ms(),
            "tts_duration_ms": self.tts_duration_ms(),
        }


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
        self._turn_summaries: list[dict[str, Any]] = []

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

    def _open_observation(
        self,
        parent: Any | None,
        *,
        name: str,
        as_type: str = "span",
        input_payload: dict[str, Any] | None = None,
    ) -> tuple[Any | None, Any | None]:
        if not self.enabled or parent is None:
            return None, None

        cm = parent.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input_payload or {},
        )
        span = cm.__enter__()
        return cm, span

    def _close_observation(self, cm: Any | None, span: Any | None, *, output: dict[str, Any] | None = None) -> None:
        if cm is None or span is None:
            return
        if output is not None:
            span.update(output=output)
        cm.__exit__(None, None, None)

    def _close_turn_observations(self) -> None:
        if self.current_turn is None:
            return

        self._close_observation(self.current_turn.tts_cm, self.current_turn.tts_span)
        self._close_observation(self.current_turn.llm_cm, self.current_turn.llm_span)
        self._close_observation(self.current_turn.stt_cm, self.current_turn.stt_span)
        self._close_observation(self.current_turn.turn_cm, self.current_turn.turn_span)

    def begin_turn(self, *, started_at: float, source: str) -> None:
        if not self.enabled:
            return

        if self.current_turn is not None:
            self._close_turn_observations()
            self._append_turn_summary()

        self.turn_index += 1
        self.current_turn = CallTurnMetrics(turn_index=self.turn_index, stt_started_at=started_at)
        self._update_root_metadata(
            turn_index=self.turn_index,
            active_turn_source=source,
            active_turn_started_at=started_at,
        )

        turn_name = f"voice.turn.{self.turn_index}"
        turn_input = {
            "turn_index": self.turn_index,
            "source": source,
            "started_at": started_at,
            "call_kind": self.call_kind,
        }
        self.current_turn.turn_cm, self.current_turn.turn_span = self._open_observation(
            self.root_span,
            name=turn_name,
            as_type="span",
            input_payload=turn_input,
        )
        self.current_turn.stt_cm, self.current_turn.stt_span = self._open_observation(
            self.current_turn.turn_span or self.root_span,
            name=f"stt.transcribe.turn.{self.turn_index}",
            as_type="generation",
            input_payload={
                "turn_index": self.turn_index,
                "source": source,
                "started_at": started_at,
            },
        )
        logger.info("Langfuse turn %s STT start at %.3f", self.turn_index, started_at)

    def on_user_state_changed(self, event: Any) -> None:
        if not self.enabled:
            return

        old_state = self._event_value(event, "old_state", "oldState")
        new_state = self._event_value(event, "new_state", "newState")
        created_at = self._event_value(event, "created_at", "createdAt")

        if old_state == "speaking" and new_state == "listening" and created_at is not None:
            self.begin_turn(started_at=float(created_at), source="user_state_changed")

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
        self.current_turn.user_language = str(language) if language is not None else None

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

        self._close_observation(
            self.current_turn.stt_cm,
            self.current_turn.stt_span,
            output={
                "turn_index": self.turn_index,
                "transcript": transcript,
                "language": language,
                "duration_ms": self.current_turn.stt_duration_ms(),
            },
        )
        self.current_turn.stt_cm = None
        self.current_turn.stt_span = None

        llm_input = {
            "turn_index": self.turn_index,
            "transcript": transcript,
            "language": language,
        }
        self.current_turn.llm_cm, self.current_turn.llm_span = self._open_observation(
            self.current_turn.turn_span or self.root_span,
            name=f"llm.generate.turn.{self.turn_index}",
            as_type="generation",
            input_payload=llm_input,
        )

    def on_speech_created(self, event: Any) -> None:
        if not self.enabled or self.current_turn is None:
            return

        created_at = self._event_value(event, "created_at", "createdAt")
        if created_at is not None:
            self.current_turn.speech_created_at = float(created_at)

        speech_source = self._event_value(event, "source")
        user_initiated = self._event_value(event, "user_initiated", "userInitiated")
        assistant_text = self._event_value(event, "text", "content", "message", "utterance", "speech_text")
        self.current_turn.assistant_speech_source = str(speech_source) if speech_source is not None else None
        self.current_turn.assistant_text = str(assistant_text) if assistant_text is not None else None
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

        self._close_observation(
            self.current_turn.llm_cm,
            self.current_turn.llm_span,
            output={
                "turn_index": self.turn_index,
                "assistant_text": self.current_turn.assistant_text,
                "assistant_speech_source": self.current_turn.assistant_speech_source,
                "assistant_user_initiated": self.current_turn.assistant_user_initiated,
                "duration_ms": self.current_turn.llm_duration_ms(),
            },
        )
        self.current_turn.llm_cm = None
        self.current_turn.llm_span = None

        tts_input: dict[str, Any] = {
            "turn_index": self.turn_index,
            "assistant_text": self.current_turn.assistant_text,
            "assistant_speech_source": self.current_turn.assistant_speech_source,
            "assistant_user_initiated": self.current_turn.assistant_user_initiated,
        }
        self.current_turn.tts_cm, self.current_turn.tts_span = self._open_observation(
            self.current_turn.turn_span or self.root_span,
            name=f"tts.synthesize.turn.{self.turn_index}",
            as_type="generation",
            input_payload=tts_input,
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

        self._close_observation(
            self.current_turn.tts_cm,
            self.current_turn.tts_span,
            output={
                "turn_index": self.turn_index,
                "duration_ms": self.current_turn.tts_duration_ms(),
                "playback_started_at": self.current_turn.playback_started_at,
            },
        )
        self.current_turn.tts_cm = None
        self.current_turn.tts_span = None

        self._append_turn_summary()

    def _append_turn_summary(self) -> None:
        if self.current_turn is None:
            return

        summary = self.current_turn.as_summary()
        self._turn_summaries = [
            existing
            for existing in self._turn_summaries
            if existing.get("turn_index") != summary["turn_index"]
        ]
        self._turn_summaries.append(summary)
        self._turn_summaries.sort(key=lambda item: int(item.get("turn_index", 0)))
        self._update_root_metadata(turns=self._turn_summaries.copy())

    def finalize(self, *, output: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        if self.current_turn is not None:
            self._close_turn_observations()
            self._append_turn_summary()

        summary: dict[str, Any] = {
            "turn_count": self.turn_index,
            "call_kind": self.call_kind,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "user_id": self.user_id,
            "turns": self._turn_summaries.copy(),
        }
        if output:
            summary.update(output)
        if self._last_summary:
            summary.update(self._last_summary)
        self.root_span.update(output=summary)


def configure_langfuse_tracing(*, agent_name: str) -> Langfuse | None:
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        logger.info("Langfuse tracing disabled: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are missing")
        return None

    os.environ["OTEL_SERVICE_NAME"] = agent_name
    resource_attributes = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "").strip()
    service_attribute = f"service.name={agent_name}"
    if service_attribute not in resource_attributes:
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (
            f"{resource_attributes},{service_attribute}" if resource_attributes else service_attribute
        )

    client = get_client()
    logger.info("Langfuse tracing enabled for %s", agent_name)
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
        metadata={
            "session_id": session_id,
            "call_kind": call_kind,
            "agent_name": agent_name,
            "user_id": user_id,
        },
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
