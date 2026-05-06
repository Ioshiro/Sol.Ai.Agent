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
    llm_completed_at: float | None = None
    llm_metrics_duration_ms: float | None = None
    speech_created_at: float | None = None
    playback_started_at: float | None = None
    assistant_committed_at: float | None = None
    user_transcript: str | None = None
    user_language: str | None = None
    assistant_text: str | None = None

    def stt_duration_ms(self) -> float | None:
        if self.stt_started_at is None or self.transcript_final_at is None:
            return None
        return max(0.0, (self.transcript_final_at - self.stt_started_at) * 1000.0)

    def llm_duration_ms(self) -> float | None:
        if self.llm_metrics_duration_ms is not None:
            return self.llm_metrics_duration_ms
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
            "stt_duration_ms": self.stt_duration_ms(),
            "llm_duration_ms": self.llm_duration_ms(),
            "tts_duration_ms": self.tts_duration_ms(),
            "user_transcript": self.user_transcript,
            "user_language": self.user_language,
            "assistant_text": self.assistant_text,
            "assistant_committed_at": self.assistant_committed_at,
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
        stt_model: str = "gpt-4o-mini-transcribe",
        tts_model: str = "gpt-4o-mini-tts",
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.enabled = enabled and root_span is not None
        self.root_span = root_span
        self.trace_name = trace_name
        self.session_id = session_id
        self.call_kind = call_kind
        self.agent_name = agent_name
        self.user_id = user_id
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.llm_model = llm_model
        self.turn_index = 0
        self.current_turn: CallTurnMetrics | None = None
        self._turn_summaries: list[dict[str, Any]] = []
        self._turn_obs: Any | None = None
        self._stt_obs: Any | None = None
        self._turn_by_index: dict[int, CallTurnMetrics] = {}
        self._assistant_turn_index: int | None = None
        self._llm_obs_by_turn: dict[int, Any] = {}
        self._tts_obs_by_turn: dict[int, Any] = {}

    def _event_value(self, event: Any, *names: str) -> Any:
        for name in names:
            if hasattr(event, name):
                return getattr(event, name)
        return None
    def _message_text(self, message: Any) -> str | None:
        text_content = self._event_value(message, "text_content", "textContent")
        if text_content is not None:
            return str(text_content)

        content = self._event_value(message, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                part_text = self._event_value(part, "text", "content", "value")
                if part_text is None and isinstance(part, str):
                    part_text = part
                if part_text is not None:
                    parts.append(str(part_text))
            if parts:
                return "\n".join(parts)

        return None

    def _turn_for_index(self, turn_index: int | None) -> CallTurnMetrics | None:
        if turn_index is None:
            return None
        return self._turn_by_index.get(turn_index)

    def _turn_for_assistant_event(self, created_at: float | None = None) -> CallTurnMetrics | None:
        if created_at is not None:
            candidate: CallTurnMetrics | None = None
            for turn_index in sorted(self._turn_by_index):
                turn = self._turn_by_index[turn_index]
                if turn.speech_created_at is None:
                    continue
                if turn.speech_created_at <= created_at:
                    candidate = turn
            if candidate is not None:
                return candidate

        return self._turn_for_index(self._assistant_turn_index) or self.current_turn

    def _append_turn_summary(self, turn: CallTurnMetrics | None = None) -> None:
        target_turn = turn if turn is not None else self.current_turn
        if target_turn is None:
            return

        summary = target_turn.as_summary()
        self._turn_summaries = [
            t for t in self._turn_summaries if t.get("turn_index") != summary["turn_index"]
        ]
        self._turn_summaries.append(summary)
        self._turn_summaries.sort(key=lambda t: int(t.get("turn_index", 0)))

    def _close_obs(self, obs: Any | None, **update_kwargs: Any) -> Any | None:
        if obs is None:
            return None
        if update_kwargs:
            obs.update(**update_kwargs)
        obs.end()
        return None

    def _close_obs_at(self, obs: Any | None, *, end_at_seconds: float | None, **update_kwargs: Any) -> Any | None:
        if obs is None:
            return None
        if update_kwargs:
            obs.update(**update_kwargs)
        if end_at_seconds is None:
            obs.end()
        else:
            obs.end(end_time=int(end_at_seconds * 1_000_000_000))
        return None

    def _close_turn_obs(
        self,
        obs_by_turn: dict[int, Any],
        turn: CallTurnMetrics | None,
        *,
        end_at_seconds: float | None,
        **update_kwargs: Any,
    ) -> Any | None:
        if turn is None:
            return None

        obs = obs_by_turn.pop(turn.turn_index, None)
        if obs is None:
            return None
        return self._close_obs_at(obs, end_at_seconds=end_at_seconds, **update_kwargs)

    def _maybe_finalize_llm_turn(self, turn: CallTurnMetrics | None = None) -> None:
        target_turn = turn if turn is not None else self.current_turn
        if target_turn is None or target_turn.llm_completed_at is None:
            return
        if target_turn.assistant_text is None:
            return

        self._close_turn_obs(
            self._llm_obs_by_turn,
            target_turn,
            end_at_seconds=target_turn.llm_completed_at,
            input={
                "turn_index": target_turn.turn_index,
                "transcript": target_turn.user_transcript,
                "language": target_turn.user_language,
            },
            output={
                "turn_index": target_turn.turn_index,
                "assistant_text": target_turn.assistant_text,
                "duration_ms": target_turn.llm_duration_ms(),
            },
        )

    def _maybe_finalize_tts_turn(self, turn: CallTurnMetrics | None = None) -> None:
        target_turn = turn if turn is not None else self.current_turn
        if target_turn is None or target_turn.playback_started_at is None:
            return
        if target_turn.assistant_text is None:
            return

        self._close_turn_obs(
            self._tts_obs_by_turn,
            target_turn,
            end_at_seconds=target_turn.playback_started_at,
            input={
                "turn_index": target_turn.turn_index,
                "assistant_text": target_turn.assistant_text,
            },
            output={
                "turn_index": target_turn.turn_index,
                "duration_ms": target_turn.tts_duration_ms(),
                "playback_started_at": target_turn.playback_started_at,
            },
        )

    def _close_current_turn(self) -> None:
        if self.current_turn is not None:
            self._maybe_finalize_llm_turn(self.current_turn)
            self._maybe_finalize_tts_turn(self.current_turn)

        self._stt_obs = self._close_obs(self._stt_obs)
        if self.current_turn is not None and self._turn_obs is not None:
            self._turn_obs.update(
                output={
                    "turn_index": self.current_turn.turn_index,
                    "stt_duration_ms": self.current_turn.stt_duration_ms(),
                    "llm_duration_ms": self.current_turn.llm_duration_ms(),
                    "tts_duration_ms": self.current_turn.tts_duration_ms(),
                    "user_transcript": self.current_turn.user_transcript,
                    "assistant_text": self.current_turn.assistant_text,
                    "assistant_committed_at": self.current_turn.assistant_committed_at,
                }
            )
            self._turn_obs.end()
            self._turn_obs = None

    def _finalize_remaining_turn_obs(self) -> None:
        for turn_index in sorted(list(self._llm_obs_by_turn)):
            turn = self._turn_for_index(turn_index)
            if turn is None:
                self._llm_obs_by_turn.pop(turn_index, None)
                continue
            self._close_turn_obs(
                self._llm_obs_by_turn,
                turn,
                end_at_seconds=turn.llm_completed_at,
                input={
                    "turn_index": turn.turn_index,
                    "transcript": turn.user_transcript,
                    "language": turn.user_language,
                },
                output={
                    "turn_index": turn.turn_index,
                    "assistant_text": turn.assistant_text,
                    "duration_ms": turn.llm_duration_ms(),
                },
            )

        for turn_index in sorted(list(self._tts_obs_by_turn)):
            turn = self._turn_for_index(turn_index)
            if turn is None:
                self._tts_obs_by_turn.pop(turn_index, None)
                continue
            self._close_turn_obs(
                self._tts_obs_by_turn,
                turn,
                end_at_seconds=turn.playback_started_at,
                input={
                    "turn_index": turn.turn_index,
                    "assistant_text": turn.assistant_text,
                },
                output={
                    "turn_index": turn.turn_index,
                    "duration_ms": turn.tts_duration_ms(),
                    "playback_started_at": turn.playback_started_at,
                },
            )
    def begin_turn(self, *, started_at: float, source: str) -> None:
        if not self.enabled:
            return

        self._close_current_turn()
        if self.current_turn is not None:
            self._append_turn_summary()

        self.turn_index += 1
        self.current_turn = CallTurnMetrics(
            turn_index=self.turn_index,
            stt_started_at=started_at,
        )
        self._turn_by_index[self.turn_index] = self.current_turn

        if self.root_span is not None:
            self._turn_obs = self.root_span.start_observation(
                name=f"voice.turn.{self.turn_index}",
                as_type="span",
                input={
                    "turn_index": self.turn_index,
                    "source": source,
                    "started_at": started_at,
                },
                metadata={
                    "turn_index": self.turn_index,
                    "source": source,
                },
            )
            self._stt_obs = self._turn_obs.start_observation(
                name=f"stt.transcribe.turn.{self.turn_index}",
                as_type="generation",
                model=self.stt_model,
                input={
                    "turn_index": self.turn_index,
                    "source": source,
                    "started_at": started_at,
                },
                metadata={
                    "turn_index": self.turn_index,
                    "phase": "stt",
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

        duration_ms = self.current_turn.stt_duration_ms()

        self._stt_obs = self._close_obs(
            self._stt_obs,
            output={
                "turn_index": self.current_turn.turn_index,
                "transcript": transcript,
                "language": language,
                "duration_ms": duration_ms,
            },
        )
        if self._turn_obs is not None:
            self._llm_obs_by_turn[self.current_turn.turn_index] = self._turn_obs.start_observation(
                name=f"llm.generate.turn.{self.current_turn.turn_index}",
                as_type="generation",
                model=self.llm_model,
                input={
                    "turn_index": self.current_turn.turn_index,
                    "transcript": transcript,
                    "language": language,
                },
                metadata={
                    "turn_index": self.current_turn.turn_index,
                    "phase": "llm",
                },
            )

        logger.info(
            "Langfuse turn %s STT final %.1fms transcript='%s'",
            self.current_turn.turn_index,
            duration_ms,
            transcript,
        )

    def on_llm_metrics_collected(self, event: Any) -> None:
        if not self.enabled or self.current_turn is None:
            return

        duration = self._event_value(event, "duration")
        timestamp = self._event_value(event, "timestamp")
        if duration is not None:
            self.current_turn.llm_metrics_duration_ms = float(duration) * 1000.0
        if timestamp is not None:
            self.current_turn.llm_completed_at = float(timestamp)

        self._maybe_finalize_llm_turn()

        logger.info(
            "Langfuse turn %s LLM metrics duration=%.1fms",
            self.current_turn.turn_index,
            self.current_turn.llm_metrics_duration_ms if self.current_turn.llm_metrics_duration_ms is not None else -1.0,
        )
    def on_conversation_item_added(self, event: Any) -> None:
        if not self.enabled:
            return

        item = self._event_value(event, "item")
        if item is None:
            return

        role = str(self._event_value(item, "role") or "").lower()
        if role not in {"assistant", "agent"}:
            return

        created_at = self._event_value(event, "created_at", "createdAt")
        target_turn = self._turn_for_assistant_event(created_at)
        if target_turn is None:
            return

        assistant_text = self._message_text(item)
        if created_at is not None:
            target_turn.assistant_committed_at = float(created_at)

        if assistant_text is not None:
            target_turn.assistant_text = assistant_text

        self._maybe_finalize_llm_turn(target_turn)
        self._maybe_finalize_tts_turn(target_turn)
        self._append_turn_summary(target_turn)

    def on_speech_created(self, event: Any) -> None:
        if not self.enabled or self.current_turn is None:
            return

        self._assistant_turn_index = self.current_turn.turn_index

        created_at = self._event_value(event, "created_at", "createdAt")
        if created_at is not None:
            self.current_turn.speech_created_at = float(created_at)

        assistant_text = self._event_value(
            event, "text", "content", "message", "utterance", "speech_text"
        )
        if assistant_text is not None:
            self.current_turn.assistant_text = str(assistant_text)

        if self._turn_obs is not None:
            self._tts_obs_by_turn[self.current_turn.turn_index] = self._turn_obs.start_observation(
                name=f"tts.synthesize.turn.{self.current_turn.turn_index}",
                as_type="generation",
                model=self.tts_model,
                input={
                    "turn_index": self.current_turn.turn_index,
                    "assistant_text": self.current_turn.assistant_text,
                },
                metadata={
                    "turn_index": self.current_turn.turn_index,
                    "phase": "tts",
                },
            )

        logger.info(
            "Langfuse turn %s TTS start at %.3f",
            self.current_turn.turn_index,
            float(created_at) if created_at is not None else time.perf_counter(),
        )

    def on_playback_started(self, event: Any) -> None:
        if not self.enabled:
            return

        created_at = self._event_value(event, "created_at", "createdAt")
        target_turn = self._turn_for_assistant_event(created_at)
        if target_turn is None:
            return

        if created_at is not None:
            target_turn.playback_started_at = float(created_at)
        self._maybe_finalize_tts_turn(target_turn)

        logger.info(
            "Langfuse turn %s TTS playback at %.3f",
            target_turn.turn_index,
            float(created_at) if created_at is not None else time.perf_counter(),
        )

        self._append_turn_summary(target_turn)

    def finalize(self, *, output: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        self._close_current_turn()
        self._finalize_remaining_turn_obs()
        if self.current_turn is not None:
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
    return propagate_attributes(**attributes, as_baggage=True)


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
