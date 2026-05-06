from __future__ import annotations

import contextlib
import sys
import types
import unittest
from types import SimpleNamespace


if "langfuse" not in sys.modules:
    fake_langfuse = types.ModuleType("langfuse")

    class _DummyLangfuse:
        pass

    def _get_client() -> _DummyLangfuse:
        return _DummyLangfuse()

    def _propagate_attributes(**_kwargs):
        return contextlib.nullcontext()

    fake_langfuse.Langfuse = _DummyLangfuse
    fake_langfuse.get_client = _get_client
    fake_langfuse.propagate_attributes = _propagate_attributes
    sys.modules["langfuse"] = fake_langfuse

from app.observability import VoiceTraceRecorder


class FakeObservation:
    def __init__(self, *, name: str, as_type: str | None = None, model: str | None = None, input=None, metadata=None):
        self.name = name
        self.as_type = as_type
        self.model = model
        self.input = input
        self.metadata = metadata
        self.updates: list[dict[str, object]] = []
        self.children: list[FakeObservation] = []
        self.ended = False

    def start_observation(self, *, name: str, as_type: str | None = None, model: str | None = None, input=None, metadata=None):
        child = FakeObservation(name=name, as_type=as_type, model=model, input=input, metadata=metadata)
        self.children.append(child)
        return child

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self, *, end_time=None):
        self.ended = True
        self.end_time = end_time


class VoiceTraceRecorderTests(unittest.TestCase):
    def test_turn_trace_captures_llm_and_final_assistant_text(self) -> None:
        root = FakeObservation(name="root", as_type="span")
        recorder = VoiceTraceRecorder(
            enabled=True,
            root_span=root,
            trace_name="trace",
            session_id="session-1",
            call_kind="sip",
            agent_name="agent",
            user_id="user-1",
        )

        recorder.begin_turn(started_at=1.0, source="user_state_changed")
        recorder.on_user_input_transcribed(
            SimpleNamespace(is_final=True, transcript="Pronto", language="it", created_at=2.0)
        )
        recorder.on_session_metrics_collected(
            SimpleNamespace(
                metrics=SimpleNamespace(
                    type="eou_metrics",
                    timestamp=2.05,
                    end_of_utterance_delay=0.05,
                    transcription_delay=0.1,
                    on_user_turn_completed_delay=0.02,
                    speech_id="sid-1",
                )
            )
        )
        recorder.on_llm_metrics_collected(
            SimpleNamespace(duration=1.5, timestamp=3.0, ttft=1.1, request_id="r1")
        )
        recorder.on_speech_created(SimpleNamespace(created_at=2.1))
        recorder.on_conversation_item_added(
            SimpleNamespace(
                item=SimpleNamespace(role="assistant", text_content="Certo, dimmi pure.", content=["Certo, dimmi pure."]),
                created_at=4.0,
            )
        )
        recorder.on_playback_started(SimpleNamespace(created_at=5.0))
        recorder.finalize(output={"close_reason": "done"})

        self.assertEqual(len(root.children), 1)
        turn = root.children[0]
        self.assertEqual(turn.name, "voice.turn.1")
        self.assertTrue(turn.ended)

        child_names = [child.name for child in turn.children]
        self.assertIn("stt.transcribe.turn.1", child_names)
        self.assertIn("llm.generate.turn.1", child_names)
        self.assertIn("tts.synthesize.turn.1", child_names)

        llm = next(child for child in turn.children if child.name == "llm.generate.turn.1")
        self.assertTrue(llm.ended)
        self.assertEqual(llm.metadata.get("llm_trace_source"), "eou_metrics")
        self.assertEqual(llm.updates[-1]["output"]["assistant_text"], "Certo, dimmi pure.")
        self.assertAlmostEqual(llm.updates[-1]["output"]["duration_ms"], 1500.0)
        self.assertEqual(llm.end_time, 3000000000)

        tts = next(child for child in turn.children if child.name == "tts.synthesize.turn.1")
        self.assertTrue(tts.ended)
        self.assertEqual(tts.updates[-1]["input"]["assistant_text"], "Certo, dimmi pure.")
        self.assertAlmostEqual(tts.updates[-1]["output"]["duration_ms"], 2900.0)
        self.assertEqual(tts.end_time, 5000000000)

        root_output = root.updates[-1]["output"]
        self.assertEqual(root_output["turn_count"], 1)
        self.assertEqual(root_output["turns"][0]["assistant_text"], "Certo, dimmi pure.")
        self.assertAlmostEqual(root_output["turns"][0]["llm_duration_ms"], 1500.0)
        self.assertAlmostEqual(root_output["turns"][0]["tts_duration_ms"], 2900.0)
        self.assertEqual(root_output["turns"][0]["assistant_committed_at"], 4.0)
        self.assertAlmostEqual(root_output["turns"][0]["eou_transcription_delay_ms"], 100.0)
        self.assertAlmostEqual(root_output["turns"][0]["eou_end_of_utterance_delay_ms"], 50.0)
        self.assertEqual(root_output["close_reason"], "done")

    def test_assistant_commit_is_assigned_to_original_turn(self) -> None:
        root = FakeObservation(name="root", as_type="span")
        recorder = VoiceTraceRecorder(
            enabled=True,
            root_span=root,
            trace_name="trace",
            session_id="session-1",
            call_kind="sip",
            agent_name="agent",
            user_id="user-1",
        )

        recorder.begin_turn(started_at=1.0, source="user_state_changed")
        recorder.on_user_input_transcribed(
            SimpleNamespace(is_final=True, transcript="Pronto", language="it", created_at=2.0)
        )
        recorder.on_llm_metrics_collected(
            SimpleNamespace(duration=1.5, timestamp=3.0, ttft=1.1, request_id="r1")
        )
        recorder.on_speech_created(SimpleNamespace(created_at=2.1))
        recorder.on_playback_started(SimpleNamespace(created_at=2.5))

        recorder.begin_turn(started_at=5.0, source="user_state_changed")
        recorder.on_conversation_item_added(
            SimpleNamespace(
                item=SimpleNamespace(role="assistant", text_content="Certo, dimmi pure.", content=["Certo, dimmi pure."]),
                created_at=6.0,
            )
        )
        recorder.finalize(output={"close_reason": "done"})

        root_output = root.updates[-1]["output"]
        self.assertEqual(root_output["turn_count"], 2)
        self.assertEqual(root_output["turns"][0]["assistant_text"], "Certo, dimmi pure.")
        self.assertEqual(root_output["turns"][0]["assistant_committed_at"], 6.0)
        self.assertEqual(root_output["turns"][1]["assistant_text"], None)


    def test_late_assistant_commit_stays_on_original_turn(self) -> None:
        root = FakeObservation(name="root", as_type="span")
        recorder = VoiceTraceRecorder(
            enabled=True,
            root_span=root,
            trace_name="trace",
            session_id="session-1",
            call_kind="sip",
            agent_name="agent",
            user_id="user-1",
        )

        recorder.begin_turn(started_at=1.0, source="user_state_changed")
        recorder.on_user_input_transcribed(
            SimpleNamespace(is_final=True, transcript="Uno", language="it", created_at=1.1)
        )
        recorder.on_llm_metrics_collected(
            SimpleNamespace(duration=1.0, timestamp=2.0, ttft=0.8, request_id="r1")
        )
        recorder.on_speech_created(SimpleNamespace(created_at=1.5))
        recorder.on_conversation_item_added(
            SimpleNamespace(
                item=SimpleNamespace(role="assistant", text_content="Risposta uno", content=["Risposta uno"]),
                created_at=1.6,
            )
        )
        recorder.on_playback_started(SimpleNamespace(created_at=1.8))

        recorder.begin_turn(started_at=3.0, source="user_state_changed")
        recorder.on_user_input_transcribed(
            SimpleNamespace(is_final=True, transcript="Due", language="it", created_at=3.1)
        )
        recorder.on_llm_metrics_collected(
            SimpleNamespace(duration=1.2, timestamp=4.0, ttft=0.9, request_id="r2")
        )
        recorder.on_speech_created(SimpleNamespace(created_at=3.2))
        recorder.on_playback_started(SimpleNamespace(created_at=3.4))

        recorder.begin_turn(started_at=4.5, source="user_state_changed")
        recorder.on_user_input_transcribed(
            SimpleNamespace(is_final=True, transcript="Tre", language="it", created_at=4.6)
        )
        recorder.on_speech_created(SimpleNamespace(created_at=5.0))
        recorder.on_conversation_item_added(
            SimpleNamespace(
                item=SimpleNamespace(role="assistant", text_content="Risposta due", content=["Risposta due"]),
                created_at=4.2,
            )
        )
        recorder.finalize(output={"close_reason": "done"})

        root_output = root.updates[-1]["output"]
        self.assertEqual(root_output["turn_count"], 3)
        self.assertEqual(root_output["turns"][1]["assistant_text"], "Risposta due")
        self.assertEqual(root_output["turns"][1]["assistant_committed_at"], 4.2)

        turn2 = next(child for child in root.children if child.name == "voice.turn.2")
        llm = next(child for child in turn2.children if child.name == "llm.generate.turn.2")
        self.assertTrue(llm.ended)
        self.assertEqual(llm.updates[-1]["output"]["turn_index"], 2)
        self.assertEqual(llm.updates[-1]["output"]["assistant_text"], "Risposta due")
        self.assertEqual(llm.end_time, 4000000000)

        tts = next(child for child in turn2.children if child.name == "tts.synthesize.turn.2")
        self.assertEqual(tts.updates[-1]["input"]["assistant_text"], "Risposta due")



if __name__ == "__main__":
    unittest.main()
