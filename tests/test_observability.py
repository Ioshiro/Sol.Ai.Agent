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



if __name__ == "__main__":
    unittest.main()
