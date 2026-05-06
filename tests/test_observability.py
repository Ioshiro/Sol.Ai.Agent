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

    def end(self):
        self.ended = True


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
        self.assertEqual(llm.updates, [])

        root_output = root.updates[-1]["output"]
        self.assertEqual(root_output["turn_count"], 1)
        self.assertEqual(root_output["turns"][0]["assistant_text"], "Certo, dimmi pure.")
        self.assertAlmostEqual(root_output["turns"][0]["llm_duration_ms"], 100.0)
        self.assertAlmostEqual(root_output["turns"][0]["tts_duration_ms"], 2900.0)
        self.assertEqual(root_output["turns"][0]["assistant_committed_at"], 4.0)
        self.assertEqual(root_output["close_reason"], "done")


if __name__ == "__main__":
    unittest.main()
