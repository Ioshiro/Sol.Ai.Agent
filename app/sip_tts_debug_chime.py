"""Optional short tones on the SIP leg for pipeline debug markers.

Uses LiveKit ``BackgroundAudioPlayer`` so audio is published to the room
(heard on the phone).

Markers (same env enables both):

- **Final STT transcript**: two short beeps (~700 Hz) on ``user_input_transcribed``
  with ``is_final=True`` (testo finale dal riconoscimento).
- **Agent voice playout**: single higher beep (~1.2 kHz) on ``playback_started``
  for ``session.output.audio`` — primo frame verso la room dopo LLM+TTS.

  Nota: **non** usare ``speech_created`` per il TTS: con preemptive generation
  LiveKit emette ``speech_created`` subito dopo il transcript finale, quasi nello
  stesso istante dell'evento STT, non quando parte l'audio sintetizzato.

Enable with env ``SIP_TTS_DEBUG_CHIME=1`` (also ``true`` / ``yes`` / ``on``).
"""

from __future__ import annotations

import math
import os
import struct
from collections.abc import AsyncIterator

from livekit import rtc
from livekit.agents import AudioConfig, BackgroundAudioPlayer

_SAMPLE_RATE = 48_000
_CHUNK_SAMPLES = 480  # 10 ms @ 48 kHz mono
_EDGE_S = 0.012


def sip_tts_debug_chime_enabled() -> bool:
    v = os.getenv("SIP_TTS_DEBUG_CHIME", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _envelope(pos: int, total: int) -> float:
    edge = int(_SAMPLE_RATE * _EDGE_S)
    if pos < edge:
        return pos / max(edge, 1)
    if pos > total - edge - 1:
        return max(0.0, (total - 1 - pos) / max(edge, 1))
    return 1.0


async def _sine_burst_frames(
    *,
    freq_hz: float,
    duration_s: float,
    amplitude: float = 0.22,
) -> AsyncIterator[rtc.AudioFrame]:
    amp = amplitude * 32767.0
    total = max(1, int(_SAMPLE_RATE * duration_s))
    n = 0
    while n < total:
        chunk_len = min(_CHUNK_SAMPLES, total - n)
        samples: list[int] = []
        for i in range(chunk_len):
            pos = n + i
            env = _envelope(pos, total)
            t = pos / _SAMPLE_RATE
            v = int(amp * env * math.sin(2.0 * math.pi * freq_hz * t))
            if v > 32767:
                v = 32767
            elif v < -32768:
                v = -32768
            samples.append(v)
        n += chunk_len
        data = struct.pack(f"<{len(samples)}h", *samples)
        yield rtc.AudioFrame(
            data=data,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=chunk_len,
        )


async def _silence_frames(duration_s: float) -> AsyncIterator[rtc.AudioFrame]:
    remaining = int(_SAMPLE_RATE * duration_s)
    while remaining > 0:
        chunk_len = min(_CHUNK_SAMPLES, remaining)
        data = b"\x00" * (chunk_len * 2)
        yield rtc.AudioFrame(
            data=data,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=chunk_len,
        )
        remaining -= chunk_len


async def _short_tone_frames() -> AsyncIterator[rtc.AudioFrame]:
    """~70 ms @ 1.2 kHz — TTS / ``speech_created``."""
    async for frame in _sine_burst_frames(freq_hz=1200.0, duration_s=0.07):
        yield frame


async def _stt_final_llm_marker_frames() -> AsyncIterator[rtc.AudioFrame]:
    """Two ~50 ms bursts @ 700 Hz with short gap — final transcript → LLM."""
    async for frame in _sine_burst_frames(freq_hz=700.0, duration_s=0.05, amplitude=0.2):
        yield frame
    async for frame in _silence_frames(0.028):
        yield frame
    async for frame in _sine_burst_frames(freq_hz=700.0, duration_s=0.05, amplitude=0.2):
        yield frame


def tts_debug_chime_audio_config() -> AudioConfig:
    """Fresh async iterator per ``play()`` call (TTS phase)."""
    return AudioConfig(_short_tone_frames(), volume=1.0)


def stt_llm_debug_chime_audio_config() -> AudioConfig:
    """Fresh async iterator per ``play()`` call (final STT → LLM)."""
    return AudioConfig(_stt_final_llm_marker_frames(), volume=1.0)


async def start_debug_chime_player(*, room: rtc.Room) -> BackgroundAudioPlayer:
    player = BackgroundAudioPlayer()
    await player.start(room=room)
    return player
