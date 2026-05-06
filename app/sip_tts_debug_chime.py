"""Optional short tone on the SIP leg when assistant speech (TTS) is created.

Uses LiveKit ``BackgroundAudioPlayer`` so audio is published to the room like
normal agent media (heard on the phone), not the worker machine speaker.

Enable with env ``SIP_TTS_DEBUG_CHIME=1`` (also ``true`` / ``yes``).
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


def sip_tts_debug_chime_enabled() -> bool:
    v = os.getenv("SIP_TTS_DEBUG_CHIME", "").strip().lower()
    return v in ("1", "true", "yes", "on")


async def _short_tone_frames() -> AsyncIterator[rtc.AudioFrame]:
    """~70 ms sine at 1.2 kHz, mono int16, gentle envelope to reduce clicks."""
    duration_s = 0.07
    freq_hz = 1200.0
    amplitude = 0.22 * 32767.0
    total = int(_SAMPLE_RATE * duration_s)
    n = 0
    while n < total:
        chunk_len = min(_CHUNK_SAMPLES, total - n)
        samples: list[int] = []
        for i in range(chunk_len):
            t = (n + i) / _SAMPLE_RATE
            env = 1.0
            edge = int(_SAMPLE_RATE * 0.012)
            pos = n + i
            if pos < edge:
                env = pos / max(edge, 1)
            elif pos > total - edge - 1:
                env = max(0.0, (total - 1 - pos) / max(edge, 1))
            v = int(amplitude * env * math.sin(2.0 * math.pi * freq_hz * t))
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


def tts_debug_chime_audio_config() -> AudioConfig:
    """Fresh async iterator per ``play()`` call."""
    return AudioConfig(_short_tone_frames(), volume=1.0)


async def start_debug_chime_player(*, room: rtc.Room) -> BackgroundAudioPlayer:
    player = BackgroundAudioPlayer()
    await player.start(room=room)
    return player
