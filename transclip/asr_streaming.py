from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .asr import TranscriptionResult
from .audio import float32_to_pcm16


@dataclass(slots=True)
class PartialTranscript:
    text: str
    language: str | None = None


class StreamingASRSession(Protocol):
    @property
    def partial_text(self) -> PartialTranscript: ...

    def feed(self, pcm16_mono: bytes) -> None: ...

    def finish(self) -> TranscriptionResult: ...

    def close(self) -> None: ...


class StreamingASRSessionFactory(Protocol):
    def __call__(self) -> StreamingASRSession: ...


def feed_pcm16_chunks(
    session: StreamingASRSession,
    mono_samples: Any,
    *,
    chunk_ms: int,
    sample_rate: int,
    after_chunk: Callable[[], None] | None = None,
) -> None:
    """Feed float32 mono audio to a streaming session in fixed PCM16 chunks."""
    step = max(1, round(chunk_ms / 1000.0 * sample_rate))
    pos = 0
    while pos < len(mono_samples):
        segment = mono_samples[pos : pos + step]
        pos += len(segment)
        session.feed(float32_to_pcm16(segment))
        if after_chunk is not None:
            after_chunk()
