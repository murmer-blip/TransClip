from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Protocol

from transclip.asr import TranscriptionResult
from transclip.asr_streaming import PartialTranscript, StreamingASRSession, StreamingASRSessionFactory
from transclip.audio import ChunkedAudioRecorder
from transclip.settings import Settings

from .types import TranscribeResponse


class ProcessAsrResult(Protocol):
    def __call__(
        self,
        asr_result: TranscriptionResult,
        *,
        cleanup: bool | None,
        source: str,
        keywords: list[str] | None = None,
        end_to_end_ms: float | None = None,
        start_time: float | None = None,
        wav_path: Path | None = None,
    ) -> TranscribeResponse: ...


class StreamingDictationAdapter:
    """Thread-safe bridge between mic chunks, a streaming ASR session, and post-ASR."""

    def __init__(
        self,
        settings: Settings,
        session_factory: StreamingASRSessionFactory,
        process_asr_result: ProcessAsrResult,
    ):
        self._settings = settings
        self._session_factory = session_factory
        self._process_asr_result = process_asr_result
        self._lock = Lock()
        self._session: StreamingASRSession | None = None

    def create_recorder(self) -> ChunkedAudioRecorder:
        with self._lock:
            previous = self._session
            self._session = self._session_factory()
        if previous is not None:
            # A leftover session means the prior recording never finished
            # cleanly; close it so its worker thread does not leak.
            previous.close()
        return ChunkedAudioRecorder(self._settings, on_chunk=self._feed_chunk)

    def partial_text(self) -> PartialTranscript:
        with self._lock:
            session = self._session
            if session is None:
                return PartialTranscript("")
            return session.partial_text

    def finish_transcription(
        self,
        cleanup: bool | None,
        source: str,
        wav_path: Path | None = None,
    ) -> TranscribeResponse:
        with self._lock:
            session = self._session
            self._session = None
        if session is None:
            raise RuntimeError("Streaming session is not active")
        start = perf_counter()
        asr_result = session.finish()
        return self._process_asr_result(
            asr_result,
            cleanup=cleanup,
            source=source,
            start_time=start,
            wav_path=wav_path,
        )

    def on_discard(self) -> None:
        with self._lock:
            session = self._session
            self._session = None
        if session is not None:
            session.close()

    def _feed_chunk(self, pcm16_mono: bytes) -> None:
        with self._lock:
            session = self._session
        if session is not None:
            session.feed(pcm16_mono)
