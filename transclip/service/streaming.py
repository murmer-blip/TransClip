from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Protocol

from transclip.asr import TranscriptionResult
from transclip.asr_streaming import PartialTranscript, StreamingASRSession, StreamingASRSessionFactory
from transclip.audio import ChunkedAudioRecorder
from transclip.settings import Settings

from .types import TranscribeResponse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamingCapture:
    recorder: ChunkedAudioRecorder
    session: StreamingASRSession


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
        self._current: StreamingASRSession | None = None

    def create_recording(self) -> StreamingCapture:
        session = self._session_factory()
        with self._lock:
            previous = self._current
            self._current = session
        if previous is not None:
            # A leftover session means the prior recording never finished
            # cleanly; close it so its worker thread does not leak.
            logger.warning("Closing leftover streaming session before starting a new recording")
            previous.close()
        return StreamingCapture(ChunkedAudioRecorder(self._settings, on_chunk=session.feed), session)

    def partial_text(self) -> PartialTranscript:
        with self._lock:
            session = self._current
            if session is None:
                return PartialTranscript("")
            return session.partial_text

    def detach_session(self, session: StreamingASRSession) -> None:
        with self._lock:
            if self._current is session:
                self._current = None

    def finish_session(
        self,
        session: StreamingASRSession,
        cleanup: bool | None,
        source: str,
        wav_path: Path | None = None,
    ) -> TranscribeResponse:
        self.detach_session(session)
        start = perf_counter()
        asr_result = session.finish()
        return self._process_asr_result(
            asr_result,
            cleanup=cleanup,
            source=source,
            start_time=start,
            wav_path=wav_path,
        )

    def discard_session(self, session: StreamingASRSession) -> None:
        self.detach_session(session)
        session.close()
