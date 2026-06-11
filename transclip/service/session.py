from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Protocol

from transclip.asr_streaming import PartialTranscript
from transclip.audio import AudioRecorder
from transclip.settings import Settings

from .streaming import StreamingDictationAdapter
from .types import RecordSessionResponse, TranscribeResponse


class Recorder(Protocol):
    def start(self) -> None: ...

    def stop_to_wav(self, output_path: Path) -> Path: ...

    def stop_capture(self) -> None: ...

    def discard(self) -> None: ...


RecorderFactory = Callable[[Settings], Recorder]
Transcriber = Callable[[Path, bool | None, str], TranscribeResponse]
Clock = Callable[[], float]


class DictationSession:
    def __init__(
        self,
        settings: Settings,
        transcribe: Transcriber,
        recorder_factory: RecorderFactory | None = None,
        clock: Clock = perf_counter,
        streaming: StreamingDictationAdapter | None = None,
    ):
        self.settings = settings
        self._transcribe = transcribe
        self._recorder_factory = recorder_factory or AudioRecorder
        self._streaming = streaming
        self._clock = clock
        self._lock = Lock()
        self._recorder: Recorder | None = None
        self._recording_started_at = 0.0
        self._last_toggle_accepted_at = 0.0

    def status(self) -> str:
        with self._lock:
            return "recording" if self._recorder else "ready"

    def partial_text(self) -> PartialTranscript:
        if self._streaming is None:
            return PartialTranscript("")
        return self._streaming.partial_text()

    def start_recording(self) -> RecordSessionResponse:
        with self._lock:
            if self._recorder is not None:
                return {"status": "recording", "already_recording": True}
            recorder = self._create_recorder()
            recorder.start()
            self._recorder = recorder
            self._recording_started_at = self._clock()
        return {"status": "recording", "already_recording": False}

    def stop_recording(
        self,
        cleanup: bool | None = None,
        discard: bool = False,
        source: str = "/record/stop",
    ) -> RecordSessionResponse:
        with self._lock:
            if self._recorder is None:
                raise RuntimeError("Recorder is not running")
            recorder = self._recorder
            started_at = self._recording_started_at
            self._recorder = None
            self._recording_started_at = 0.0
        return self._finish_recording(
            recorder,
            started_at,
            cleanup=cleanup,
            discard=discard,
            source=source,
        )

    def toggle_recording(
        self,
        cleanup: bool | None = None,
    ) -> RecordSessionResponse:
        now = self._clock()
        with self._lock:
            cooldown_seconds = max(0, self.settings.toggle_cooldown_ms) / 1000
            if (
                cooldown_seconds
                and self._last_toggle_accepted_at
                and now - self._last_toggle_accepted_at < cooldown_seconds
            ):
                return {
                    "status": "recording" if self._recorder is not None else "ready",
                    "action": "ignored",
                    "reason": "toggle_cooldown",
                    "cooldown_ms": self.settings.toggle_cooldown_ms,
                }
            self._last_toggle_accepted_at = now
            if self._recorder is None:
                recorder = self._create_recorder()
                recorder.start()
                self._recorder = recorder
                self._recording_started_at = now
                return {"status": "recording", "action": "started", "already_recording": False}

            recorder = self._recorder
            started_at = self._recording_started_at
            self._recorder = None
            self._recording_started_at = 0.0

        duration_ms = (self._clock() - started_at) * 1000
        over_maximum = (
            self.settings.max_recording_ms > 0
            and duration_ms > self.settings.max_recording_ms
        )
        discard = duration_ms < self.settings.min_recording_ms or over_maximum
        result = self._finish_recording(
            recorder,
            started_at,
            cleanup=cleanup,
            discard=discard,
            discard_reason="max_recording_duration" if over_maximum else None,
            source="/record/toggle",
        )
        result["status"] = "ready"
        result["action"] = "discarded" if discard else "stopped"
        return result

    def _finish_recording(
        self,
        recorder: Recorder,
        started_at: float,
        *,
        cleanup: bool | None,
        discard: bool,
        discard_reason: str | None = None,
        source: str,
    ) -> RecordSessionResponse:
        duration_ms = round((self._clock() - started_at) * 1000, 3)
        if discard:
            recorder.discard()
            if self._streaming is not None:
                self._streaming.on_discard()
            result: RecordSessionResponse = {"status": "ready", "duration_ms": duration_ms, "discarded": True}
            if discard_reason:
                result["reason"] = discard_reason
                result["max_recording_ms"] = self.settings.max_recording_ms
            return result
        if self._streaming is not None:
            if self.settings.debug_capture:
                with tempfile.TemporaryDirectory() as tmp:
                    wav_path = recorder.stop_to_wav(Path(tmp) / "recording.wav")
                    result = dict(self._streaming.finish_transcription(cleanup, source, wav_path=wav_path))
            else:
                recorder.stop_capture()
                result = dict(self._streaming.finish_transcription(cleanup, source))
        else:
            with tempfile.TemporaryDirectory() as tmp:
                wav_path = recorder.stop_to_wav(Path(tmp) / "recording.wav")
                result = dict(self._transcribe(wav_path, cleanup, source))
        result["duration_ms"] = duration_ms
        return result

    def _create_recorder(self) -> Recorder:
        if self._streaming is not None:
            return self._streaming.create_recorder()
        return self._recorder_factory(self.settings)
