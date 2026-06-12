import unittest
from pathlib import Path
from unittest.mock import patch

from transclip.service import DictationSession
from transclip.service.streaming import StreamingDictationAdapter
from transclip.settings import Settings

from tests.service_helpers import FakeRecorder, FakeStreamingASR, FakeStreamingSessionFactory


class StepClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


class DictationSessionTests(unittest.TestCase):
    def test_toggle_discards_recording_under_minimum_duration(self):
        session = DictationSession(
            Settings(min_recording_ms=500, toggle_cooldown_ms=0),
            transcribe=lambda _wav, _cleanup, _source: {"text": "should not run"},
            recorder_factory=FakeRecorder,
            clock=StepClock([1.0, 1.0, 1.1, 1.1, 1.1]),
        )

        started = session.toggle_recording()
        stopped = session.toggle_recording()

        self.assertEqual(started["action"], "started")
        self.assertEqual(stopped["action"], "discarded")
        self.assertTrue(stopped["discarded"])
        self.assertEqual(stopped["duration_ms"], 100.0)

    def test_toggle_discards_recording_over_maximum_duration_without_transcribing(self):
        transcribe_calls = []

        session = DictationSession(
            Settings(max_recording_ms=1_000, min_recording_ms=0, toggle_cooldown_ms=0),
            transcribe=lambda _wav, _cleanup, _source: transcribe_calls.append(_wav),
            recorder_factory=FakeRecorder,
            clock=StepClock([1.0, 1.0, 2.25, 2.25, 2.25]),
        )

        session.toggle_recording()
        stopped = session.toggle_recording()

        self.assertEqual(stopped["action"], "discarded")
        self.assertTrue(stopped["discarded"])
        self.assertEqual(stopped["reason"], "max_recording_duration")
        self.assertEqual(stopped["max_recording_ms"], 1_000)
        self.assertEqual(transcribe_calls, [])

    def test_stop_calls_transcriber_with_recorded_wav_and_source(self):
        calls = []

        def transcribe(wav_path: Path, cleanup, source):
            calls.append((wav_path, cleanup, source, wav_path.exists()))
            return {"text": "Hello.", "status": "ready"}

        session = DictationSession(
            Settings(min_recording_ms=0, toggle_cooldown_ms=0),
            transcribe=transcribe,
            recorder_factory=FakeRecorder,
            clock=StepClock([2.0, 2.25]),
        )

        session.start_recording()
        result = session.stop_recording(cleanup=True, source="/record/stop")

        self.assertEqual(result["text"], "Hello.")
        self.assertEqual(result["duration_ms"], 250.0)
        self.assertEqual(calls[0][1:], (True, "/record/stop", True))

    def test_streaming_finish_uses_adapter_instead_of_wav_transcriber(self):
        streaming = FakeStreamingASR(final_text="streamed final")
        factory = FakeStreamingSessionFactory(streaming)
        settings = Settings(min_recording_ms=0, toggle_cooldown_ms=0)
        transcribe_calls = []

        def process_asr_result(asr_result, *, cleanup, source, **kwargs):
            return {
                "text": asr_result.text,
                "raw_asr": asr_result.text,
                "status": "ready",
                "cleanup": {},
                "cleanup_enabled": True,
                "timings_ms": asr_result.timings_ms,
            }

        adapter = StreamingDictationAdapter(settings, factory, process_asr_result)

        class FakeChunkedRecorder(FakeRecorder):
            def __init__(self, recorder_settings, *, on_chunk=None, **kwargs):
                super().__init__(recorder_settings)

            def stop_capture(self):
                self.discarded = True

        session = DictationSession(
            settings,
            transcribe=lambda *_args: transcribe_calls.append(True),
            streaming=adapter,
            clock=StepClock([1.0, 1.2]),
        )
        with patch("transclip.service.streaming.ChunkedAudioRecorder", FakeChunkedRecorder):
            session.start_recording()
        session.partial_text()
        streaming.feed(b"\x00" * 100)
        result = session.stop_recording()
        self.assertEqual(result["text"], "streamed final")
        self.assertEqual(transcribe_calls, [])


if __name__ == "__main__":
    unittest.main()
