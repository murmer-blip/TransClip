import itertools
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from transclip.asr import TranscriptionResult
from transclip.asr_incremental import (
    BUCKET_S,
    IncrementalNarSession,
    _find_commit_cut,
    _pad_to_bucket,
    incremental_transcription_enabled,
)
from transclip.settings import Settings

SR = 16000


def tone_pcm16(seconds: float, amplitude: float = 0.5) -> bytes:
    n = int(seconds * SR)
    return (np.full(n, amplitude) * 32767.0).astype(np.int16).tobytes()


def silence_pcm16(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * SR)


class ScriptedTranscriber:
    """Fake transcribe_waveform: records received waveforms, returns seg<N>."""

    def __init__(self):
        self.calls: list[np.ndarray] = []
        self.called = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, waveform) -> TranscriptionResult:
        with self._lock:
            self.calls.append(np.asarray(waveform))
            index = len(self.calls)
        self.called.set()
        return TranscriptionResult(f"seg{index}", {"asr": 5.0}, "fake", "fake-model")

    def nonzero_seconds(self) -> float:
        with self._lock:
            return sum(int(np.count_nonzero(call)) for call in self.calls) / SR


def make_session(transcriber, threshold_s: float = 10.0) -> IncrementalNarSession:
    return IncrementalNarSession(
        transcriber,
        sample_rate=SR,
        commit_threshold_s=threshold_s,
        backend_name="fake",
        model_name="fake-model",
    )


def feed_chunks(session: IncrementalNarSession, pcm: bytes, chunk_ms: int = 500) -> None:
    step = SR * chunk_ms // 1000 * 2
    for start in range(0, len(pcm), step):
        session.feed(pcm[start : start + step])


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class FindCommitCutTests(unittest.TestCase):
    def _cut(self, pcm: bytes):
        return _find_commit_cut(
            pcm,
            sample_rate=SR,
            tail_bytes=2 * SR * 2,
            min_commit_bytes=2 * SR * 2,
        )

    def test_cuts_at_silence_midpoint(self):
        pcm = tone_pcm16(6.0) + silence_pcm16(1.0) + tone_pcm16(5.0)
        cut, has_speech = self._cut(pcm)
        self.assertIsNotNone(cut)
        cut_seconds = cut / 2 / SR
        self.assertGreater(cut_seconds, 6.2)
        self.assertLess(cut_seconds, 6.8)
        self.assertTrue(has_speech)
        self.assertEqual(cut % 2, 0)

    def test_continuous_speech_returns_none(self):
        cut, _ = self._cut(tone_pcm16(12.0))
        self.assertIsNone(cut)

    def test_silence_only_in_tail_is_ignored(self):
        pcm = tone_pcm16(10.5) + silence_pcm16(1.5)
        cut, _ = self._cut(pcm)
        self.assertIsNone(cut)

    def test_too_early_silence_returns_none(self):
        pcm = tone_pcm16(0.5) + silence_pcm16(1.0) + tone_pcm16(10.5)
        cut, _ = self._cut(pcm)
        self.assertIsNone(cut)

    def test_pure_silence_reports_no_speech(self):
        pcm = silence_pcm16(12.0)
        cut, has_speech = self._cut(pcm)
        self.assertIsNotNone(cut)
        self.assertFalse(has_speech)

    def test_noisy_floor_falls_back_to_relative_threshold(self):
        rng = np.random.default_rng(7)
        loud = (rng.uniform(-0.5, 0.5, 8 * SR) * 32767).astype(np.int16).tobytes()
        hum = (rng.uniform(-0.02, 0.02, int(1.0 * SR)) * 32767).astype(np.int16).tobytes()
        pcm = loud + hum + (rng.uniform(-0.5, 0.5, 3 * SR) * 32767).astype(np.int16).tobytes()
        cut, has_speech = self._cut(pcm)
        self.assertIsNotNone(cut)
        cut_seconds = cut / 2 / SR
        self.assertGreater(cut_seconds, 8.0)
        self.assertLess(cut_seconds, 9.0)
        self.assertTrue(has_speech)


class PadToBucketTests(unittest.TestCase):
    def test_pads_to_bucket_multiple(self):
        samples = np.ones(int(6.5 * SR), dtype=np.float32)
        padded = _pad_to_bucket(samples, SR)
        self.assertEqual(len(padded) % int(BUCKET_S * SR), 0)
        self.assertEqual(len(padded), 8 * SR)
        self.assertTrue(np.all(padded[len(samples) :] == 0.0))

    def test_exact_multiple_untouched(self):
        samples = np.ones(8 * SR, dtype=np.float32)
        self.assertIs(_pad_to_bucket(samples, SR), samples)


class IncrementalNarSessionTests(unittest.TestCase):
    def test_short_utterance_is_a_single_pass(self):
        transcriber = ScriptedTranscriber()
        session = make_session(transcriber)
        feed_chunks(session, tone_pcm16(5.0))
        result = session.finish()
        self.assertEqual(len(transcriber.calls), 1)
        self.assertEqual(result.text, "seg1")
        self.assertEqual(result.backend, "fake")
        self.assertNotIn("commits", result.timings_ms)

    def test_commit_at_silence_then_residual(self):
        transcriber = ScriptedTranscriber()
        session = make_session(transcriber)
        feed_chunks(session, tone_pcm16(6.0) + silence_pcm16(1.0) + tone_pcm16(5.0))
        self.assertTrue(wait_for(lambda: session.partial_text.text == "seg1"))
        result = session.finish()
        self.assertEqual(len(transcriber.calls), 2)
        self.assertEqual(result.text, "seg1 seg2")
        self.assertEqual(result.timings_ms["commits"], 1.0)
        self.assertGreater(result.timings_ms["asr_committed_total"], 0.0)
        bucket_samples = int(BUCKET_S * SR)
        self.assertEqual(len(transcriber.calls[0]) % bucket_samples, 0)
        self.assertAlmostEqual(transcriber.nonzero_seconds(), 11.0, delta=0.3)

    def test_no_silence_means_no_commits(self):
        transcriber = ScriptedTranscriber()
        session = make_session(transcriber)
        feed_chunks(session, tone_pcm16(12.0))
        time.sleep(0.3)
        self.assertEqual(session.partial_text.text, "")
        result = session.finish()
        self.assertEqual(len(transcriber.calls), 1)
        self.assertEqual(result.text, "seg1")

    def test_committed_text_never_changes(self):
        transcriber = ScriptedTranscriber()
        session = make_session(transcriber)
        pcm = b"".join(
            [
                tone_pcm16(6.0),
                silence_pcm16(1.0),
                tone_pcm16(6.0),
                silence_pcm16(1.0),
                tone_pcm16(6.0),
                silence_pcm16(1.0),
                tone_pcm16(3.0),
            ]
        )
        partials = [session.partial_text.text]
        step = SR // 2 * 2
        for start in range(0, len(pcm), step):
            session.feed(pcm[start : start + step])
            time.sleep(0.002)
            partials.append(session.partial_text.text)
        wait_for(lambda: session.partial_text.text != "")
        partials.append(session.partial_text.text)
        result = session.finish()
        for earlier, later in itertools.pairwise(partials):
            self.assertTrue(later.startswith(earlier))
        self.assertTrue(result.text.startswith(partials[-1]))

    def test_speech_free_commit_trims_without_transcribing(self):
        transcriber = ScriptedTranscriber()
        session = make_session(transcriber)
        feed_chunks(session, silence_pcm16(12.0))
        self.assertTrue(wait_for(lambda: session._commit_count >= 1))
        self.assertEqual(len(transcriber.calls), 0)
        self.assertEqual(session.partial_text.text, "")
        result = session.finish()
        self.assertEqual(len(transcriber.calls), 1)
        self.assertEqual(result.timings_ms["commits"], 1.0)

    def test_finish_with_no_audio(self):
        transcriber = ScriptedTranscriber()
        session = make_session(transcriber)
        result = session.finish()
        self.assertEqual(result.text, "")
        self.assertEqual(len(transcriber.calls), 0)

    def test_close_terminates_worker_and_ignores_feeds(self):
        transcriber = ScriptedTranscriber()
        session = make_session(transcriber)
        session.feed(tone_pcm16(1.0))
        session.close()
        self.assertFalse(session._worker.is_alive())
        session.feed(tone_pcm16(1.0))
        with session._lock:
            self.assertEqual(len(session._buffer), 0)

    def test_transcriber_failure_keeps_audio_for_finish(self):
        calls: list[int] = []

        def flaky(waveform) -> TranscriptionResult:
            calls.append(len(waveform))
            if len(calls) == 1:
                raise RuntimeError("GPU fell over")
            return TranscriptionResult("recovered", {"asr": 5.0}, "fake", "fake-model")

        session = make_session(flaky)
        feed_chunks(session, tone_pcm16(6.0) + silence_pcm16(1.0) + tone_pcm16(5.0))
        self.assertTrue(wait_for(lambda: len(calls) >= 1))
        result = session.finish()
        self.assertEqual(result.text, "recovered")
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(calls[-1], 12 * SR)

    def test_concurrent_feeds_conserve_audio(self):
        transcriber = ScriptedTranscriber()
        session = make_session(transcriber, threshold_s=100.0)
        chunk = tone_pcm16(0.25)

        def pump():
            for _ in range(8):
                session.feed(chunk)

        threads = [threading.Thread(target=pump) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        result = session.finish()
        self.assertEqual(result.text, "seg1")
        self.assertAlmostEqual(transcriber.nonzero_seconds(), 3 * 8 * 0.25, delta=0.01)


class GatingTests(unittest.TestCase):
    def test_disabled_by_settings_flag(self):
        settings = Settings(incremental_transcription=False)
        self.assertFalse(incremental_transcription_enabled(settings))

    def test_follows_profile_capability(self):
        settings = Settings(incremental_transcription=True)
        with patch("transclip.platform.profiles.detect_runtime_profile") as detect:
            detect.return_value.incremental_transcription_supported = True
            self.assertTrue(incremental_transcription_enabled(settings))
            detect.return_value.incremental_transcription_supported = False
            self.assertFalse(incremental_transcription_enabled(settings))


class EngineGatingTests(unittest.TestCase):
    def test_engine_without_waveform_backend_has_no_streaming(self):
        from transclip.cleanup import FaithfulRuleCleanupBackend
        from transclip.service import InferenceEngine

        from tests.service_helpers import FakeASR, FakeTextBackend

        engine = InferenceEngine(
            Settings(incremental_transcription=True),
            asr_backend=FakeASR(),
            cleanup_backend=FaithfulRuleCleanupBackend(),
            text_backend=FakeTextBackend(),
        )
        self.assertIsNone(engine._streaming)

    def test_engine_with_waveform_backend_builds_adapter(self):
        from transclip.cleanup import FaithfulRuleCleanupBackend
        from transclip.service import InferenceEngine

        from tests.service_helpers import FakeASR, FakeTextBackend

        class WaveformFakeASR(FakeASR):
            def transcribe_waveform(self, waveform, sample_rate: int = SR) -> TranscriptionResult:
                return TranscriptionResult(self.text, {"asr": 1.0}, self.name, self.model)

        with patch("transclip.service.engine.incremental_transcription_enabled", return_value=True):
            engine = InferenceEngine(
                Settings(incremental_transcription=True),
                asr_backend=WaveformFakeASR(),
                cleanup_backend=FaithfulRuleCleanupBackend(),
                text_backend=FakeTextBackend(),
            )
        self.assertIsNotNone(engine._streaming)
        self.assertTrue(engine.health()["streaming_partial_supported"])
        session = engine._streaming._session_factory()
        try:
            self.assertIsInstance(session, IncrementalNarSession)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
