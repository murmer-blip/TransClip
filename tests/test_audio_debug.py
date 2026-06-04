import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
from transclip.audio import AudioRecorder, recording_debug, sounddevice_summary
from transclip.settings import Settings


class FakeRecorder:
    samples = np.array([[0], [1000], [-1000]], dtype=np.int16)

    def __init__(self, settings):
        self.settings = settings
        self.started = False

    def start(self):
        self.started = True

    def stop_samples(self):
        return type(self).samples


class FakeChunk:
    def __init__(self, pcm: bytes):
        self.pcm = pcm

    def copy(self):
        raise AssertionError("recorder should not retain copied callback chunks")

    def tobytes(self):
        return self.pcm


class FakeInputStream:
    instances: ClassVar[list["FakeInputStream"]] = []

    def __init__(self, *, callback, **kwargs):
        del kwargs
        self.callback = callback
        self.stopped = False
        self.closed = False
        type(self).instances.append(self)

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeRawAudio:
    def __init__(self):
        self.data = bytearray()
        self.position = 0
        self.closed = False

    def write(self, data):
        self.data.extend(data)

    def seek(self, position):
        self.position = position

    def read(self, size=-1):
        if size is None or size < 0:
            raise AssertionError("WAV output should stream raw audio in bounded chunks")
        chunk = bytes(self.data[self.position : self.position + size])
        self.position += len(chunk)
        return chunk

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class FakeStream:
    def __init__(self):
        self.stopped = False
        self.closed = False

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class AudioDebugTests(unittest.TestCase):
    def test_audio_recorder_streams_callback_audio_to_wav_without_copying_chunks(self):
        FakeInputStream.instances = []
        raw_audio = FakeRawAudio()
        with (
            patch.dict("sys.modules", {"sounddevice": SimpleNamespace(InputStream=FakeInputStream)}),
            patch("transclip.audio.tempfile.TemporaryFile", return_value=raw_audio),
        ):
            recorder = AudioRecorder(Settings(sample_rate=8000))
            recorder.start()
            stream = FakeInputStream.instances[-1]
            stream.callback(FakeChunk(b"\x01\x00\x02\x00"), 2, None, None)
            stream.callback(FakeChunk(b"\x03\x00"), 1, None, None)

            with tempfile.TemporaryDirectory() as tmp:
                output = recorder.stop_to_wav(Path(tmp) / "recording.wav")
                with wave.open(str(output), "rb") as wav:
                    self.assertEqual(wav.getframerate(), 8000)
                    self.assertEqual(wav.getnframes(), 3)
                    self.assertEqual(wav.readframes(3), b"\x01\x00\x02\x00\x03\x00")

        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)
        self.assertTrue(raw_audio.closed)

    def test_audio_recorder_fails_if_running_stream_has_no_audio_buffer(self):
        recorder = AudioRecorder(Settings())
        stream = FakeStream()
        recorder._stream = stream
        recorder._raw_audio = None

        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(RuntimeError, "Recorder audio buffer is not available"),
        ):
            recorder.stop_to_wav(Path(tmp) / "recording.wav")

        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)

    def test_recording_debug_reports_audio_metrics(self):
        with patch("transclip.audio.time.sleep"):
            result = recording_debug(Settings(sample_rate=3), recorder_cls=FakeRecorder)

        self.assertEqual(result["sample_rate"], 3)
        self.assertEqual(result["channel_count"], 1)
        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["duration"], 1.0)
        self.assertEqual(result["peak_amplitude"], 1000.0)
        self.assertFalse(result["silent"])

    def test_recording_debug_reports_silence(self):
        FakeRecorder.samples = np.zeros((4, 1), dtype=np.int16)
        try:
            with patch("transclip.audio.time.sleep"):
                result = recording_debug(Settings(sample_rate=4), recorder_cls=FakeRecorder)
        finally:
            FakeRecorder.samples = np.array([[0], [1000], [-1000]], dtype=np.int16)

        self.assertTrue(result["silent"])
        self.assertEqual(result["rms_amplitude"], 0.0)

    def test_sounddevice_summary_handles_missing_dependency(self):
        with patch.dict("sys.modules", {"sounddevice": None}):
            self.assertEqual(sounddevice_summary(), "sounddevice unavailable")

    def test_sounddevice_summary_handles_missing_portaudio(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "sounddevice":
                raise OSError("PortAudio library not found")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(sounddevice_summary(), "sounddevice unavailable")


if __name__ == "__main__":
    unittest.main()
