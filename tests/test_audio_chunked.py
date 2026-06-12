import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from transclip.audio import ChunkedAudioRecorder
from transclip.settings import Settings


class ChunkedAudioRecorderTests(unittest.TestCase):
    def test_flushes_fixed_size_chunks_to_callback(self):
        chunks: list[bytes] = []
        recorder = ChunkedAudioRecorder(
            Settings(sample_rate=16000, streaming_chunk_ms=500),
            on_chunk=chunks.append,
        )
        frame_bytes = 2
        recorder._chunk_byte_target = frame_bytes * 4
        samples = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=np.int16)
        for row in samples:
            recorder._on_capture_frame(row)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), frame_bytes * 4)
        self.assertEqual(len(chunks[1]), frame_bytes * 4)

    def test_stop_to_wav_flushes_remainder(self):
        chunks: list[bytes] = []
        recorder = ChunkedAudioRecorder(
            Settings(sample_rate=16000, streaming_chunk_ms=500),
            on_chunk=chunks.append,
        )
        recorder._chunk_byte_target = 8
        recorder._on_capture_frame(np.array([[1], [2], [3]], dtype=np.int16))
        self.assertEqual(chunks, [])
        with patch.object(ChunkedAudioRecorder.__bases__[0], "stop_to_wav", return_value=Path("/tmp/x.wav")):
            recorder.stop_to_wav(Path("/tmp/x.wav"))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 6)

    def test_stop_capture_flushes_remainder_without_wav(self):
        chunks: list[bytes] = []
        recorder = ChunkedAudioRecorder(
            Settings(sample_rate=16000, streaming_chunk_ms=500),
            on_chunk=chunks.append,
        )
        recorder._chunk_byte_target = 8
        recorder._on_capture_frame(np.array([[1], [2], [3]], dtype=np.int16))
        recorder.stop_capture()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 6)


if __name__ == "__main__":
    unittest.main()
