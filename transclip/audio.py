from __future__ import annotations

import tempfile
import time
import wave
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from .settings import Settings

OnChunkCallback = Callable[[bytes], None]


class AudioRecorder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sd = None
        self._np = None
        self._stream = None
        self._raw_audio = None

    def start(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install transclip[audio] for microphone capture.") from exc
        self._np = np
        self._sd = sd
        self._raw_audio = tempfile.TemporaryFile()  # noqa: SIM115 - closed by stop_to_wav, stop_samples, or discard.

        def callback(indata, frames, time, status):
            del frames, time
            if status:
                return
            raw_audio = self._raw_audio
            if raw_audio is not None:
                raw_audio.write(indata.tobytes())
            self._on_capture_frame(indata)

        errors: list[str] = []
        for device in _candidate_input_devices(sd, self.settings.audio_input_device):
            try:
                kwargs = {
                    "samplerate": self.settings.sample_rate,
                    "channels": 1,
                    "dtype": "int16",
                    "callback": callback,
                }
                if device is not None:
                    kwargs["device"] = device
                self._stream = sd.InputStream(**kwargs)
                self._stream.start()
                return
            except Exception as exc:
                errors.append(_device_error_detail(sd, device, exc))
                with suppress(Exception):
                    self._stop_stream()
        try:
            self.discard()
        finally:
            detail = "; ".join(errors) if errors else "no input devices were attempted"
            raise _recording_start_error(RuntimeError(detail))

    def _on_capture_frame(self, indata) -> None:
        del indata

    def stop_to_wav(self, output_path: Path) -> Path:
        raw_audio = self._stop_raw_audio()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_audio, wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.settings.sample_rate)
            while chunk := raw_audio.read(1024 * 1024):
                wav.writeframesraw(chunk)
        return output_path

    def stop_samples(self):
        if self._stream is None or self._np is None:
            raise RuntimeError("Recorder is not running")
        with self._stop_raw_audio() as raw_audio:
            pcm = raw_audio.read()
            if not pcm:
                return self._np.zeros((0, 1), dtype="int16")
            return self._np.frombuffer(pcm, dtype=self._np.int16).reshape(-1, 1).copy()

    def discard(self) -> None:
        try:
            self._stop_stream()
        finally:
            self._close_raw_audio()

    def stop_capture(self) -> None:
        """Stop the mic stream without writing a WAV."""
        self.discard()

    def _stop_raw_audio(self):
        if self._stream is None and self._raw_audio is None:
            raise RuntimeError("Recorder is not running")
        self._stop_stream()
        raw_audio = self._raw_audio
        if raw_audio is None:
            raise RuntimeError("Recorder audio buffer is not available")
        self._raw_audio = None
        raw_audio.seek(0)
        return raw_audio

    def _stop_stream(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def _close_raw_audio(self) -> None:
        raw_audio = self._raw_audio
        self._raw_audio = None
        if raw_audio is not None:
            raw_audio.close()


class ChunkedAudioRecorder(AudioRecorder):
    """Microphone recorder that flushes fixed-size PCM chunks to a callback."""

    def __init__(
        self,
        settings: Settings,
        *,
        on_chunk: OnChunkCallback | None = None,
        chunk_ms: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(settings)
        self._on_chunk = on_chunk
        self._chunk_ms = chunk_ms if chunk_ms is not None else settings.streaming_chunk_ms
        self._clock = clock
        self._chunk_buffer = bytearray()
        self._chunk_byte_target = max(
            2,
            round(settings.sample_rate * self._chunk_ms / 1000.0) * 2,
        )

    def start(self) -> None:
        self._chunk_buffer = bytearray()
        super().start()

    def _on_capture_frame(self, indata) -> None:
        if self._on_chunk is None:
            return
        self._chunk_buffer.extend(bytes(indata))
        while len(self._chunk_buffer) >= self._chunk_byte_target:
            payload = bytes(self._chunk_buffer[: self._chunk_byte_target])
            del self._chunk_buffer[: self._chunk_byte_target]
            self._on_chunk(payload)

    def _flush_remaining_chunks(self) -> None:
        if self._on_chunk is None or not self._chunk_buffer:
            return
        self._on_chunk(bytes(self._chunk_buffer))
        self._chunk_buffer.clear()

    def stop_capture(self) -> None:
        """Stop the mic stream, then flush pending chunks without writing a WAV.

        The stream must stop first: the capture callback mutates the chunk
        buffer concurrently, and any frames it delivers after the flush would
        be silently discarded (clipping the tail of the last word).
        """
        self._stop_stream()
        self._flush_remaining_chunks()
        self._close_raw_audio()

    def stop_to_wav(self, output_path: Path) -> Path:
        self._stop_stream()
        self._flush_remaining_chunks()
        return super().stop_to_wav(output_path)

    def discard(self) -> None:
        self._chunk_buffer.clear()
        super().discard()


def _recording_start_error(exc: Exception) -> RuntimeError:
    detail = str(exc)
    lowered = detail.lower()
    if "permission" in lowered or "denied" in lowered or "not authorized" in lowered:
        return RuntimeError(
            "Microphone permission denied. On macOS, grant Microphone access in "
            "System Settings > Privacy & Security > Microphone for Terminal, your IDE, "
            "or the LaunchAgent Python process."
        )
    if "no input" in lowered or "device" in lowered:
        return RuntimeError(f"No usable microphone input device: {detail}")
    return RuntimeError(f"Could not start microphone capture: {detail}")


def recording_debug(
    settings: Settings,
    seconds: float = 1.0,
    recorder_cls: type[AudioRecorder] = AudioRecorder,
) -> dict[str, Any]:
    recorder = recorder_cls(settings)
    recorder.start()
    try:
        time.sleep(seconds)
        audio = recorder.stop_samples()
    except Exception:
        with suppress(Exception):
            recorder.stop_samples()
        raise
    frame_count = int(audio.shape[0]) if hasattr(audio, "shape") else 0
    channel_count = int(audio.shape[1]) if hasattr(audio, "shape") and len(audio.shape) > 1 else 1
    duration = frame_count / settings.sample_rate if settings.sample_rate else 0.0
    peak = 0.0
    rms = 0.0
    if frame_count:
        import numpy as np

        audio_float = audio.astype("float32")
        peak = float(np.max(np.abs(audio_float)))
        rms = float(np.sqrt(np.mean(audio_float * audio_float)))
    return {
        "device": sounddevice_summary(),
        "sample_rate": settings.sample_rate,
        "channel_count": channel_count,
        "frame_count": frame_count,
        "duration": duration,
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "silent": peak == 0.0,
    }


def sounddevice_summary() -> str:
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        return "sounddevice unavailable"
    try:
        default = getattr(sd.default, "device", None)
        input_device = _default_input_device(default)
        if input_device in {None, -1}:
            return f"default input={input_device}"
        info = sd.query_devices(input_device, "input")
        name = info.get("name", "unknown") if isinstance(info, dict) else str(info)
        return f"default input={input_device} {name}"
    except Exception as exc:
        return f"sounddevice query failed: {exc}"


def _candidate_input_devices(sd, preferred: str) -> Iterator[object]:
    seen: set[object] = set()
    emitted = False

    def should_emit(candidate: object) -> bool:
        nonlocal emitted
        if candidate in seen:
            return False
        seen.add(candidate)
        emitted = True
        return True

    preferred = preferred.strip()
    default_owner = getattr(sd, "default", None)
    default = _default_input_device(getattr(default_owner, "device", None))
    if preferred:
        if preferred.isdigit():
            if should_emit(int(preferred)):
                yield int(preferred)
        else:
            if _default_device_matches_preferred(sd, default, preferred) and should_emit(default):
                yield default
            for candidate in _matching_input_devices(sd, preferred):
                if should_emit(candidate):
                    yield candidate
    elif should_emit(default):
        yield default
    for candidate in _all_input_devices(sd):
        if should_emit(candidate):
            yield candidate
    if not emitted:
        yield None


def _default_device_matches_preferred(sd, default: object, preferred: str) -> bool:
    if default in {None, -1}:
        return False
    try:
        info = sd.query_devices(default, "input")
    except Exception:
        return False
    name = str(info.get("name", "") if isinstance(info, dict) else info)
    return preferred.lower() == name.lower()


def _matching_input_devices(sd, preferred: str) -> list[object]:
    if preferred.isdigit():
        return [int(preferred)]
    lowered = preferred.lower()
    matches: list[object] = []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    for index, device in enumerate(devices):
        if not _is_input_device(device):
            continue
        name = str(device.get("name", "") if isinstance(device, dict) else device)
        if lowered == name.lower() or lowered in name.lower():
            matches.append(index)
    return matches


def _all_input_devices(sd) -> list[object]:
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    return [index for index, device in enumerate(devices) if _is_input_device(device)]


def _is_input_device(device: object) -> bool:
    if isinstance(device, dict):
        return int(device.get("max_input_channels", 0) or 0) > 0
    return " in," in str(device) or " in)" in str(device)


def _device_error_detail(sd, device: object, exc: Exception) -> str:
    label = "default"
    if device not in {None, -1}:
        label = str(device)
        with suppress(Exception):
            info = sd.query_devices(device, "input")
            if isinstance(info, dict):
                label = f"{device} {info.get('name', 'unknown')}"
    return f"{label}: {exc}"


def _default_input_device(default: object) -> object:
    if isinstance(default, tuple):
        return default[0] if default else None
    if isinstance(default, list):
        return default[0] if default else None
    return default


def write_wav(path: Path, pcm16_mono: bytes, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16_mono)
    return path


def pcm16_to_float32(pcm16_mono: bytes):
    import numpy as np

    if not pcm16_mono:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm16_mono, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def float32_to_pcm16(samples) -> bytes:
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()
