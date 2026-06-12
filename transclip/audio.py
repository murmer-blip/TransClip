from __future__ import annotations

import tempfile
import time
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any

from .settings import Settings


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

    def _stop_raw_audio(self):
        if self._stream is None:
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


def _candidate_input_devices(sd, preferred: str) -> list[object]:
    candidates: list[object] = []
    preferred = preferred.strip()
    if preferred:
        candidates.extend(_matching_input_devices(sd, preferred))
    default_owner = getattr(sd, "default", None)
    default = _default_input_device(getattr(default_owner, "device", None))
    candidates.append(default)
    candidates.extend(_all_input_devices(sd))
    seen: set[object] = set()
    unique: list[object] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique or [None]


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
