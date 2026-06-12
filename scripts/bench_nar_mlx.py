"""Benchmark granite NAR MLX pass time on Apple Silicon — the Mac incremental gate.

Run on an M-series Mac after `transclip models prefetch --model
mlx-community/granite-speech-4.1-2b-nar-mlx`. Measures warm pass time vs
window length over the synthetic eval audio and writes
eval/macos/nar-mlx-bench.json with a pass/fail verdict.

Gate: warm 8 s window pass <= 900 ms. If it passes, enabling Mac incremental
transcription is a follow-up commit: flip darwin_arm_mlx's
incremental_transcription_supported in transclip/platform/profiles.py and add
a transcribe_waveform entry point to the MLX backend.
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transclip.asr import MlxAudioASRBackend  # noqa: E402
from transclip.settings import load_settings  # noqa: E402

MODEL_ID = "mlx-community/granite-speech-4.1-2b-nar-mlx"
SR = 16000
WINDOW_SECONDS = (2.0, 4.0, 8.0, 16.0, 30.0)
GATE_WINDOW_S = 8.0
GATE_MAX_MS = 900.0
REPS = 3
OUTPUT = REPO_ROOT / "eval" / "macos" / "nar-mlx-bench.json"


def load_speech() -> np.ndarray:
    clips = []
    for wav in sorted((REPO_ROOT / "eval" / "v1-synthetic" / "audio").glob("case_*.wav")):
        data, rate = sf.read(str(wav), dtype="float32", always_2d=True)
        assert rate == SR, f"{wav} is {rate} Hz, expected {SR}"
        clips.append(data.mean(axis=1) if data.shape[1] > 1 else data[:, 0])
    speech = np.concatenate(clips)
    print(f"speech available: {len(speech) / SR:.1f}s")
    return speech


def main() -> None:
    if platform.system() != "Darwin":
        sys.exit("This benchmark must run on an Apple Silicon Mac.")
    settings = load_settings()
    backend = MlxAudioASRBackend(MODEL_ID, settings)
    speech = load_speech()
    tmpdir = Path(tempfile.mkdtemp(prefix="nar-mlx-bench-"))

    def run_window(seconds: float) -> float:
        n = int(seconds * SR)
        buf = speech[:n] if len(speech) >= n else np.tile(speech, int(np.ceil(n / len(speech))))[:n]
        wav = tmpdir / f"window_{seconds:g}.wav"
        sf.write(str(wav), buf, SR)
        started = time.perf_counter()
        result = backend.transcribe(wav)
        elapsed_ms = (time.perf_counter() - started) * 1000
        del result
        return elapsed_ms

    print("warmup (includes model load + compile)...")
    warmup_ms = run_window(4.0)
    print(f"warmup pass: {warmup_ms:.0f}ms")
    run_window(4.0)

    results: dict[str, dict[str, float]] = {}
    for seconds in WINDOW_SECONDS:
        times = sorted(run_window(seconds) for _ in range(REPS))
        results[f"{seconds:g}"] = {"best_ms": round(times[0], 1), "median_ms": round(times[REPS // 2], 1)}
        print(f"window={seconds:>4g}s best={times[0]:6.0f}ms median={times[REPS // 2]:6.0f}ms")

    gate_ms = results[f"{GATE_WINDOW_S:g}"]["median_ms"]
    passed = gate_ms <= GATE_MAX_MS
    payload = {
        "pass": passed,
        "gate": {"window_s": GATE_WINDOW_S, "max_ms": GATE_MAX_MS, "measured_ms": gate_ms},
        "model": MODEL_ID,
        "results_ms_by_window_s": results,
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'PASS' if passed else 'FAIL'}: warm {GATE_WINDOW_S:g}s pass = {gate_ms:.0f}ms "
          f"(gate {GATE_MAX_MS:.0f}ms)")
    print(f"wrote {OUTPUT}")
    if passed:
        print("Next: flip darwin_arm_mlx incremental_transcription_supported in "
              "transclip/platform/profiles.py and add transcribe_waveform to the MLX backend.")
    else:
        print("Gate failed: keep Mac on batch; consider the Moonshine contingency "
              "(see plans/2026-06-12-streaming-investigation-report.md).")


if __name__ == "__main__":
    main()
