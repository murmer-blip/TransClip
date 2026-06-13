"""Benchmark warm macOS MLX TXT latency with generated WAV cases.

This exercises the in-process ASR plus post-processing path used after a
recording stops, without requiring live recording, accessibility permissions, or
the synthetic eval audio assets.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transclip.service import InferenceEngine  # noqa: E402
from transclip.settings import load_settings  # noqa: E402

SAMPLE_RATE = 16_000
DEFAULT_CASES = (
    ("silence_0_75s", 0.75, "silence"),
    ("tone_1_50s", 1.50, "tone"),
    ("chirp_3_00s", 3.00, "chirp"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--label", default="mlx-txt-latency")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error("--reps must be at least 1")

    settings = load_settings()
    engine = InferenceEngine(settings)
    with tempfile.TemporaryDirectory(prefix="transclip-txt-bench-") as tmp:
        root = Path(tmp)
        warmup = None if args.skip_warmup else run_warmup(engine, root)
        rows = run_cases(engine, root, reps=args.reps)

    payload = {
        "label": args.label,
        "settings": {
            "asr_backend": settings.asr_backend,
            "asr_model": settings.asr_model,
            "models_local_files_only": settings.models_local_files_only,
        },
        "warmup": warmup,
        "summary": summarize(rows),
        "by_case": summarize_by_case(rows),
        "rows": rows,
    }
    encoded = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def run_warmup(engine: InferenceEngine, root: Path) -> dict[str, Any]:
    wav = write_case(root, "warmup_0_25s", 0.25, "silence")
    start = time.perf_counter()
    result = engine.transcribe(wav, cleanup=True, keywords=[])
    return {
        "wall_ms": round((time.perf_counter() - start) * 1000, 3),
        "timings_ms": result["timings_ms"],
        "text_len": len(result["text"]),
    }


def run_cases(engine: InferenceEngine, root: Path, *, reps: int) -> list[dict[str, Any]]:
    rows = []
    for rep in range(1, reps + 1):
        for name, seconds, kind in DEFAULT_CASES:
            wav = write_case(root, f"{name}_rep{rep}", seconds, kind)
            start = time.perf_counter()
            result = engine.transcribe(wav, cleanup=True, keywords=[])
            rows.append(
                {
                    "case": name,
                    "rep": rep,
                    "duration_s": seconds,
                    "wall_ms": round((time.perf_counter() - start) * 1000, 3),
                    "timings_ms": result["timings_ms"],
                    "text_len": len(result["text"]),
                }
            )
    return rows


def write_case(root: Path, name: str, seconds: float, kind: str) -> Path:
    path = root / f"{name}.wav"
    sf.write(str(path), samples(kind, seconds), SAMPLE_RATE)
    return path


def samples(kind: str, seconds: float) -> np.ndarray:
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    if kind == "silence":
        return np.zeros(n, dtype=np.float32)
    if kind == "tone":
        return (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if kind == "chirp":
        freq = 180 + 420 * (t / max(seconds, 1e-6))
        return (0.12 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    raise ValueError(f"unknown generated audio kind: {kind}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    wall = [row["wall_ms"] for row in rows]
    end_to_end = [row["timings_ms"].get("end_to_end", 0.0) for row in rows]
    asr = [row["timings_ms"].get("asr", 0.0) for row in rows]
    return {
        "cases": len(rows),
        "median_wall_ms": round(statistics.median(wall), 3),
        "mean_wall_ms": round(statistics.mean(wall), 3),
        "best_wall_ms": round(min(wall), 3),
        "worst_wall_ms": round(max(wall), 3),
        "median_end_to_end_ms": round(statistics.median(end_to_end), 3),
        "median_asr_ms": round(statistics.median(asr), 3),
    }


def summarize_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_case = {}
    for name, *_ in DEFAULT_CASES:
        values = [row["wall_ms"] for row in rows if row["case"] == name]
        by_case[name] = {
            "median_wall_ms": round(statistics.median(values), 3),
            "best_wall_ms": round(min(values), 3),
            "worst_wall_ms": round(max(values), 3),
        }
    return by_case


if __name__ == "__main__":
    raise SystemExit(main())
