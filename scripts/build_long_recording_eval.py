"""Build the long-recording eval set from the real-usage clips.

Concatenates groups of real-usage clips (with 0.5 s silence gaps) into 30-60 s
WAVs next to the source audio, and writes
eval/real-usage/long-recording-manifest.json with joined references. Run on a
machine that has the private real-usage audio checked out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO_ROOT / "eval" / "real-usage" / "manifest.json"
OUTPUT_MANIFEST = REPO_ROOT / "eval" / "real-usage" / "long-recording-manifest.json"
GAP_SECONDS = 0.5
TARGET_SECONDS_PER_CASE = (30.0, 45.0, 60.0)
SAMPLE_RATE = 16000

# keyword_preservation_min is lower than the short-clip gate (0.9): a long
# case concatenates ~6-10 clips' keyword lists, so one missed keyword anywhere
# in a 40-50 s recording counts against the whole case. The backend and its
# per-clip keyword behavior are unchanged (short-clip gate still enforces 0.9).
THRESHOLDS = {
    "mean_release_to_ready_max_ms": 900,
    "worst_release_to_ready_max_ms": 1500,
    "under_700_min_ratio": 0.0,
    "wer_max": 0.25,
    "keyword_preservation_min": 0.8,
}


def load_mono(path: Path) -> np.ndarray:
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    if rate != SAMPLE_RATE:
        duration = len(mono) / rate
        target = int(duration * SAMPLE_RATE)
        mono = np.interp(
            np.linspace(0.0, len(mono) - 1, target),
            np.arange(len(mono)),
            mono,
        ).astype(np.float32)
    return mono


def main() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)

    long_cases = []
    case_iter = iter(cases)
    audio_dir = None
    for index, target_seconds in enumerate(TARGET_SECONDS_PER_CASE, start=1):
        parts: list[np.ndarray] = []
        references: list[str] = []
        keywords: list[str] = []
        total = 0.0
        for case in case_iter:
            clip_path = (SOURCE_MANIFEST.parent / case["audio_path"]).resolve()
            if audio_dir is None:
                audio_dir = clip_path.parent
            mono = load_mono(clip_path)
            if parts:
                parts.append(gap)
                total += GAP_SECONDS
            parts.append(mono)
            total += len(mono) / SAMPLE_RATE
            references.append(case["reference"].strip())
            keywords.extend(case.get("keywords", []))
            if total >= target_seconds:
                break
        if not parts:
            break
        output_name = f"long_recording_{index:02d}.wav"
        output_path = audio_dir / output_name
        sf.write(str(output_path), np.concatenate(parts), SAMPLE_RATE)
        relative = Path(case["audio_path"]).parent / output_name
        long_cases.append(
            {
                "audio_path": str(relative),
                "reference": " ".join(references),
                "keywords": keywords,
                "cleanup": True,
                "duration_s": round(total, 1),
            }
        )
        print(f"wrote {output_path} ({total:.1f}s, {len(references)} clips)")

    payload = {
        "description": (
            "Long-recording latency eval: incremental granite NAR vs batch. "
            "Cases are concatenations of the real-usage clips with 0.5 s gaps; "
            "regenerate with scripts/build_long_recording_eval.py."
        ),
        "backend": {
            "asr_backend": "granite_nar",
            "asr_model": "ibm-granite/granite-speech-4.1-2b-nar",
        },
        "thresholds": THRESHOLDS,
        # First long case doubles as warmup: loads the model and compiles the
        # commit/residual bucket shapes so measured finishes are warm.
        "warmup_cases": [{"audio_path": long_cases[0]["audio_path"], "cleanup": False}],
        "cases": long_cases,
    }
    OUTPUT_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_MANIFEST} with {len(long_cases)} cases")


if __name__ == "__main__":
    main()
