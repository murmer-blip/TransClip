"""Run the incremental-transcription latency eval on real hardware.

Usage:
    .venv/bin/python3 scripts/run_incremental_eval.py [manifest] [--pacing 1.0]

pacing=1.0 feeds audio in real time (like live dictation), so background
commits happen while "speaking"; pacing=0 feeds as fast as possible (the
worker barely gets to run, approximating worst case).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transclip.asr_incremental import IncrementalNarSession  # noqa: E402
from transclip.eval_harness import run_incremental_eval  # noqa: E402
from transclip.service import InferenceEngine  # noqa: E402
from transclip.settings import load_settings  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "eval" / "real-usage" / "long-recording-manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--pacing", type=float, default=1.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    settings = load_settings()
    engine = InferenceEngine(settings)
    transcribe_waveform = getattr(engine.asr_backend, "transcribe_waveform", None)
    if transcribe_waveform is None:
        sys.exit(f"Backend {engine.asr_backend.name} has no transcribe_waveform; cannot run incremental eval.")

    def session_factory() -> IncrementalNarSession:
        return IncrementalNarSession(
            transcribe_waveform,
            sample_rate=settings.sample_rate,
            commit_threshold_s=settings.incremental_commit_threshold_s,
            backend_name=engine.asr_backend.name,
            model_name=engine.asr_backend.model,
        )

    result = run_incremental_eval(
        Path(args.manifest),
        engine,
        session_factory,
        pacing=args.pacing,
    )
    payload = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    summary = result["summary"]
    print(json.dumps(summary, indent=2))
    for case in result["results"]:
        timings = case["timings_ms"]
        print(
            f"{Path(case['audio_path']).name}: end_to_end={timings.get('end_to_end'):.0f}ms "
            f"commits={timings.get('commits', 0):.0f} "
            f"first_commit={case.get('time_to_first_partial_ms') or float('nan'):.0f}ms "
            f"wer={case.get('wer')}"
        )


if __name__ == "__main__":
    main()
