#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from types import SimpleNamespace
from typing import Any, ClassVar

from transclip.audio import AudioRecorder
from transclip.settings import Settings


class FakeInputStream:
    instances: ClassVar[list[FakeInputStream]] = []
    fail_default = False

    def __init__(self, **kwargs: Any):
        if self.fail_default and "device" not in kwargs:
            raise RuntimeError("default input failed")
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeSoundDevice:
    default = SimpleNamespace(device=[None, 2])
    InputStream = FakeInputStream
    query_delay_s = 0.0
    query_calls = 0

    @classmethod
    def query_devices(cls, device=None, kind=None):
        del kind
        cls.query_calls += 1
        if cls.query_delay_s:
            time.sleep(cls.query_delay_s)
        devices = [
            {"name": "MacBook Microphone", "max_input_channels": 1},
            {"name": "Display Audio", "max_input_channels": 0},
            {"name": "USB Microphone", "max_input_channels": 1},
        ]
        if device is None:
            return devices
        return devices[int(device)]


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.9))
    return {
        "min": round(ordered[0], 3),
        "median": round(statistics.median(ordered), 3),
        "mean": round(statistics.fmean(ordered), 3),
        "p90": round(ordered[p90_index], 3),
        "max": round(ordered[-1], 3),
    }


def run_benchmark(repetitions: int, warmups: int, query_delay_ms: float, fail_default: bool) -> dict[str, Any]:
    previous_numpy = sys.modules.get("numpy")
    previous_sounddevice = sys.modules.get("sounddevice")
    FakeSoundDevice.query_delay_s = query_delay_ms / 1000
    FakeSoundDevice.query_calls = 0
    FakeInputStream.instances = []
    FakeInputStream.fail_default = fail_default
    sys.modules["numpy"] = SimpleNamespace(int16="int16")
    sys.modules["sounddevice"] = FakeSoundDevice
    try:
        elapsed_ms: list[float] = []
        total_runs = repetitions + warmups
        for index in range(total_runs):
            recorder = AudioRecorder(Settings())
            started = time.perf_counter_ns()
            recorder.start()
            ended = time.perf_counter_ns()
            recorder.discard()
            if index >= warmups:
                elapsed_ms.append((ended - started) / 1_000_000)
        devices = [str(stream.kwargs.get("device", "default")) for stream in FakeInputStream.instances[warmups:]]
        return {
            "repetitions": repetitions,
            "warmups": warmups,
            "query_delay_ms": query_delay_ms,
            "fail_default": fail_default,
            "query_devices_calls": FakeSoundDevice.query_calls,
            "opened_device_counts": dict(Counter(devices)),
            "recorder_start_ms": _stats(elapsed_ms),
        }
    finally:
        if previous_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = previous_numpy
        if previous_sounddevice is None:
            sys.modules.pop("sounddevice", None)
        else:
            sys.modules["sounddevice"] = previous_sounddevice


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark AudioRecorder.start with faked sounddevice latency.")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--query-delay-ms", type=float, default=5.0)
    parser.add_argument("--fail-default", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    print(
        json.dumps(
            run_benchmark(args.repetitions, args.warmups, args.query_delay_ms, args.fail_default),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
