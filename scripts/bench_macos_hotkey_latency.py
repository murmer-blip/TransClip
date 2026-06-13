#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from transclip.desktop.hotkey.macos import build_macos_toggle_wrapper
from transclip.settings import Settings


@dataclass(frozen=True, slots=True)
class RequestEvent:
    method: str
    path: str
    ns: int


class BenchRuntime:
    def __init__(self, home: Path):
        self._home = home

    def system(self) -> str:
        return "Darwin"

    def home_dir(self) -> Path:
        return self._home

    def environ(self, name: str, default: str | None = None) -> str | None:
        return os.environ.get(name, default)

    def env_snapshot(self, names=()) -> dict[str, str]:
        return {name: os.environ.get(name, "") for name in names}

    def which(self, program: str) -> str | None:
        return shutil.which(program)

    def run(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, **kwargs)

    def check_output(self, command: list[str], **kwargs: Any) -> str:
        kwargs.setdefault("text", True)
        output = subprocess.check_output(command, **kwargs)
        return output.decode() if isinstance(output, bytes) else output


class FakeTransClipService(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), FakeTransClipHandler)
        self.events: list[RequestEvent] = []
        self._events_lock = threading.Lock()
        self.state = "ready"

    def reset(self, state: str) -> None:
        with self._events_lock:
            self.events = []
        self.state = state

    def record_event(self, method: str, path: str) -> None:
        with self._events_lock:
            self.events.append(RequestEvent(method, path, time.perf_counter_ns()))

    def snapshot_events(self) -> list[RequestEvent]:
        with self._events_lock:
            return list(self.events)


class FakeTransClipHandler(BaseHTTPRequestHandler):
    server: FakeTransClipService

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        self.server.record_event("GET", path)
        if path == "/health":
            self._json(200, {"status": self.server.state})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        self.server.record_event("POST", path)
        if path == "/record/start":
            if self.server.state == "recording":
                self._json(200, {"status": "recording", "already_recording": True})
                return
            self.server.state = "recording"
            self._json(200, {"status": "recording", "already_recording": False})
            return
        if path == "/record/stop":
            self.server.state = "ready"
            self._json(200, {"status": "ready", "text": "bench transcript"})
            return
        if path == "/record/toggle":
            if self.server.state == "recording":
                self.server.state = "ready"
                self._json(200, {"status": "ready", "action": "stopped", "text": "bench transcript"})
                return
            self.server.state = "recording"
            self._json(200, {"status": "recording", "action": "started", "already_recording": False})
            return
        self._json(404, {"error": "not found"})

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


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


def _write_wrapper(path: Path, script: str, lock_path: Path) -> None:
    safe_script = script.replace("LOCK=/tmp/transclip-toggle.lock", f"LOCK={shlex.quote(str(lock_path))}")
    path.write_text(safe_script, encoding="utf-8")
    path.chmod(0o755)


def _write_fake_pbcopy(path: Path) -> None:
    path.write_text("#!/bin/sh\ncat >/dev/null\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def _first_record_event(events: list[RequestEvent]) -> RequestEvent | None:
    for event in events:
        if event.method == "POST" and event.path in {"/record/start", "/record/stop", "/record/toggle"}:
            return event
    return None


def _first_state_change_event(events: list[RequestEvent], scenario: str) -> RequestEvent | None:
    target_paths = {"/record/start", "/record/toggle"} if scenario == "start" else {"/record/stop", "/record/toggle"}
    for event in events:
        if event.method == "POST" and event.path in target_paths:
            return event
    return None


def run_benchmark(repetitions: int, warmups: int, scenario: str, timeout: float) -> dict[str, Any]:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required to benchmark the generated macOS wrapper")

    with tempfile.TemporaryDirectory(prefix="transclip-hotkey-bench-") as tmp:
        root = Path(tmp)
        fake_bin = root / "bin"
        fake_bin.mkdir()
        _write_fake_pbcopy(fake_bin / "pbcopy")

        service = FakeTransClipService()
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        host, port = service.server_address

        try:
            runtime = BenchRuntime(root / "home")
            wrapper = root / "transclip-toggle"
            _write_wrapper(
                wrapper,
                build_macos_toggle_wrapper(Settings(host=host, port=port), runtime=runtime),
                root / "transclip-toggle.lock",
            )
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"

            record_latencies_ms: list[float] = []
            state_change_latencies_ms: list[float] = []
            wrapper_runtime_ms: list[float] = []
            request_counts: dict[str, int] = {}
            total_runs = warmups + repetitions
            initial_state = "recording" if scenario == "stop" else "ready"

            for index in range(total_runs):
                service.reset(initial_state)
                started_ns = time.perf_counter_ns()
                completed = subprocess.run(
                    [str(wrapper)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                ended_ns = time.perf_counter_ns()
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"wrapper exited {completed.returncode}: "
                        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
                    )
                events = service.snapshot_events()
                first_record = _first_record_event(events)
                if first_record is None:
                    raise RuntimeError(f"wrapper did not reach a record endpoint; events={events!r}")
                state_change = _first_state_change_event(events, scenario)
                if state_change is None:
                    raise RuntimeError(f"wrapper did not reach the {scenario} state-change endpoint; events={events!r}")
                if index >= warmups:
                    record_latencies_ms.append((first_record.ns - started_ns) / 1_000_000)
                    state_change_latencies_ms.append((state_change.ns - started_ns) / 1_000_000)
                    wrapper_runtime_ms.append((ended_ns - started_ns) / 1_000_000)
                    for event in events:
                        request_counts[event.path] = request_counts.get(event.path, 0) + 1

            return {
                "scenario": scenario,
                "repetitions": repetitions,
                "warmups": warmups,
                "request_counts": request_counts,
                "shortcut_to_record_request_ms": _stats(record_latencies_ms),
                "shortcut_to_state_change_request_ms": _stats(state_change_latencies_ms),
                "wrapper_runtime_ms": _stats(wrapper_runtime_ms),
            }
        finally:
            service.shutdown()
            service.server_close()
            thread.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark generated macOS hotkey wrapper latency with a fake service."
    )
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--scenario", choices=("start", "stop"), default="start")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")

    print(json.dumps(run_benchmark(args.repetitions, args.warmups, args.scenario, args.timeout), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
