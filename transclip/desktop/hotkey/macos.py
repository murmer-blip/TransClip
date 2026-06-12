from __future__ import annotations

import plistlib
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from transclip.daemon.common import CommandResult, logs_dir, repo_root
from transclip.paths import service_settings_path
from transclip.platform.runtime import PlatformRuntime, get_runtime, user_cache_dir
from transclip.product import CACHE_DIR_NAME, IMPORT_PACKAGE
from transclip.settings import Settings

Runner = Callable[..., subprocess.CompletedProcess[str]]

HOTKEY_APP_NAME = "TransClipHotkey"
HOTKEY_BUNDLE_ID = "com.paulbrav.TransClipHotkey"
HOTKEY_LAUNCHD_LABEL = "com.paulbrav.transclip-hotkey"
HOTKEY_LOG_NAME = "hotkey.log"
TOGGLE_WRAPPER_NAME = "transclip-toggle"
DEFAULT_STOP_TIMEOUT_SECONDS = 180
STALE_LOCK_SECONDS = 240


@dataclass(frozen=True, slots=True)
class MacOSHotkeyInstall:
    app_path: Path
    launch_agent_path: Path
    wrapper_path: Path
    source_path: Path


def macos_hotkey_app_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "Applications" / f"{HOTKEY_APP_NAME}.app"


def macos_hotkey_launch_agent_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "Library" / "LaunchAgents" / f"{HOTKEY_LAUNCHD_LABEL}.plist"


def macos_toggle_wrapper_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "bin" / TOGGLE_WRAPPER_NAME


def macos_hotkey_source_path(runtime: PlatformRuntime | None = None) -> Path:
    return user_cache_dir(CACHE_DIR_NAME, runtime) / "macos-hotkey" / f"{HOTKEY_APP_NAME}.swift"


def macos_hotkey_log_path(runtime: PlatformRuntime | None = None) -> Path:
    return logs_dir(runtime) / HOTKEY_LOG_NAME


def macos_hotkey_target(runtime: PlatformRuntime | None = None) -> str:
    return f"{macos_launchd_gui_domain(runtime)}/{HOTKEY_LAUNCHD_LABEL}"


def macos_launchd_gui_domain(runtime: PlatformRuntime | None = None) -> str:
    output = get_runtime(runtime).check_output(["id", "-u"])
    if isinstance(output, bytes):
        output = output.decode()
    return f"gui/{output.strip()}"


def build_macos_toggle_wrapper(
    settings: Settings,
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
    *,
    stop_timeout_seconds: int = DEFAULT_STOP_TIMEOUT_SECONDS,
    stale_lock_seconds: int = STALE_LOCK_SECONDS,
) -> str:
    log_path = logs_dir(runtime) / "toggle-record.log"
    base_url = f"http://{settings.host}:{settings.port}"
    python = shlex.quote(sys.executable)
    cli = f"{python} -m {shlex.quote(IMPORT_PACKAGE + '.cli')}"
    if settings_path:
        cli += f" --settings {shlex.quote(service_settings_path(settings_path))}"
    restart_command = f'cd {shlex.quote(str(repo_root()))} && {cli} restart >> "$LOG" 2>&1'
    return f"""#!/bin/sh
set -u

LOG={shlex.quote(str(log_path))}
BASE={shlex.quote(base_url)}
MAX_SECONDS={int(stop_timeout_seconds)}
STALE_LOCK_SECONDS={int(stale_lock_seconds)}
LOCK=/tmp/transclip-toggle.lock

mkdir -p "$(dirname "$LOG")"
printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') wrapper invoked" >> "$LOG"

if ! mkdir "$LOCK" 2>/dev/null; then
  now=$(date +%s)
  lock_mtime=$(stat -f %m "$LOCK" 2>/dev/null || printf '0')
  lock_age=$((now - lock_mtime))
  if [ "$lock_age" -lt "$STALE_LOCK_SECONDS" ]; then
    printf '%s\\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S%z') ignored: previous TransClip action still running (${{lock_age}}s)" \
      >> "$LOG"
    exit 0
  fi
  printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') clearing stale TransClip action lock (${{lock_age}}s)" >> "$LOG"
  rm -rf "$LOCK"
  if ! mkdir "$LOCK" 2>/dev/null; then
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') ignored: could not acquire TransClip action lock" >> "$LOG"
    exit 0
  fi
fi
trap 'rm -rf "$LOCK"' EXIT HUP INT TERM

health=$(curl -sS --max-time 5 "$BASE/health" 2>>"$LOG") || {{
  printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') health failed; restarting service" >> "$LOG"
  {restart_command}
  exit 0
}}
status=$(printf '%s' "$health" | {python} -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>>"$LOG")
printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') status=${{status}}" >> "$LOG"

if [ "$status" = "recording" ]; then
  response=$(curl -sS --max-time "$MAX_SECONDS" -X POST "$BASE/record/stop" \
    -H 'content-type: application/json' --data '{{}}' 2>>"$LOG") || {{
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') stop failed; restarting service" >> "$LOG"
    {restart_command}
    exit 0
  }}
  printf '%s\\n' "$response" >> "$LOG"
  text=$(printf '%s' "$response" | {python} -c \
    'import json,sys; print(json.load(sys.stdin).get("text", ""), end="")' 2>>"$LOG")
  if [ -n "$text" ]; then
    printf '%s' "$text" | pbcopy
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') copied transcript chars=${{#text}}" >> "$LOG"
    osascript -e 'tell application "System Events" to keystroke "v" using command down' >> "$LOG" 2>&1
    paste_status=$?
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') paste exited ${{paste_status}}" >> "$LOG"
  else
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') stop returned no text" >> "$LOG"
  fi
else
  response=$(curl -sS --max-time 10 -X POST "$BASE/record/start" \
    -H 'content-type: application/json' --data '{{}}' 2>>"$LOG") || {{
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') start failed; restarting service" >> "$LOG"
    {restart_command}
    exit 0
  }}
  printf '%s\\n' "$response" >> "$LOG"
fi

exit 0
"""


def build_macos_hotkey_source(
    wrapper_path: Path,
    log_path: Path,
) -> str:
    return f"""import ApplicationServices
import Carbon
import Foundation

let logPath = "{_swift_string(str(log_path))}"
let wrapperPath = "{_swift_string(str(wrapper_path))}"
let spaceKeyCode: Int64 = 49

func log(_ message: String) {{
    let formatter = ISO8601DateFormatter()
    let line = "\\(formatter.string(from: Date())) \\(message)\\n"
    guard let data = line.data(using: .utf8) else {{ return }}

    if FileManager.default.fileExists(atPath: logPath),
       let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: logPath)) {{
        defer {{ try? handle.close() }}
        try? handle.seekToEnd()
        try? handle.write(contentsOf: data)
    }} else {{
        try? data.write(to: URL(fileURLWithPath: logPath))
    }}
}}

func runWrapper() {{
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/sh")
    process.arguments = ["-lc", wrapperPath]
    do {{
        try process.run()
        log("launched wrapper pid=\\(process.processIdentifier)")
    }} catch {{
        log("failed to launch wrapper: \\(error)")
    }}
}}

let promptKey = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
let trusted = AXIsProcessTrustedWithOptions([promptKey: true] as CFDictionary)
log("event tap starting axTrusted=\\(trusted)")

let callback: CGEventTapCallBack = {{ _, type, event, _ in
    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {{
        log("event tap disabled type=\\(type.rawValue)")
        return Unmanaged.passUnretained(event)
    }}

    guard type == .keyDown else {{
        return Unmanaged.passUnretained(event)
    }}

    let keyCode = event.getIntegerValueField(.keyboardEventKeycode)
    let flags = event.flags
    let hasOption = flags.contains(.maskAlternate)
    let hasCommand = flags.contains(.maskCommand)
    let hasControl = flags.contains(.maskControl)

    if keyCode == spaceKeyCode && hasOption && !hasCommand && !hasControl {{
        log("Option+Space detected")
        runWrapper()
        return nil
    }}

    return Unmanaged.passUnretained(event)
}}

let mask = CGEventMask(1 << CGEventType.keyDown.rawValue)
guard let eventTap = CGEvent.tapCreate(
    tap: .cgSessionEventTap,
    place: .headInsertEventTap,
    options: .defaultTap,
    eventsOfInterest: mask,
    callback: callback,
    userInfo: nil
) else {{
    log("failed to create event tap; Accessibility/Input Monitoring is required")
    exit(2)
}}

let runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventTap, 0)
CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
CGEvent.tapEnable(tap: eventTap, enable: true)
log("event tap listening for Option+Space")
CFRunLoopRun()
"""


def build_macos_hotkey_launch_agent(runtime: PlatformRuntime | None = None) -> bytes:
    log_root = logs_dir(runtime)
    payload = {
        "Label": HOTKEY_LAUNCHD_LABEL,
        "ProgramArguments": [str(_macos_hotkey_executable_path(runtime))],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_root / "hotkey.out.log"),
        "StandardErrorPath": str(log_root / "hotkey.err.log"),
    }
    return plistlib.dumps(payload, sort_keys=True)


def install_macos_hotkey(
    settings: Settings,
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> tuple[MacOSHotkeyInstall, list[CommandResult]]:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() != "Darwin":
        raise RuntimeError("macOS hotkey helper is only supported on Darwin")

    paths = MacOSHotkeyInstall(
        app_path=macos_hotkey_app_path(platform_runtime),
        launch_agent_path=macos_hotkey_launch_agent_path(platform_runtime),
        wrapper_path=macos_toggle_wrapper_path(platform_runtime),
        source_path=macos_hotkey_source_path(platform_runtime),
    )
    results: list[CommandResult] = []
    logs_dir(platform_runtime).mkdir(parents=True, exist_ok=True)

    paths.wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    paths.wrapper_path.write_text(
        build_macos_toggle_wrapper(settings, settings_path, platform_runtime),
        encoding="utf-8",
    )
    paths.wrapper_path.chmod(0o755)
    results.append(CommandResult(True, f"wrote {paths.wrapper_path}"))

    swiftc = platform_runtime.which("swiftc")
    if not swiftc:
        results.append(
            CommandResult(False, "swiftc missing; install Xcode Command Line Tools with: xcode-select --install")
        )
        return paths, results

    executable = _macos_hotkey_executable_path(platform_runtime)
    executable.parent.mkdir(parents=True, exist_ok=True)
    paths.source_path.parent.mkdir(parents=True, exist_ok=True)
    paths.source_path.write_text(
        build_macos_hotkey_source(paths.wrapper_path, macos_hotkey_log_path(platform_runtime)),
        encoding="utf-8",
    )
    _macos_hotkey_info_plist_path(platform_runtime).write_bytes(_build_info_plist())
    results.append(CommandResult(True, f"wrote {paths.source_path}"))
    results.append(_run_command([swiftc, str(paths.source_path), "-o", str(executable)], runner))
    if not results[-1].ok:
        return paths, results
    executable.chmod(0o755)

    codesign = platform_runtime.which("codesign") or "codesign"
    results.append(_run_command([codesign, "--force", "--deep", "--sign", "-", str(paths.app_path)], runner))
    if not results[-1].ok:
        return paths, results

    paths.launch_agent_path.parent.mkdir(parents=True, exist_ok=True)
    paths.launch_agent_path.write_bytes(build_macos_hotkey_launch_agent(platform_runtime))
    results.append(CommandResult(True, f"wrote {paths.launch_agent_path}"))
    target = macos_hotkey_target(platform_runtime)
    domain = macos_launchd_gui_domain(platform_runtime)
    results.append(_run_command(["launchctl", "bootout", target], runner, tolerate_failure=True))
    results.append(_run_command(["launchctl", "bootstrap", domain, str(paths.launch_agent_path)], runner))
    results.append(
        CommandResult(
            True,
            (
                "Grant Accessibility to TransClipHotkey.app. On first paste, allow "
                "TransClipHotkey to control System Events."
            ),
        )
    )
    return paths, results


def uninstall_macos_hotkey(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    platform_runtime = get_runtime(runtime)
    results = [
        _run_command(["launchctl", "bootout", macos_hotkey_target(platform_runtime)], runner, tolerate_failure=True)
    ]
    path = macos_hotkey_launch_agent_path(platform_runtime)
    if path.exists():
        path.unlink()
        results.append(CommandResult(True, f"removed {path}"))
    app_path = macos_hotkey_app_path(platform_runtime)
    if app_path.exists():
        shutil.rmtree(app_path)
        results.append(CommandResult(True, f"removed {app_path}"))
    wrapper_path = macos_toggle_wrapper_path(platform_runtime)
    if wrapper_path.exists():
        wrapper_path.unlink()
        results.append(CommandResult(True, f"removed {wrapper_path}"))
    source_path = macos_hotkey_source_path(platform_runtime)
    if source_path.exists():
        source_path.unlink()
        results.append(CommandResult(True, f"removed {source_path}"))
    return results


def _run_command(command: list[str], runner: Runner, tolerate_failure: bool = False) -> CommandResult:
    try:
        result = runner(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    except FileNotFoundError as exc:
        return CommandResult(tolerate_failure, f"{command[0]} missing: {exc}")
    output = result.stdout.strip()
    ok = result.returncode == 0 or tolerate_failure
    detail = shlex.join(command)
    if output:
        detail += f": {output}"
    elif result.returncode != 0:
        detail += f": exit {result.returncode}"
    return CommandResult(ok, detail)


def _build_info_plist() -> bytes:
    return plistlib.dumps(
        {
            "CFBundleExecutable": HOTKEY_APP_NAME,
            "CFBundleIdentifier": HOTKEY_BUNDLE_ID,
            "CFBundleName": HOTKEY_APP_NAME,
            "CFBundleDisplayName": HOTKEY_APP_NAME,
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "1.0",
            "CFBundleVersion": "1",
            "LSUIElement": True,
        },
        sort_keys=True,
    )


def _macos_hotkey_executable_path(runtime: PlatformRuntime | None = None) -> Path:
    return macos_hotkey_app_path(runtime) / "Contents" / "MacOS" / HOTKEY_APP_NAME


def _macos_hotkey_info_plist_path(runtime: PlatformRuntime | None = None) -> Path:
    return macos_hotkey_app_path(runtime) / "Contents" / "Info.plist"


def _swift_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
