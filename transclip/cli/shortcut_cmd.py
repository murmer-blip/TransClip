from __future__ import annotations

import argparse

from transclip.desktop.hotkey import install_macos_hotkey, install_shortcut, uninstall_macos_hotkey
from transclip.settings import Settings


def handle_gnome_shortcut(args: argparse.Namespace, settings: Settings) -> int:
    result = install_shortcut(
        settings_path=args.settings,
        command=args.shortcut_command,
        binding=args.binding or settings.hotkey_linux,
    )
    print(f"Installed {result.name}")
    print(f"Path: {result.path}")
    print(f"Binding: {result.binding}")
    print(f"Command: {result.command}")
    return 0


def handle_macos_hotkey(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "uninstall-macos-hotkey":
        results = uninstall_macos_hotkey()
        for result in results:
            print(("ok" if result.ok else "fail") + "\t" + result.detail)
        return 0 if all(result.ok for result in results) else 1

    install, results = install_macos_hotkey(settings=settings, settings_path=args.settings)
    print(f"App: {install.app_path}")
    print(f"LaunchAgent: {install.launch_agent_path}")
    print(f"Wrapper: {install.wrapper_path}")
    for result in results:
        print(("ok" if result.ok else "fail") + "\t" + result.detail)
    return 0 if all(result.ok for result in results) else 1
