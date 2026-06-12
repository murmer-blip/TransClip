# macOS Local Development

Use this loop when developing TransClip on macOS Apple Silicon from a checkout.

## Setup

```bash
uv sync --extra audio --extra mlx --extra macos-ui
uv run -m transclip.cli init-config
uv run -m transclip.cli models prefetch --model mlx-community/whisper-large-v3-turbo-asr-fp16
uv run -m transclip.cli install
uv run -m transclip.cli install-macos-hotkey
```

`install` writes the service LaunchAgent. `install-macos-hotkey` rewrites:

- `~/bin/transclip-toggle`
- `~/Applications/TransClipHotkey.app`
- `~/Library/LaunchAgents/com.paulbrav.transclip-hotkey.plist`

The helper owns the menu-bar `TC` item and reads stage updates from
`~/Library/Logs/transclip/hotkey-state.tsv`.

After the final `install-macos-hotkey` run, refresh Accessibility for
`TransClipHotkey` before testing. Recompiling and re-signing the helper can make
macOS treat it as a new app, so another reinstall may require the refresh again.

## Edit And Test Loop

Run focused tests for the area you changed:

```bash
uv run -m unittest tests.test_macos_hotkey -v
uv run ruff check transclip/desktop/hotkey/macos.py tests/test_macos_hotkey.py
uv run ruff format --check transclip/desktop/hotkey/macos.py tests/test_macos_hotkey.py
```

For service-side Python changes, restart the service:

```bash
uv run -m transclip.cli restart
uv run -m transclip.cli status
```

For macOS hotkey or wrapper changes, reinstall the helper from the checkout:

```bash
uv run -m transclip.cli install-macos-hotkey
```

If the helper app was recompiled or re-signed, do the Accessibility refresh now,
after the final reinstall and before the manual smoke test:

```bash
launchctl bootout gui/$(id -u)/com.paulbrav.transclip-hotkey
```

In **System Settings > Privacy & Security > Accessibility**, delete and re-add
`TransClipHotkey`, or toggle it off and back on. Then start the helper again:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.paulbrav.transclip-hotkey.plist
```

Then verify the generated wrapper picked up your edit and the helper can see the
hotkey:

```bash
rg -n 'STATE|MAX_SECONDS|STALE_LOCK_SECONDS' ~/bin/transclip-toggle
cat ~/Library/Logs/transclip/hotkey-state.tsv
tail -n 20 ~/Library/Logs/transclip/hotkey.log
tail -n 20 ~/Library/Logs/transclip/toggle-record.log
```

## Accessibility Reset

This reset is required after first setup and often after any helper rebuild.
Reinstalling `TransClipHotkey.app` recompiles and re-signs the helper, and macOS
may keep a stale Accessibility grant. If notifications repeat or the hotkey log
shows `axTrusted=false`, stop the helper before changing permissions:

```bash
launchctl bootout gui/$(id -u)/com.paulbrav.transclip-hotkey
```

In **System Settings > Privacy & Security > Accessibility**, delete and re-add
`TransClipHotkey`, or toggle it off and back on. Then start the helper again:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.paulbrav.transclip-hotkey.plist
```

The healthy log lines are:

```text
event tap starting axTrusted=true
event tap listening for Option+Space
```

## Smoke Test

Put the cursor in a text field, press `Option+Space`, say a short phrase, then
press `Option+Space` again. Confirm:

```bash
uv run -m transclip.cli status
tail -n 20 ~/Library/Logs/transclip/toggle-record.log
```

A healthy run ends with copied transcript output and `paste exited 0`.
