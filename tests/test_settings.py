import tempfile
import unittest
from pathlib import Path

from transclip.settings import (
    DEFAULT_HOTKEY_LINUX,
    Settings,
    active_hotkey,
    coerce_setting_value,
    load_settings,
    paste_shortcut,
    patch_settings,
    set_setting,
    write_default_settings,
    write_settings,
)

from tests.service_helpers import FakeRuntime, linux_gpu_runtime, patch_linux_gpu_runtime


class SettingsTests(unittest.TestCase):
    def test_default_files_round_trip(self):
        with patch_linux_gpu_runtime(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = linux_gpu_runtime()
            settings_file = write_default_settings(root / "settings.toml", runtime=runtime)

            settings = load_settings(settings_file, runtime=runtime)

            self.assertEqual(settings.hotkey_linux, DEFAULT_HOTKEY_LINUX)
            self.assertEqual(settings.toggle_cooldown_ms, 500)
            self.assertEqual(settings.asr_backend, "granite_nar")
            self.assertEqual(settings.asr_model, "ibm-granite/granite-speech-4.1-2b-nar")
            self.assertTrue(settings.voice_mode_routing_enabled)
            self.assertFalse(settings.voice_model_cleanup_always_on)
            self.assertTrue(settings.voice_mode_shell_enabled)
            self.assertEqual(settings.text_model_runtime, "transformers")
            self.assertEqual(settings.text_model, "Qwen/Qwen3.5-4B")
            self.assertTrue(settings.shell_syntax_validation_enabled)
            self.assertTrue(settings.shellcheck_enabled)
            self.assertTrue(settings.models_local_files_only)
            self.assertEqual(settings.model_cache_dir, "")

    def test_unknown_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            path.write_text('hotkey_linux = "Ctrl+Space"\nwat = true\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(path)

    def test_removed_setting_is_stripped_when_settings_are_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            path.write_text(
                'hotkey_linux = "<Super><Shift>XF86TouchpadOff"\nmax_recording_seconds = 60\n',
                encoding="utf-8",
            )
            write_settings(load_settings(path), path)
            self.assertNotIn("max_recording_seconds", path.read_text(encoding="utf-8"))

    def test_legacy_max_recording_seconds_migrates_to_milliseconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            path.write_text("max_recording_seconds = 60\n", encoding="utf-8")

            settings = load_settings(path)
            write_settings(settings, path)

            text = path.read_text(encoding="utf-8")
            self.assertEqual(settings.max_recording_ms, 60_000)
            self.assertIn("max_recording_ms = 60000", text)
            self.assertNotIn("max_recording_seconds", text)

    def test_set_setting_rejects_removed_max_recording_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            write_default_settings(path)
            with self.assertRaises(ValueError):
                set_setting(path, "max_recording_seconds", "60")

    def test_legacy_qwen_streaming_keys_migrate_and_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            path.write_text(
                "qwen_streaming_chunk_ms = 400\n"
                "qwen_streaming_chunk_size_sec = 2.0\n"
                "qwen_streaming_unfixed_chunk_num = 2\n"
                "qwen_streaming_unfixed_token_num = 5\n"
                "qwen_streaming_gpu_memory_utilization = 0.8\n"
                "qwen_streaming_max_new_tokens = 32\n",
                encoding="utf-8",
            )

            settings = load_settings(path)
            write_settings(settings, path)

            text = path.read_text(encoding="utf-8")
            self.assertEqual(settings.streaming_chunk_ms, 400)
            self.assertIn("streaming_chunk_ms = 400", text)
            self.assertNotIn("qwen_streaming", text)

    def test_incremental_transcription_defaults(self):
        settings = Settings()
        self.assertFalse(settings.incremental_transcription)
        self.assertEqual(settings.incremental_commit_threshold_s, 10.0)
        self.assertEqual(settings.streaming_chunk_ms, 500)

    def test_platform_helpers_have_defaults(self):
        runtime = FakeRuntime(system="Linux", home=Path("/home/user"))
        settings = Settings()
        self.assertIn("XF86TouchpadOff", active_hotkey(settings, runtime))
        self.assertIn("V", paste_shortcut(settings, runtime))

    def test_active_hotkey_uses_macos_binding_on_darwin(self):
        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"))
        settings = Settings()
        self.assertEqual(active_hotkey(settings, runtime), "Option+Space")
        self.assertNotIn("XF86TouchpadOff", active_hotkey(settings, runtime))
        self.assertEqual(paste_shortcut(settings, runtime), "Command+V")

    def test_active_hotkey_uses_windows_binding(self):
        runtime = FakeRuntime(system="Windows", home=Path("C:/Users/test"))
        settings = Settings()
        self.assertEqual(active_hotkey(settings, runtime), "ctrl+shift+space")
        self.assertEqual(paste_shortcut(settings, runtime), "Ctrl+V")

    def test_patch_settings_returns_new_object_without_mutating_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            write_default_settings(path)
            original = load_settings(path)

            updated = patch_settings(path, toggle_cooldown_ms=750)

            self.assertEqual(updated.toggle_cooldown_ms, 750)
            self.assertEqual(original.toggle_cooldown_ms, 500)
            self.assertEqual(load_settings(path).toggle_cooldown_ms, 750)

    def test_set_setting_rewrites_canonical_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            write_default_settings(path)

            updated = set_setting(path, "toggle_cooldown_ms", "750")

            self.assertEqual(updated.toggle_cooldown_ms, 750)
            self.assertEqual(load_settings(path).toggle_cooldown_ms, 750)
            self.assertIn("toggle_cooldown_ms = 750", path.read_text(encoding="utf-8"))

    def test_setting_type_coercion_and_unknown_field(self):
        self.assertIs(coerce_setting_value("cleanup_enabled", "false"), False)
        self.assertIs(coerce_setting_value("voice_model_cleanup_always_on", "on"), True)
        self.assertEqual(coerce_setting_value("sample_rate", "22050"), 22050)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            write_default_settings(path)
            with self.assertRaises(ValueError):
                set_setting(path, "wat", "true")


if __name__ == "__main__":
    unittest.main()
