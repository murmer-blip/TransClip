import unittest
from dataclasses import dataclass
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from transclip.recording_ops import toggle_recording
from transclip.settings import Settings


class FakeClient:
    base_url = "http://service"

    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error

    def record_toggle(self):
        if self.error:
            raise self.error
        return dict(self.result)


@dataclass
class FakePasteResult:
    copied: bool = True
    pasted: bool = False
    restored: bool = False
    transcript_left_on_clipboard: bool = True
    clipboard_backend: str = "fake-clipboard"
    paste_backend: str | None = None
    error_detail: str = "fake paste failed"


class RecordingOpsTests(unittest.TestCase):
    def setUp(self):
        self._log_patch = patch("transclip.recording_ops.append_toggle_log")
        self._log_patch.start()

    def tearDown(self):
        self._log_patch.stop()

    def test_service_unavailable_is_renderable_error(self):
        outcome = toggle_recording(
            Settings(),
            client=FakeClient(error=URLError("refused")),
        )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.notification_message, "TransClip service is not running.")

    def test_http_rejection_is_renderable_error(self):
        outcome = toggle_recording(
            Settings(),
            client=FakeClient(error=HTTPError("http://service", 500, "boom", None, None)),
        )

        self.assertFalse(outcome.ok)
        self.assertIn("HTTP 500", outcome.notification_message)

    def test_started_and_discarded_do_not_expose_latest_transcript(self):
        started = toggle_recording(Settings(), client=FakeClient({"action": "started", "status": "recording"}))
        discarded = toggle_recording(Settings(), client=FakeClient({"action": "discarded", "status": "ready"}))

        self.assertEqual(started.latest_transcript, "")
        self.assertEqual(discarded.latest_transcript, "")

    def test_stopped_paste_failure_carries_transcript_and_message(self):
        with patch("transclip.recording_ops.paste_transcript", return_value=FakePasteResult()):
            outcome = toggle_recording(
                Settings(paste_injection_delay_ms=0),
                paste=True,
                client=FakeClient({"action": "stopped", "status": "ready", "text": "Hello."}),
            )

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.latest_transcript, "Hello.")
        self.assertIn("still on the clipboard", outcome.notification_message)
        self.assertFalse(outcome.payload["paste"]["pasted"])

    def test_stopped_paste_waits_for_hotkey_modifiers_to_release(self):
        events = []

        def fake_sleep(seconds):
            events.append(("sleep", seconds))

        def fake_paste(_transcript, _settings):
            events.append(("paste", 0))
            return FakePasteResult(pasted=True, error_detail="")

        with (
            patch("transclip.recording_ops.time.sleep", side_effect=fake_sleep),
            patch("transclip.recording_ops.paste_transcript", side_effect=fake_paste),
        ):
            outcome = toggle_recording(
                Settings(paste_injection_delay_ms=300),
                paste=True,
                client=FakeClient({"action": "stopped", "status": "ready", "text": "Hello."}),
            )

        self.assertTrue(outcome.ok)
        # The delay is measured from the toggle keypress, so the sleep covers
        # the remainder of the 300ms window after the (instant) fake round-trip.
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0][0], "sleep")
        self.assertAlmostEqual(events[0][1], 0.3, delta=0.05)
        self.assertEqual(events[1], ("paste", 0))

    def test_stopped_paste_skips_sleep_when_round_trip_exceeds_delay(self):
        events = []

        def fake_paste(_transcript, _settings):
            events.append(("paste", 0))
            return FakePasteResult(pasted=True, error_detail="")

        clock = iter([0.0, 1.0])  # toggle start, then post-round-trip check

        with (
            patch("transclip.recording_ops.time.monotonic", side_effect=lambda: next(clock)),
            patch("transclip.recording_ops.time.sleep", side_effect=lambda s: events.append(("sleep", s))),
            patch("transclip.recording_ops.paste_transcript", side_effect=fake_paste),
        ):
            outcome = toggle_recording(
                Settings(paste_injection_delay_ms=300),
                paste=True,
                client=FakeClient({"action": "stopped", "status": "ready", "text": "Hello."}),
            )

        self.assertTrue(outcome.ok)
        self.assertEqual(events, [("paste", 0)])


if __name__ == "__main__":
    unittest.main()
