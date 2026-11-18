# Tests for alert manager.
# ─────────────────────────────────────────────────────────────────
# Three layers:
#   1. Unit tests (SoundPlayer)   — resolution logic, volume, repr
#   2. Unit tests (AlertManager)  — dispatch routing for every
#      alert mode × event type, mock tray + sound
#   3. Integration                — real video + CV engines fire
#      events that AlertManager dispatches (no live camera).  Needs
#      the private video fixture, so it self-skips in CI.
#
# QtMultimedia can hang on some machines, so CI runs this file as its
# own time-limited step and the pre-push hook leaves it out.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from acuity.core.alert_manager import AlertManager
from acuity.core.cv_pipeline import EventData
from acuity.core.sound_player import SoundPlayer
from acuity.utils.constants import (
    ALERT_LOG_ONLY,
    ALERT_NOTIFICATION_AND_SOUND,
    ALERT_NOTIFICATION_ONLY,
    ALERT_SOUND_ONLY,
    EVENT_AWAY,
    EVENT_DISTRACTION_POSE,
    EVENT_DISTRACTION_WINDOW,
    EVENT_FATIGUE,
    EVENT_FATIGUE_WARNING,
)

# ── Paths ───────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "private" / "test_full_session.mp4"
_FACE_MODEL = _PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"
_POSE_MODEL = _PROJECT_ROOT / "assets" / "models" / "pose_landmarker.task"

_HAS_VIDEO = _VIDEO_PATH.exists()
_HAS_MODELS = _FACE_MODEL.exists() and _POSE_MODEL.exists()


# ── Helpers ─────────────────────────────────────────────────────


def _make_event(
    event_type: str = EVENT_DISTRACTION_POSE,
    reason: str = "",
    duration: float = 5.0,
) -> EventData:
    return EventData(
        event_type=event_type,
        reason=reason,
        duration=duration,
        timestamp=time.time(),
    )


@pytest.fixture()
def mock_config():
    """Patch ConfigManager inside alert_manager with deterministic defaults."""
    with patch("acuity.core.alert_manager.ConfigManager") as MockCfg:
        cfg = MagicMock()
        _defaults = {
            "alert_distraction": ALERT_LOG_ONLY,
            "alert_fatigue": ALERT_NOTIFICATION_AND_SOUND,
            "alert_fatigue_warning": ALERT_NOTIFICATION_ONLY,
            "audio_distraction": "",
            "audio_fatigue": "",
            "audio_fatigue_warning": "",
            # Not alert categories — rest start/end are announced state
            # changes, but they resolve a custom audio path the same way.
            "audio_rest_start": "",
            "audio_rest_end": "",
        }
        cfg.get_str.side_effect = lambda k: str(_defaults[k])
        cfg.get_path.side_effect = lambda k: Path(_defaults[k]) if _defaults[k] else None
        cfg._defaults = _defaults
        MockCfg.return_value = cfg
        yield cfg


@pytest.fixture()
def mock_tray():
    """Create a mock QSystemTrayIcon."""
    tray = MagicMock()
    tray.isSystemTrayAvailable.return_value = True
    tray.showMessage = MagicMock()
    return tray


@pytest.fixture()
def alert_mgr(qtbot, mock_config, mock_tray):
    """Create an AlertManager with mocked config and tray."""
    mgr = AlertManager(tray_icon=mock_tray)
    # Replace sound player with a mock to avoid actual audio.
    mgr._sound_player = MagicMock(spec=SoundPlayer)
    yield mgr


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS — SoundPlayer
# ═══════════════════════════════════════════════════════════════


class TestSoundPlayerUnit:
    """Tests for SoundPlayer in isolation."""

    def test_initial_state(self, qtbot):
        player = SoundPlayer()
        assert not player.is_playing
        assert "idle" in repr(player)

    def test_volume_property(self, qtbot):
        player = SoundPlayer()
        player.volume = 0.5
        assert player.volume == pytest.approx(0.5, abs=0.01)

    def test_volume_clamped_high(self, qtbot):
        player = SoundPlayer()
        player.volume = 2.0
        assert player.volume <= 1.0

    def test_volume_clamped_low(self, qtbot):
        player = SoundPlayer()
        player.volume = -1.0
        assert player.volume >= 0.0

    def test_play_with_no_sound_file_does_not_crash(self, qtbot):
        """play() with an unknown category and no default file must
        not crash (intent: graceful degradation when assets are
        missing — the user just doesn't hear anything)."""
        player = SoundPlayer()
        player.play("nonexistent_category")
        assert not player.is_playing

    def test_custom_path_that_does_not_exist_falls_back_to_default(self, qtbot):
        """Custom path that doesn't exist must fall through to the
        bundled default (intent: bad config shouldn't crash the app,
        and the user still hears the category's own sound)."""
        player = SoundPlayer()
        resolved = player._resolve_sound("fatigue", Path("/nonexistent/fake.wav"))
        assert resolved is not None
        assert resolved.name == "fatigue.wav"

    def test_every_alert_category_has_a_bundled_default(self, qtbot):
        """Each alert category the AlertManager can dispatch must
        resolve to a shipped sound file, and the four must be distinct
        (intent: no category is silently mute out of the box)."""
        from acuity.core.alert_manager import _EVENT_CATEGORY

        player = SoundPlayer()
        resolved = {
            category: player._resolve_sound(category, None)
            for category in set(_EVENT_CATEGORY.values())
        }
        missing = [c for c, path in resolved.items() if path is None]
        assert not missing, f"categories with no default sound: {missing}"
        names = [path.name for path in resolved.values()]
        assert len(set(names)) == len(names), f"duplicate sounds: {names}"

    def test_play_with_valid_custom_path(self, qtbot, tmp_path):
        """play() with a valid custom file must set it as the source
        (intent: custom audio preference is respected)."""
        # Create a tiny dummy WAV file.
        dummy = tmp_path / "custom_alert.wav"
        dummy.write_bytes(b"RIFF" + b"\x00" * 40)  # minimal header

        player = SoundPlayer()
        player.play("fatigue", dummy)
        # Player attempted to play — verify source was set.
        source = player._player.source()
        assert "custom_alert.wav" in source.toLocalFile()

    def test_stop_when_not_playing_is_noop(self, qtbot):
        player = SoundPlayer()
        player.stop()  # should not crash
        assert not player.is_playing

    def test_resolve_prefers_custom_over_default(self, qtbot, tmp_path):
        """_resolve_sound must prefer custom_path over the default
        bundled file (intent: user-set custom audio takes priority)."""
        custom = tmp_path / "my_alert.mp3"
        custom.write_bytes(b"\x00" * 10)

        player = SoundPlayer()
        result = player._resolve_sound("fatigue", custom)
        assert result == custom

    def test_resolve_returns_none_when_nothing_available(self, qtbot):
        """_resolve_sound must return None when no custom file and no
        default file exists (intent: no crash, just no sound)."""
        player = SoundPlayer()
        result = player._resolve_sound("unknown_category", None)
        assert result is None


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS — AlertManager dispatch logic
# ═══════════════════════════════════════════════════════════════


class TestAlertManagerDispatch:
    """Tests for handle_event routing based on alert mode config."""

    # ── Log Only (default for distraction) ──────────────────────

    def test_log_only_does_not_notify_or_play(self, alert_mgr, mock_tray):
        """log_only mode must not send a notification or play a sound
        (intent: minimal disturbance — event is captured in logs only,
        the default for distractions)."""
        event = _make_event(EVENT_DISTRACTION_POSE)
        alert_mgr.handle_event(event)

        mock_tray.showMessage.assert_not_called()
        alert_mgr._sound_player.play.assert_not_called()

    def test_log_only_applies_to_both_distraction_types(self, alert_mgr, mock_tray):
        """Both Distraction_Pose and Distraction_Window must route
        through the same 'distraction' alert category (intent:
        single config key controls both triggers)."""
        for event_type in (EVENT_DISTRACTION_POSE, EVENT_DISTRACTION_WINDOW):
            event = _make_event(event_type)
            alert_mgr.handle_event(event)

        mock_tray.showMessage.assert_not_called()
        alert_mgr._sound_player.play.assert_not_called()

    # ── Notification Only ───────────────────────────────────────

    def test_notification_only_sends_toast_no_sound(self, alert_mgr, mock_config, mock_tray):
        """notification_only mode must send a toast but not play audio
        (intent: visual-only alert for users who prefer silence)."""
        mock_config._defaults["alert_fatigue"] = ALERT_NOTIFICATION_ONLY
        event = _make_event(EVENT_FATIGUE, reason="microsleep", duration=0.5)
        alert_mgr.handle_event(event)

        mock_tray.showMessage.assert_called_once()
        alert_mgr._sound_player.play.assert_not_called()

    def test_notification_title_and_body(self, alert_mgr, mock_config, mock_tray):
        """Toast notification must include a meaningful title and body
        (intent: user can understand the alert at a glance)."""
        mock_config._defaults["alert_fatigue"] = ALERT_NOTIFICATION_ONLY
        event = _make_event(EVENT_FATIGUE, reason="microsleep", duration=0.5)
        alert_mgr.handle_event(event)

        call_args = mock_tray.showMessage.call_args
        title = call_args[0][0]
        body = call_args[0][1]
        assert "Fatigue" in title
        assert "microsleep" in body

    def test_window_toast_names_the_app_and_duration(self, alert_mgr, mock_config, mock_tray):
        """A window distraction says which app pulled the user away and
        for how long (intent: the user can tell it from a head-pose alert
        at a glance, and the reason is only ever the app's file name)."""
        mock_config._defaults["alert_distraction"] = ALERT_NOTIFICATION_ONLY
        event = _make_event(EVENT_DISTRACTION_WINDOW, reason="chat.exe", duration=30.0)
        alert_mgr.handle_event(event)

        title, body = mock_tray.showMessage.call_args[0][:2]
        assert title == "Distraction — Window"
        assert body == "Away from your focus apps for 30s (chat.exe)."

    # ── Sound Only ──────────────────────────────────────────────

    def test_sound_only_plays_audio_no_toast(self, alert_mgr, mock_config, mock_tray):
        """sound_only mode must play audio but not send a toast
        (intent: audio-only alert doesn't clutter the screen)."""
        mock_config._defaults["alert_distraction"] = ALERT_SOUND_ONLY
        event = _make_event(EVENT_DISTRACTION_WINDOW, reason="chat.exe", duration=30.0)
        alert_mgr.handle_event(event)

        alert_mgr._sound_player.play.assert_called_once_with("distraction", None)
        mock_tray.showMessage.assert_not_called()

    # ── Notification + Sound (default for fatigue) ──────────────

    def test_notification_and_sound_does_both(self, alert_mgr, mock_config, mock_tray):
        """notification_and_sound mode must both send a toast AND play
        audio (intent: maximum urgency for fatigue alerts)."""
        event = _make_event(EVENT_FATIGUE, reason="microsleep", duration=0.5)
        alert_mgr.handle_event(event)

        mock_tray.showMessage.assert_called_once()
        alert_mgr._sound_player.play.assert_called_once_with("fatigue", None)

    # ── Away events (non-alertable) ─────────────────────────────

    def test_away_event_is_log_only(self, alert_mgr, mock_tray):
        """Away events must not trigger any alert (intent: Away has
        its own modal dialog handled by SessionManager, not
        AlertManager)."""
        event = _make_event(EVENT_AWAY, duration=20.0)
        alert_mgr.handle_event(event)

        mock_tray.showMessage.assert_not_called()
        alert_mgr._sound_player.play.assert_not_called()

    # ── Config changes take effect immediately ──────────────────

    def test_config_change_takes_effect_immediately(self, alert_mgr, mock_config, mock_tray):
        """Changing alert mode mid-session must affect the next event
        (intent: Settings changes are live, no restart required)."""
        # Start with log_only for distraction — no sound.
        event1 = _make_event(EVENT_DISTRACTION_POSE)
        alert_mgr.handle_event(event1)
        alert_mgr._sound_player.play.assert_not_called()

        # Switch to sound_only — next event should play.
        mock_config._defaults["alert_distraction"] = ALERT_SOUND_ONLY
        event2 = _make_event(EVENT_DISTRACTION_POSE)
        alert_mgr.handle_event(event2)
        alert_mgr._sound_player.play.assert_called_once()

    # ── No tray icon (headless mode) ────────────────────────────

    def test_notification_skipped_when_no_tray(self, qtbot, mock_config):
        """When tray_icon is None, notification dispatch must be
        silently skipped (intent: headless testing / pre-init)."""
        mock_config._defaults["alert_fatigue"] = ALERT_NOTIFICATION_ONLY
        mgr = AlertManager(tray_icon=None)
        mgr._sound_player = MagicMock(spec=SoundPlayer)

        event = _make_event(EVENT_FATIGUE, reason="microsleep")
        mgr.handle_event(event)
        # No crash, no notification — just silently skipped.

    # ── Tray icon setter ────────────────────────────────────────

    def test_tray_icon_setter(self, qtbot, mock_config):
        """tray_icon property must be settable after construction
        (intent: tray may be created after AlertManager)."""
        mgr = AlertManager(tray_icon=None)
        assert mgr.tray_icon is None

        new_tray = MagicMock()
        mgr.tray_icon = new_tray
        assert mgr.tray_icon is new_tray

    # ── Custom audio path ───────────────────────────────────────

    def test_custom_audio_path_passed_to_sound_player(self, alert_mgr, mock_config, tmp_path):
        """When a custom audio file is configured and exists, it must
        be passed to SoundPlayer.play() (intent: custom audio
        support)."""
        custom_file = tmp_path / "my_fatigue_sound.wav"
        custom_file.write_bytes(b"RIFF" + b"\x00" * 40)

        mock_config._defaults["audio_fatigue"] = str(custom_file)
        mock_config.get_path.side_effect = lambda k: (
            Path(mock_config._defaults[k]) if mock_config._defaults.get(k) else None
        )

        event = _make_event(EVENT_FATIGUE, reason="microsleep", duration=0.5)
        alert_mgr.handle_event(event)

        alert_mgr._sound_player.play.assert_called_once_with(
            "fatigue",
            custom_file,
        )

    def test_invalid_custom_audio_passes_none(self, alert_mgr, mock_config):
        """When the custom audio path doesn't exist, None must be
        passed to SoundPlayer (intent: falls through to default
        sound gracefully)."""
        mock_config._defaults["audio_fatigue"] = "/nonexistent/file.wav"
        mock_config.get_path.side_effect = lambda k: (
            Path(mock_config._defaults[k]) if mock_config._defaults.get(k) else None
        )

        event = _make_event(EVENT_FATIGUE, reason="microsleep", duration=0.5)
        alert_mgr.handle_event(event)

        alert_mgr._sound_player.play.assert_called_once_with("fatigue", None)

    # ── Repr ────────────────────────────────────────────────────

    def test_repr(self, alert_mgr):
        r = repr(alert_mgr)
        assert "AlertManager" in r
        assert "tray=yes" in r


class TestRestAnnouncements:
    """Rest start/end are state changes, not advice.

    Both bypass the per-category alert_* preferences on purpose: the
    session has already switched, and a user who misses the cue keeps
    working through a break (start) or overruns it (end).
    """

    def test_rest_end_sends_toast_and_sound(self, alert_mgr, mock_tray):
        alert_mgr.announce_rest_end()

        mock_tray.showMessage.assert_called_once()
        alert_mgr._sound_player.play.assert_called_once_with("rest_end", None)

    def test_rest_end_ignores_alert_preferences(self, alert_mgr, mock_config, mock_tray):
        """Silencing every configurable category must not silence this
        (intent: alert_* governs warnings; this is a state change)."""
        for key in (
            "alert_distraction",
            "alert_fatigue",
            "alert_fatigue_warning",
        ):
            mock_config._defaults[key] = ALERT_LOG_ONLY

        alert_mgr.announce_rest_end()

        mock_tray.showMessage.assert_called_once()
        alert_mgr._sound_player.play.assert_called_once_with("rest_end", None)

    def test_rest_end_survives_missing_tray(self, alert_mgr, mock_config):
        """No tray must not stop the sound (intent: the audio cue is the
        half that reaches a user who has walked away from the screen)."""
        alert_mgr.tray_icon = None
        alert_mgr.announce_rest_end()

        alert_mgr._sound_player.play.assert_called_once_with("rest_end", None)

    def test_rest_end_uses_a_different_sound_from_rest_start(self, alert_mgr, mock_tray):
        """The two are heard in the same place and mean opposite things
        (intent: distinguishable without looking at the screen)."""
        alert_mgr.announce_rest_start(block_number=1, rest_seconds=300)
        alert_mgr.announce_rest_end()

        played = [c.args[0] for c in alert_mgr._sound_player.play.call_args_list]
        assert played == ["rest_start", "rest_end"]

    def test_rest_start_sends_toast_and_sound(self, alert_mgr, mock_tray):
        alert_mgr.announce_rest_start(block_number=2, rest_seconds=300)

        mock_tray.showMessage.assert_called_once()
        alert_mgr._sound_player.play.assert_called_once_with("rest_start", None)


class TestAlertManagerAllEventTypes:
    """Verify every alertable event type routes to the correct category."""

    @pytest.mark.parametrize(
        "event_type, expected_category",
        [
            (EVENT_DISTRACTION_POSE, "distraction"),
            (EVENT_DISTRACTION_WINDOW, "distraction"),
            (EVENT_FATIGUE, "fatigue"),
            (EVENT_FATIGUE_WARNING, "fatigue_warning"),
        ],
    )
    def test_event_type_maps_to_category(
        self,
        qtbot,
        mock_config,
        mock_tray,
        event_type,
        expected_category,
    ):
        """Each event type must map to the correct alert category
        (intent: config key lookup uses the right category)."""
        # Force sound_only so we can verify the category in the
        # play() call argument.
        mock_config._defaults[f"alert_{expected_category}"] = ALERT_SOUND_ONLY
        mgr = AlertManager(tray_icon=mock_tray)
        mgr._sound_player = MagicMock(spec=SoundPlayer)

        event = _make_event(event_type, reason="test", duration=5.0)
        mgr.handle_event(event)

        mgr._sound_player.play.assert_called_once()
        call_args = mgr._sound_player.play.call_args
        assert call_args[0][0] == expected_category


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION — video footage + CV engines + AlertManager
# ═══════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not (_HAS_VIDEO and _HAS_MODELS),
    reason="Requires test_full_session.mp4 and MediaPipe model files.",
)
class TestAlertManagerWithVideoFootage:
    """Feed recorded footage through real CV analyzers, trigger
    engine events, and verify AlertManager dispatches correctly.

    No live camera is used — CameraManager reads from video.
    """

    def test_distraction_events_from_video(self, qtbot, mock_config):
        """Feed video through face mesh + distraction engine with a
        low threshold, then verify AlertManager handles the resulting
        events correctly with the configured alert mode.

        Intent: end-to-end proof that real CV output → engine events
        → AlertManager dispatch works with recorded footage.
        """
        import cv2

        from acuity.core.calibration import CalibrationRunner
        from acuity.core.distraction_engine import DistractionEngine
        from acuity.core.face_mesh import FaceMeshAnalyzer
        from acuity.core.pose_tracker import PoseAnalyzer
        from acuity.core.presence import determine_presence

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened(), f"Cannot open test video: {_VIDEO_PATH}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Skip 3s pre-roll.
        for _ in range(int(3 * fps)):
            cap.read()

        face_analyzer = FaceMeshAnalyzer()
        pose_analyzer = PoseAnalyzer()

        try:
            # ── Phase 1: Calibrate from video ───────────────────
            # 1.0 s, not 2.0 s: calibration runs on the wall clock, and without
            # the pose model each frame is fast enough that 5 s of video can
            # be read in under 2 s (as in test_distraction_engine.py).
            cal_runner = CalibrationRunner(duration_s=1.0)
            frames = 0
            while not cal_runner.is_complete and frames < int(5 * fps):
                ret, frame = cap.read()
                if not ret:
                    break
                face = face_analyzer.process(frame)
                cal_runner.feed(face)
                frames += 1

            assert cal_runner.is_complete
            cal = cal_runner.result

            # ── Phase 2: Track with distraction engine ──────────
            events_collected = []

            def on_distraction(event):
                events_collected.append(event)

            engine = DistractionEngine(
                baseline_pitch=cal.baseline_pitch,
                baseline_yaw=cal.baseline_yaw,
                head_pose_threshold_s=2.0,  # low threshold
                callback=on_distraction,
            )

            for _ in range(int(5 * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                face = face_analyzer.process(frame)
                pose = pose_analyzer.process(frame)
                presence = determine_presence(face, pose)
                engine.update(face, presence)

            # ── Phase 3: Feed events through AlertManager ───────
            # Set distraction to sound_only so we can verify dispatch.
            mock_config._defaults["alert_distraction"] = ALERT_SOUND_ONLY

            mgr = AlertManager(tray_icon=None)
            mgr._sound_player = MagicMock(spec=SoundPlayer)

            for engine_event in events_collected:
                event_data = EventData(
                    event_type=engine_event.event_type,
                    reason="",
                    duration=engine_event.duration,
                    timestamp=engine_event.timestamp,
                )
                mgr.handle_event(event_data)

            # If any distraction events fired, sound should have played.
            if events_collected:
                assert mgr._sound_player.play.call_count == len(events_collected)
                for call in mgr._sound_player.play.call_args_list:
                    assert call[0][0] == "distraction"
            # If no events fired (user was focused in the video),
            # that's also valid — no dispatch is expected.

        finally:
            face_analyzer.close()
            pose_analyzer.close()
            cap.release()

    def test_fatigue_events_from_video(self, qtbot, mock_config):
        """Feed video through face mesh + fatigue engine, then verify
        AlertManager dispatches notification + sound for fatigue events.

        Intent: prove fatigue detection → AlertManager toast+sound
        dispatch works with real footage.
        """
        import cv2

        from acuity.core.calibration import CalibrationRunner
        from acuity.core.face_mesh import FaceMeshAnalyzer
        from acuity.core.fatigue_engine import FatigueEngine
        from acuity.core.pose_tracker import PoseAnalyzer
        from acuity.core.presence import determine_presence

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for _ in range(int(3 * fps)):
            cap.read()

        face_analyzer = FaceMeshAnalyzer()
        pose_analyzer = PoseAnalyzer()

        try:
            # ── Calibrate ───────────────────────────────────────
            # 1.0 s, not 2.0 s: calibration runs on the wall clock, and without
            # the pose model each frame is fast enough that 5 s of video can
            # be read in under 2 s (as in test_distraction_engine.py).
            cal_runner = CalibrationRunner(duration_s=1.0)
            frames = 0
            while not cal_runner.is_complete and frames < int(5 * fps):
                ret, frame = cap.read()
                if not ret:
                    break
                face = face_analyzer.process(frame)
                cal_runner.feed(face)
                frames += 1

            assert cal_runner.is_complete
            cal = cal_runner.result

            # ── Track with fatigue engine ───────────────────────
            fatigue_events = []

            def on_fatigue(event):
                fatigue_events.append(event)

            engine = FatigueEngine(
                baseline_ear=cal.baseline_ear,
                callback=on_fatigue,
            )

            for _ in range(int(5 * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                face = face_analyzer.process(frame)
                pose = pose_analyzer.process(frame)
                presence = determine_presence(face, pose)
                engine.update(face, presence)

            # ── Dispatch through AlertManager ───────────────────
            mock_tray = MagicMock()
            mock_tray.isSystemTrayAvailable.return_value = True

            mgr = AlertManager(tray_icon=mock_tray)
            mgr._sound_player = MagicMock(spec=SoundPlayer)

            for engine_event in fatigue_events:
                event_data = EventData(
                    event_type=engine_event.event_type,
                    reason=engine_event.reason,
                    duration=engine_event.duration,
                    timestamp=engine_event.timestamp,
                )
                mgr.handle_event(event_data)

            # Default fatigue alert = notification_and_sound.
            if fatigue_events:
                assert mock_tray.showMessage.call_count == len(fatigue_events)
                assert mgr._sound_player.play.call_count == len(fatigue_events)
                for call in mgr._sound_player.play.call_args_list:
                    assert call[0][0] == "fatigue"

        finally:
            face_analyzer.close()
            pose_analyzer.close()
            cap.release()

    def test_mixed_events_from_full_pipeline(self, qtbot, mock_config):
        """Run both engines against video footage simultaneously,
        collect all events, then verify AlertManager routes each event
        to the correct dispatch path.

        Intent: full end-to-end integration — all engines + AlertManager
        working together with recorded footage.
        """
        import cv2

        from acuity.core.calibration import CalibrationRunner
        from acuity.core.distraction_engine import DistractionEngine
        from acuity.core.face_mesh import FaceMeshAnalyzer
        from acuity.core.fatigue_engine import FatigueEngine
        from acuity.core.pose_tracker import PoseAnalyzer
        from acuity.core.presence import determine_presence

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for _ in range(int(3 * fps)):
            cap.read()

        face_analyzer = FaceMeshAnalyzer()
        pose_analyzer = PoseAnalyzer()

        try:
            # ── Calibrate ───────────────────────────────────────
            # 1.0 s, not 2.0 s: calibration runs on the wall clock, and without
            # the pose model each frame is fast enough that 5 s of video can
            # be read in under 2 s (as in test_distraction_engine.py).
            cal_runner = CalibrationRunner(duration_s=1.0)
            frames = 0
            while not cal_runner.is_complete and frames < int(5 * fps):
                ret, frame = cap.read()
                if not ret:
                    break
                face = face_analyzer.process(frame)
                cal_runner.feed(face)
                frames += 1

            assert cal_runner.is_complete
            cal = cal_runner.result

            # ── Build all engines ───────────────────────────────
            all_events = []

            distraction_engine = DistractionEngine(
                baseline_pitch=cal.baseline_pitch,
                baseline_yaw=cal.baseline_yaw,
                head_pose_threshold_s=2.0,
                callback=lambda e: all_events.append(
                    EventData(e.event_type, "", e.duration, e.timestamp)
                ),
            )
            fatigue_engine = FatigueEngine(
                baseline_ear=cal.baseline_ear,
                callback=lambda e: all_events.append(
                    EventData(e.event_type, e.reason, e.duration, e.timestamp)
                ),
            )

            # ── Process frames through all engines ──────────────
            for _ in range(int(5 * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                face = face_analyzer.process(frame)
                pose = pose_analyzer.process(frame)
                presence = determine_presence(face, pose)

                distraction_engine.update(face, presence)
                fatigue_engine.update(face, presence)

            # ── Configure AlertManager: each category differently ─
            mock_config._defaults["alert_distraction"] = ALERT_LOG_ONLY
            mock_config._defaults["alert_fatigue"] = ALERT_NOTIFICATION_AND_SOUND

            mock_tray = MagicMock()
            mock_tray.isSystemTrayAvailable.return_value = True

            mgr = AlertManager(tray_icon=mock_tray)
            mgr._sound_player = MagicMock(spec=SoundPlayer)

            # ── Dispatch all events ─────────────────────────────
            for event in all_events:
                mgr.handle_event(event)

            # ── Verify dispatch correctness ─────────────────────
            # Distraction = log_only → no sound, no toast for those.
            # Fatigue = notification+sound → both toast and sound.
            fatigue_count = sum(1 for e in all_events if e.event_type == EVENT_FATIGUE)
            expected_toasts = fatigue_count
            expected_sounds = fatigue_count

            assert mock_tray.showMessage.call_count == expected_toasts
            assert mgr._sound_player.play.call_count == expected_sounds

            # Regardless of events fired, no crash occurred and
            # dispatch logic was exercised end-to-end.

        finally:
            face_analyzer.close()
            pose_analyzer.close()
            cap.release()
