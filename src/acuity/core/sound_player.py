# Plays default or custom .wav/.mp3 alert audio.
# ─────────────────────────────────────────────────────────────────
# QMediaPlayer-based audio playback for alert sounds.
#
# Supports:
#   • Per-category custom audio (distraction / fatigue_warning /
#     fatigue / rest_start / rest_end) configured via ConfigManager
#   • Default bundled sound files in assets/sounds/, one per category
#   • Safe rapid-fire: stops any in-progress playback before
#     starting the new clip
#
# Usage:
#   player = SoundPlayer()
#   player.play("fatigue")                      # default sound
#   player.play("distraction", Path("custom.wav"))  # custom file
#
# Thread: GUI.  QMediaPlayer lives on the thread that created it.
#
# Requires PySide6.QtMultimedia (part of PySide6).  Some machines hang
# in its backend, so tests/test_alert_manager.py runs as a separate,
# time-limited CI step.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from acuity.utils.logger import get_logger

log = get_logger(__name__)

# ── Resolve default sound directory ─────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (src/acuity/core/..)
_SOUNDS_DIR = _PROJECT_ROOT / "assets" / "sounds"

# Default sound file per alert category.  Looked up at play-time
# so missing files don't crash import.  Keys must match the category
# names in alert_manager._EVENT_CATEGORY.
#
# The bundled clips are deliberately distinct in pitch contour,
# rhythm and timbre so they are distinguishable without looking at the
# screen; see tools/generate_default_sounds.py for how they are made.
#
# rest_start / rest_end are not event categories — they are announced
# state transitions dispatched by AlertManager.announce_rest_* rather
# than by handle_event, so they have no _EVENT_CATEGORY entry.
_DEFAULT_SOUNDS: dict[str, str] = {
    "distraction": "distraction.wav",  # two quick rising blips
    "fatigue_warning": "fatigue_warning.wav",  # soft descending chime
    "fatigue": "fatigue.wav",  # three urgent rising beeps
    "rest_start": "rest_start.wav",  # rising major arpeggio
    "rest_end": "rest_end.wav",  # the same, inverted
}


class SoundPlayer(QObject):
    """QMediaPlayer wrapper for playing alert sound files.

    Creates a single QMediaPlayer + QAudioOutput pair and reuses
    them across calls to :meth:`play`.  Previous playback is stopped
    before each new play to avoid overlapping audio.

    Parameters
    ----------
    parent : QObject or None
        Qt parent.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)

        # Wire error signal for diagnostics.
        self._player.errorOccurred.connect(self._on_error)

        log.info("SoundPlayer initialised.")

    # ── Properties ──────────────────────────────────────────────

    @property
    def volume(self) -> float:
        """Current volume (0.0 – 1.0)."""
        return self._audio_output.volume()

    @volume.setter
    def volume(self, value: float) -> None:
        self._audio_output.setVolume(max(0.0, min(1.0, value)))

    @property
    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    # ── Public API ──────────────────────────────────────────────

    def play(
        self,
        category: str,
        custom_path: Path | None = None,
    ) -> None:
        """Play an alert sound for *category*.

        Parameters
        ----------
        category : str
            One of ``"distraction"``, ``"fatigue_warning"``,
            ``"fatigue"``, ``"rest_start"``, ``"rest_end"``.
        custom_path : Path or None
            If provided and the file exists, play this file instead
            of the default bundled sound.
        """
        sound_file = self._resolve_sound(category, custom_path)
        if sound_file is None:
            log.warning(
                "No sound file available for category '%s' — skipping.",
                category,
            )
            return

        # Stop any in-progress playback.
        if self.is_playing:
            self._player.stop()

        self._player.setSource(QUrl.fromLocalFile(str(sound_file)))
        self._player.play()
        log.debug("Playing sound: %s", sound_file.name)

    def stop(self) -> None:
        """Stop any in-progress playback."""
        if self.is_playing:
            self._player.stop()

    # ── Private helpers ─────────────────────────────────────────

    def _resolve_sound(
        self,
        category: str,
        custom_path: Path | None,
    ) -> Path | None:
        """Determine which sound file to play.

        Priority:
        1. *custom_path* if it exists and is a file.
        2. Default bundled file in assets/sounds/.
        3. None (no sound available).
        """
        if custom_path is not None and custom_path.is_file():
            return custom_path

        default_name = _DEFAULT_SOUNDS.get(category)
        if default_name is not None:
            default_path = _SOUNDS_DIR / default_name
            if default_path.is_file():
                return default_path

        return None

    def _on_error(
        self,
        error: QMediaPlayer.Error,
        message: str,
    ) -> None:
        """Log QMediaPlayer errors."""
        if error != QMediaPlayer.Error.NoError:
            log.error("SoundPlayer error: %s — %s", error.name, message)

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        state = "playing" if self.is_playing else "idle"
        return f"<SoundPlayer {state}>"
