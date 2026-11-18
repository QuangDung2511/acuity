# QLabel showing webcam frames during calibration.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   VideoFeedWidget()
#   .push_qimage(img: QImage)       – display a QImage frame (primary)
#   .push_frame(frame: np.ndarray)  – display a BGR OpenCV frame (alt)
#   .show_calibration_view()        – enter live-feed mode
#   .show_tracking_active()         – switch to static indicator
#   .show_idle()                    – blank / "session not started"
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from acuity.utils.logger import get_logger

log = get_logger(__name__)

# Minimum display dimensions for the video panel.
_FEED_MIN_W = 320
_FEED_MIN_H = 180


class VideoFeedWidget(QLabel):
    """Displays webcam frames during calibration only.

    State machine (controlled by FocusTab / session wiring):
      • IDLE          – grey placeholder, shown before session starts.
      • CALIBRATING   – live BGR frames painted via push_frame().
      • TRACKING      – static "Tracking Active" text indicator.

    BGR→RGB conversion is done here; the caller passes raw OpenCV frames.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setMinimumSize(_FEED_MIN_W, _FEED_MIN_H)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("videoFeed")

        # Minimal styling — border matches the dark theme surface border.
        self.setStyleSheet(
            "QLabel#videoFeed {"
            "  background-color: #1A1A1F;"
            "  border: 1px solid #2C2C34;"
            "  border-radius: 6px;"
            "  color: #888895;"
            "  font-size: 13px;"
            "}"
        )

        self.show_idle()

    # ── Public state transitions ───────────────────────────────────

    def show_idle(self) -> None:
        """Reset to blank state (no session running)."""
        self.clear()
        self.setText("No active session")
        self._live = False

    def show_calibration_view(self) -> None:
        """Enter live-feed mode; push_frame() will render frames."""
        self.clear()
        self.setText("Calibrating…")
        self._live = True
        log.debug("VideoFeedWidget: calibration view active.")

    def show_tracking_active(self) -> None:
        """Switch to static indicator once calibration ends."""
        self.clear()
        self.setText("● Tracking Active")
        self.setStyleSheet(
            "QLabel#videoFeed {"
            "  background-color: #1A1A1F;"
            "  border: 1px solid #2C2C34;"
            "  border-radius: 6px;"
            "  color: #4CAF82;"  # success green
            "  font-size: 14px;"
            "  font-weight: 600;"
            "}"
        )
        self._live = False
        log.debug("VideoFeedWidget: showing 'Tracking Active'.")

    def show_camera_off(self, message: str = "Camera off") -> None:
        """Blank the feed while the camera is released (REST / AWAY).

        The pipeline is torn down on these transitions, so no further
        frames arrive.  Without an explicit clear the widget keeps
        painting the last frame it received, which reads as a frozen
        live feed rather than a released camera.
        """
        self.clear()
        self.setText(message)
        self.setStyleSheet(
            "QLabel#videoFeed {"
            "  background-color: #1A1A1F;"
            "  border: 1px solid #2C2C34;"
            "  border-radius: 6px;"
            "  color: #888895;"
            "  font-size: 13px;"
            "}"
        )
        self._live = False
        log.debug("VideoFeedWidget: camera off (%s).", message)

    # ── Frame ingestion ────────────────────────────────────────────

    def push_qimage(self, image: QImage) -> None:
        """Paint a QImage into the label (primary path).

        CVPipelineThread.frame_ready emits QImage directly, so this
        avoids a redundant numpy round-trip.

        Silently ignored when not in calibration (live) mode.
        """
        if not self._live:
            return
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image).scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def push_frame(self, frame: np.ndarray) -> None:
        """Paint a BGR OpenCV frame into the label (alternate path).

        Args:
            frame: H×W×3 uint8 array in BGR colour order.

        Silently ignored when not in calibration (live) mode.
        """
        if not self._live:
            return

        if frame is None or frame.size == 0:
            return

        try:
            # BGR → RGB
            rgb = frame[:, :, ::-1].copy()
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(
                self.width(),
                self.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(pixmap)
        except Exception:  # noqa: BLE001
            log.exception("VideoFeedWidget: failed to paint frame.")
