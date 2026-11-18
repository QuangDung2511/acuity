# Camera lifecycle manager (open / release / select device).
# ─────────────────────────────────────────────────────────────────
# Wraps cv2.VideoCapture with:
#   • Explicit open / release lifecycle for REST / AWAY states
#     (aggressive resource release)
#   • Graceful degradation on mid-session disconnect
#   • Device enumeration for the Settings tab dropdown
#
# Thread: CV worker.  This class is designed to be owned and called
# exclusively by the CV pipeline thread (QThread).  The GUI thread
# should only call the static `list_devices()` method.
#
# Every capture is opened through the system camera adapter
# (acuity.system.camera.open_source), which picks the OS backend and is
# the seam a test replaces to feed a synthetic or recorded source.  It
# is called through the module, not imported by name, so a patched
# open_source is seen here.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from acuity.system import camera as camera_adapter
from acuity.system.camera import FrameSource
from acuity.utils.logger import get_logger

log = get_logger(__name__)

# Maximum camera index to probe when scanning for devices
_MAX_PROBE_INDEX = 10


# ── Data structures ─────────────────────────────────────────────


@dataclass(frozen=True)
class CameraInfo:
    """Describes a detected camera device."""

    index: int
    width: int
    height: int
    fps: float
    backend: str

    def __str__(self) -> str:
        return (
            f"Camera {self.index}: {self.width}x{self.height} @ {self.fps:.0f} FPS ({self.backend})"
        )


class CameraError(Exception):
    """Raised when a camera operation fails in a non-recoverable way."""


# ── CameraManager ───────────────────────────────────────────────


class CameraManager:
    """Manages a single OpenCV VideoCapture lifecycle.

    Typical usage inside the CV pipeline thread::

        cam = CameraManager()
        cam.open(0)               # start of focus block
        while running:
            ok, frame = cam.read_frame()
            if not ok:
                # Camera lost mid-session: degrade gracefully, don't crash.
                break
            ...process frame...
        cam.release()             # rest / away / session end

    The OS capture backend comes from the system camera adapter.
    """

    def __init__(self) -> None:
        self._cap: FrameSource | None = None
        self._device_index: int = -1
        self._consecutive_failures: int = 0

    # ── Properties ──────────────────────────────────────────────

    @property
    def is_opened(self) -> bool:
        """True if the capture device is currently open and usable."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def device_index(self) -> int:
        """Return the index of the currently opened device (-1 if none)."""
        return self._device_index

    @property
    def frame_width(self) -> int:
        """Width of captured frames in pixels (0 if not opened)."""
        if not self.is_opened:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        """Height of captured frames in pixels (0 if not opened)."""
        if not self.is_opened:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """Reported FPS of the camera (0.0 if not opened)."""
        if not self.is_opened:
            return 0.0
        raw = self._cap.get(cv2.CAP_PROP_FPS)
        return raw if raw > 0 else 30.0  # fallback for cameras that report 0

    # ── Core lifecycle ──────────────────────────────────────────

    def open(self, device_index: int = 0) -> None:
        """Open the camera at *device_index*.

        If a different camera is already open it is released first.

        Raises
        ------
        CameraError
            If the device cannot be opened (bad index, hardware issue,
            driver problem, etc.).
        """
        # Release any existing capture before opening a new one
        if self.is_opened:
            if self._device_index == device_index:
                log.debug("Camera %d already open — skipping re-open.", device_index)
                return
            log.info("Switching camera %d → %d.", self._device_index, device_index)
            self.release()

        log.info("Opening camera index %d …", device_index)
        cap = camera_adapter.open_source(device_index)

        if not cap.isOpened():
            cap.release()
            if device_index != 0:
                log.warning(
                    "Failed to open camera index %d. Falling back to camera 0.", device_index
                )
                self.open(0)
                return

            msg = (
                f"Failed to open camera index {device_index}.  "
                "Check that the device is connected and not in use "
                "by another application."
            )
            log.error(msg)
            raise CameraError(msg)

        self._cap = cap
        self._device_index = device_index
        self._consecutive_failures = 0

        log.info(
            "Camera %d opened: %dx%d @ %.0f FPS (backend: %s).",
            device_index,
            self.frame_width,
            self.frame_height,
            self.fps,
            self._cap.getBackendName(),
        )

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame from the open camera.

        Returns
        -------
        (success, frame)
            ``success`` is True when a valid frame was captured.
            On failure, ``frame`` is None.

        Graceful degradation
        -----------------------------
        A single dropped frame is common and non-fatal — the method
        returns ``(False, None)`` and the caller can simply retry on
        the next tick.  After ``MAX_CONSECUTIVE_FAILURES`` consecutive
        failures the camera is considered disconnected and the method
        logs an error so the session manager can pause the session.
        """
        if not self.is_opened:
            return False, None

        ret, frame = self._cap.read()

        if ret:
            self._consecutive_failures = 0
            return True, frame

        # ── Frame read failed ───────────────────────────────────
        self._consecutive_failures += 1

        if self._consecutive_failures == 1:
            # First failure — likely a transient glitch
            log.debug("Camera %d: frame read failed (transient).", self._device_index)
        elif self._consecutive_failures <= 5:
            log.warning(
                "Camera %d: %d consecutive read failures.",
                self._device_index,
                self._consecutive_failures,
            )
        else:
            # Sustained failure — camera is likely disconnected
            log.error(
                "Camera %d: %d consecutive read failures — device appears disconnected.",
                self._device_index,
                self._consecutive_failures,
            )

        return False, None

    def release(self) -> None:
        """Release the camera resource immediately.

        Safe to call multiple times or when no camera is open.
        The camera is released aggressively: during REST, AWAY and
        on application exit.
        """
        if self._cap is not None:
            try:
                self._cap.release()
                log.info("Camera %d released.", self._device_index)
            except Exception as exc:
                # Even release() can fail if the driver is in a bad
                # state — we must not crash.
                log.error("Error releasing camera %d: %s", self._device_index, exc)
            finally:
                self._cap = None
                self._device_index = -1
                self._consecutive_failures = 0

    # ── Device enumeration ──────────────────────────────────────

    @staticmethod
    def list_devices(
        max_index: int = _MAX_PROBE_INDEX,
    ) -> list[CameraInfo]:
        """Scan for available camera devices.

        Probes indices ``0`` through ``max_index - 1`` and returns a
        :class:`CameraInfo` for each one that can be opened.

        This is used by the Settings tab dropdown.  It is a
        static method so it can be called without an active
        CameraManager instance.

        Parameters
        ----------
        max_index : int
            Upper bound (exclusive) on camera indices to probe.

        Returns
        -------
        list[CameraInfo]
            Sorted by device index.
        """
        devices: list[CameraInfo] = []

        for idx in range(max_index):
            cap = camera_adapter.open_source(idx)
            if cap.isOpened():
                info = CameraInfo(
                    index=idx,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                    backend=cap.getBackendName(),
                )
                devices.append(info)
                cap.release()

        log.info("Device scan found %d camera(s).", len(devices))
        for dev in devices:
            log.debug("  %s", dev)

        return devices

    # ── Helpers ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        state = "opened" if self.is_opened else "closed"
        return f"<CameraManager device={self._device_index} {state}>"

    def __del__(self) -> None:
        """Safety net — release on garbage collection."""
        self.release()
