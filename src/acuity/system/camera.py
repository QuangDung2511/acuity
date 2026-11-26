# Camera backend and frame source: how OpenCV opens a capture.
# ─────────────────────────────────────────────────────────────────
# Responsibilities:
#   • create_camera(): the capture backend for this OS (DirectShow on
#     Windows, AVFoundation on macOS, V4L2 on Linux), with an `available`
#     flag and a human-readable `reason` that Settings shows.
#   • default_backend(): that backend, cached, for open_source().
#   • open_source(spec): open a frame source from a camera index or a
#     video file path.
#
# Thread: any.  The returned source belongs to the thread that reads it
# (the CV worker).
#
# Design:
#   • Every backend is a constant in OpenCV itself, so create_camera()
#     imports no OS-specific module; it only picks by sys.platform.
#   • Backend support is read from this OpenCV build (videoio_registry),
#     without opening a camera.  Whether a physical camera is plugged in
#     is the device scan's job.  The Windows wheel in use includes
#     DirectShow; the macOS and Linux wheels are expected to include
#     theirs but are untested here.
#   • An unmapped OS, or a build without the native backend (a distro
#     OpenCV without V4L2, say), gets OpenCV's automatic choice rather
#     than a refusal: another backend can usually still reach the
#     camera.  `reason` and the log flag the fallback.  `available` is
#     False only when the build has no camera backend at all.
#   • The frame source is the test seam.  The app opens its camera only
#     through open_source(), so a test can replace it with a synthetic
#     or recorded source and drive the real pipeline headless.
#   • FrameSource is the part of cv2.VideoCapture the camera manager
#     uses.  A real VideoCapture satisfies it.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from acuity.utils.logger import get_logger

log = get_logger(__name__)


class FrameSource(Protocol):
    """Where the pipeline's frames come from: a webcam or a video file.

    The part of ``cv2.VideoCapture`` the camera manager uses, so a real
    VideoCapture satisfies it.  A stand-in (synthetic frames, recorded
    footage) implements these five methods and nothing else.
    """

    def isOpened(self) -> bool:
        """True while frames can be read; False if opening failed or
        after ``release()``."""
        ...

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return ``(True, frame)`` with a BGR ``uint8`` image of shape
        ``(height, width, 3)``, or ``(False, None)`` when no frame came
        (end of file, camera unplugged, released)."""
        ...

    def release(self) -> None:
        """Free the device or file.  Safe to call more than once."""
        ...

    def get(self, prop_id: int) -> float:
        """Read a ``cv2.CAP_PROP_*`` property: the manager asks for frame
        width, height and FPS.  0.0 when unknown."""
        ...

    def getBackendName(self) -> str:
        """Name of the capture backend, for logs and the device list."""
        ...


# Native capture backend per sys.platform, with the name Settings shows.
_NATIVE_BACKENDS: dict[str, tuple[int, str]] = {
    "win32": (cv2.CAP_DSHOW, "DirectShow"),
    "darwin": (cv2.CAP_AVFOUNDATION, "AVFoundation"),
    "linux": (cv2.CAP_V4L2, "V4L2"),
}


@dataclass(frozen=True)
class CameraAdapter:
    """The capture backend for this OS, and whether OpenCV can use it."""

    backend: int
    available: bool
    reason: str


def create_camera() -> CameraAdapter:
    """Choose the camera backend for ``sys.platform``.

    Opens no camera.  Falls back to OpenCV's automatic choice, with a
    logged warning, when the OS is unmapped or its native backend is not
    in this OpenCV build.
    """
    compiled = cv2.videoio_registry.getCameraBackends()
    if not compiled:
        reason = "This OpenCV build has no camera support, so no camera can open."
        log.warning(reason)
        return CameraAdapter(backend=cv2.CAP_ANY, available=False, reason=reason)
    if sys.platform not in _NATIVE_BACKENDS:
        fallback = f"No camera backend mapped for {sys.platform}"
    else:
        backend, name = _NATIVE_BACKENDS[sys.platform]
        if backend in compiled:
            return CameraAdapter(
                backend=backend, available=True, reason=f"Camera opens with {name}."
            )
        fallback = f"This OpenCV build has no {name} support"
    reason = f"{fallback}; using OpenCV's automatic choice."
    log.warning(reason)
    return CameraAdapter(backend=cv2.CAP_ANY, available=True, reason=reason)


@functools.cache
def default_backend() -> int:
    """Return the OpenCV capture backend for this OS (cached)."""
    return create_camera().backend


def open_source(spec: int | str | os.PathLike[str]) -> FrameSource:
    """Open a frame source: a camera index, or the path of a video file.

    An ``int`` opens that webcam with :func:`default_backend`; a ``str``
    or path opens the video file with OpenCV's automatic choice.  The
    source may fail to open; callers check ``isOpened()``.

    Every capture the app makes goes through this function, looked up as
    ``acuity.system.camera.open_source`` at call time.  A test drives the
    real pipeline without a webcam by replacing it::

        monkeypatch.setattr(camera, "open_source", lambda spec: MySource())

    ``tests/test_session_e2e.py`` does this with synthetic frames.
    """
    if isinstance(spec, int):
        return cv2.VideoCapture(spec, default_backend())
    return cv2.VideoCapture(os.fspath(spec))
