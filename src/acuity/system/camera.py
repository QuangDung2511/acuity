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
#   • `available` means this OpenCV build includes the backend
#     (videoio_registry), checked without opening a camera.  Whether a
#     physical camera is plugged in is the device scan's job.  The
#     Windows wheel in use includes DirectShow; the macOS and Linux
#     wheels are expected to include theirs but are untested here.
#   • An unmapped OS gets OpenCV's automatic choice, flagged in `reason`
#     and the log rather than refused: its camera may still work.
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
    """What the camera manager needs from a capture (cv2.VideoCapture)."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...

    def get(self, prop_id: int) -> float: ...

    def getBackendName(self) -> str: ...


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

    Opens no camera.
    """
    if sys.platform not in _NATIVE_BACKENDS:
        reason = f"No camera backend mapped for {sys.platform}; using OpenCV's automatic choice."
        log.warning(reason)
        return CameraAdapter(backend=cv2.CAP_ANY, available=True, reason=reason)
    backend, name = _NATIVE_BACKENDS[sys.platform]
    if backend not in cv2.videoio_registry.getCameraBackends():
        reason = f"This OpenCV build has no {name} support, so the camera cannot open."
        log.warning(reason)
        return CameraAdapter(backend=backend, available=False, reason=reason)
    return CameraAdapter(backend=backend, available=True, reason=f"Camera opens with {name}.")


@functools.cache
def default_backend() -> int:
    """Return the OpenCV capture backend for this OS (cached)."""
    return create_camera().backend


def open_source(spec: int | str | os.PathLike[str]) -> FrameSource:
    """Open a frame source: a camera index, or the path of a video file.

    The source may fail to open; callers check ``isOpened()``.
    """
    if isinstance(spec, int):
        return cv2.VideoCapture(spec, default_backend())
    return cv2.VideoCapture(os.fspath(spec))
