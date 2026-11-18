# Camera backend and frame source: how OpenCV opens a capture.
# ─────────────────────────────────────────────────────────────────
# Responsibilities:
#   • default_backend(): the OpenCV capture backend for this OS.
#   • open_source(spec): open a frame source from a camera index or a
#     video file path.
#
# Thread: any.  The returned source belongs to the thread that reads it
# (the CV worker).
#
# Design:
#   • Minimal, Windows-tested adapter (ticket 03).  Ticket 07 maps the
#     macOS and Linux backends and adds `available` / `reason` and the
#     create_*() factory the package header describes.  Until then any
#     other OS gets OpenCV's automatic choice, with a logged warning.
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


@functools.cache
def default_backend() -> int:
    """Return the OpenCV capture backend for this OS.

    DirectShow on Windows (the original app's backend).  Cached, so the
    unhandled-platform warning is logged once.
    """
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    log.warning(
        "No camera backend mapped for %s yet; using OpenCV's automatic choice.",
        sys.platform,
    )
    return cv2.CAP_ANY


def open_source(spec: int | str | os.PathLike[str]) -> FrameSource:
    """Open a frame source: a camera index, or the path of a video file.

    The source may fail to open; callers check ``isOpened()``.
    """
    if isinstance(spec, int):
        return cv2.VideoCapture(spec, default_backend())
    return cv2.VideoCapture(os.fspath(spec))
