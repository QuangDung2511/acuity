# Tests for the camera adapter (create_camera, open_source).
# ─────────────────────────────────────────────────────────────────
# The adapter decides which OpenCV backend opens the webcam on each OS,
# and open_source() is the one door every frame comes through, camera or
# video file.  Expected backends come from the spec's adapter table
# (DirectShow / AVFoundation / V4L2), not from the code's own mapping.
# No test opens a real camera: the platform tests only build the
# adapter, and the open test reads a video file written into tmp_path.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from acuity.system import camera


@pytest.mark.parametrize(
    ("platform", "backend", "name"),
    [
        ("win32", cv2.CAP_DSHOW, "DirectShow"),
        ("darwin", cv2.CAP_AVFOUNDATION, "AVFoundation"),
        ("linux", cv2.CAP_V4L2, "V4L2"),
    ],
)
def test_each_platform_gets_its_native_backend(monkeypatch, platform, backend, name):
    """OpenCV's automatic choice is slow or wrong on some OSes (MSMF on
    Windows), so each OS opens its camera with its native backend, and
    Settings names it so a user can tell which one is in use."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(camera.cv2.videoio_registry, "getCameraBackends", lambda: (backend,))

    adapter = camera.create_camera()

    assert adapter.backend == backend
    assert adapter.available
    assert name in adapter.reason


def test_backend_missing_from_opencv_build_is_reported_unavailable(monkeypatch):
    """A custom OpenCV build without V4L2 cannot open the Linux camera.
    Settings must say why, instead of the session failing with a bare
    "cannot open camera"."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(camera.cv2.videoio_registry, "getCameraBackends", lambda: ())

    adapter = camera.create_camera()

    assert adapter.backend == cv2.CAP_V4L2
    assert not adapter.available
    assert "V4L2" in adapter.reason
    assert "OpenCV" in adapter.reason


def test_unmapped_platform_falls_back_to_automatic_choice_and_says_so(monkeypatch):
    """An OS the adapter has no mapping for (e.g. FreeBSD) may still have
    a working camera, so it gets OpenCV's automatic choice, but the
    reason flags the guess rather than hiding it."""
    monkeypatch.setattr(sys, "platform", "freebsd14")

    adapter = camera.create_camera()

    assert adapter.backend == cv2.CAP_ANY
    assert adapter.available
    assert "freebsd14" in adapter.reason


@pytest.mark.parametrize("as_path", [True, False], ids=["Path", "str"])
def test_video_file_opens_through_the_same_frame_source_interface(tmp_path, as_path):
    """The headless and recorded-footage tests feed the real pipeline a
    video file instead of a webcam, so a file spec must yield a working
    FrameSource exactly like a camera index does."""
    video = tmp_path / "clip.avi"
    frame = np.full((48, 64, 3), 128, dtype=np.uint8)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for _ in range(3):
        writer.write(frame)
    writer.release()

    source = camera.open_source(video if as_path else str(video))
    try:
        assert source.isOpened()
        ok, read = source.read()
        assert ok
        assert read.shape == (48, 64, 3)
        assert source.get(cv2.CAP_PROP_FRAME_WIDTH) == 64
        assert source.getBackendName()
    finally:
        source.release()
