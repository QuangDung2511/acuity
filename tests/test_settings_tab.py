# Tests for the Settings tab's camera group.
# ─────────────────────────────────────────────────────────────────
# Settings is where a user looks when the camera will not open, so the
# camera adapter's availability and reason must be on screen there.
# The device scan is stubbed so no real camera is probed.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel

from acuity.core.camera import CameraManager
from acuity.system import camera
from acuity.ui.tabs.settings_tab import SettingsTab


@pytest.fixture()
def no_device_scan(monkeypatch):
    monkeypatch.setattr(CameraManager, "list_devices", staticmethod(lambda: []))


@pytest.mark.parametrize(
    ("available", "status"),
    [(True, "Available"), (False, "Unavailable")],
)
def test_camera_group_shows_adapter_availability_and_reason(
    qtbot, monkeypatch, no_device_scan, available, status
):
    """A user whose camera will not open sees whether the backend is
    usable on this machine, and why, without reading the log."""
    reason = "Some explanation the adapter gave."
    monkeypatch.setattr(
        camera,
        "create_camera",
        lambda: camera.CameraAdapter(backend=0, available=available, reason=reason),
    )

    tab = SettingsTab()
    qtbot.addWidget(tab)

    label = tab.findChild(QLabel, "settingsCameraStatus")
    assert label is not None
    assert label.text().startswith(status)
    assert reason in label.text()
