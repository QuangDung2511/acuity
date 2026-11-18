# Headless end-to-end session: real SessionManager, pipeline thread and DB.
# ─────────────────────────────────────────────────────────────────
# Drives a whole session through the real Qt wiring with no camera and
# no private files, so it runs in CI on every OS:
#
#   idle → calibrating → focus → head-pose event → DB row → rest
#        → second block (fresh pipeline) → one more event → end
#
# and, through the real MainWindow, the event review: with the
# review_events setting on, the review dialog is requested at rest
# start and at session end; with it off, never.
#
# Seams:
#   • Frame source: acuity.system.camera.open_source is replaced by a
#     synthetic source of blank frames.  The camera manager, pipeline
#     thread, engines, signals and DB are the real ones.
#   • Face and pose analyzers: replaced by stand-ins that report a face
#     looking at the screen or turned away, as the test scripts it.
#     MediaPipe finds no face in a synthetic frame, and no recorded
#     face can ship with the repo, so this is the only way a detection
#     event can happen in CI.  The private-video tests in
#     test_session_manager.py cover the real analyzers locally.
#   • Timing: calibration is shortened to 0.3 s and the head-pose
#     minimum threshold lowered, so the test takes seconds, not 30 s.
#   • Review dialog: replaced by a stand-in that records what it was
#     asked to show and returns "Skip", since a real modal would block.
#     The sound player is stubbed too (conftest.silent_sound_player).
#
# What it catches: PySide6 signal-wiring breaks between the pipeline
# thread and the session manager, events that never reach the DB, and
# pipeline signals connected again on each focus block (which would
# log every event twice from the second block on).
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import functools

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog

import acuity.core.cv_pipeline as cv_pipeline
import acuity.core.distraction_engine as distraction_engine
import acuity.db.database as database
from acuity.core.calibration import CalibrationRunner
from acuity.core.face_mesh import FaceResult
from acuity.core.pose_tracker import PoseResult
from acuity.core.session_manager import SessionManager, SessionState
from acuity.db import queries
from acuity.system import camera
from acuity.ui import main_window
from acuity.utils.config import ConfigManager
from acuity.utils.constants import EVENT_DISTRACTION_POSE

_WAIT_MS = 10_000


class _Scene:
    """What the stand-in analyzers report; the test flips it mid-run."""

    def __init__(self) -> None:
        self.head_yaw = 0.0  # degrees from the screen; > 40 counts as turned away


class _SyntheticSource:
    """A FrameSource of blank frames, recording its open/release."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._open = True
        log.append("open")

    def isOpened(self) -> bool:
        return self._open

    def read(self):
        if not self._open:
            return False, None
        return True, np.zeros((240, 320, 3), dtype=np.uint8)

    def release(self) -> None:
        if self._open:
            self._open = False
            self._log.append("release")

    def get(self, prop_id: int) -> float:
        return 0.0

    def getBackendName(self) -> str:
        return "synthetic"


class _FakeFaceMesh:
    def __init__(self, scene: _Scene) -> None:
        self._scene = scene

    def process(self, frame) -> FaceResult:
        return FaceResult(
            ear_left=0.30,
            ear_right=0.30,
            ear_avg=0.30,
            mar=0.05,
            pitch=0.0,
            yaw=self._scene.head_yaw,
            roll=0.0,
            face_detected=True,
        )

    def close(self) -> None:
        pass


class _FakePose:
    def process(self, frame) -> PoseResult:
        return PoseResult(landmarks=None, pose_detected=True)

    def close(self) -> None:
        pass


@pytest.fixture()
def stubs(monkeypatch, qtbot):
    """Synthetic camera, stand-in analyzers and a fresh DB.

    Yields (scene, source_log).
    """
    scene = _Scene()
    source_log: list[str] = []

    def open_source(spec):
        return _SyntheticSource(source_log)

    monkeypatch.setattr(camera, "open_source", open_source)
    monkeypatch.setattr(cv_pipeline, "FaceMeshAnalyzer", lambda: _FakeFaceMesh(scene))
    monkeypatch.setattr(cv_pipeline, "PoseAnalyzer", _FakePose)
    monkeypatch.setattr(
        cv_pipeline, "CalibrationRunner", functools.partial(CalibrationRunner, duration_s=0.3)
    )
    monkeypatch.setattr(distraction_engine, "HEAD_POSE_THRESHOLD_MIN_S", 0.0)
    ConfigManager().set("head_pose_threshold", 0.5)

    # get_db() caches its path; give this test its own DB under tmp_path.
    monkeypatch.setattr(database, "_instance", None)
    db = database.get_db()
    db.connect()

    yield scene, source_log

    db.close()


@pytest.fixture()
def harness(stubs):
    """A SessionManager on a fresh DB, fed by the synthetic source."""
    scene, source_log = stubs
    sm = SessionManager()
    states: list[str] = []
    logged: list[str] = []
    sm.state_changed.connect(states.append)
    sm.event_logged.connect(logged.append)

    yield sm, scene, source_log, states, logged

    if sm.state != SessionState.IDLE:
        sm.end_session()
    sm._stop_all_timers()


def _turn_away_until_logged(qtbot, scene: _Scene, logged: list[str], count: int) -> None:
    """Turn the head away until the count-th event is logged, then back."""
    scene.head_yaw = 90.0
    qtbot.waitUntil(lambda: len(logged) >= count, timeout=_WAIT_MS)
    scene.head_yaw = 0.0


def test_session_runs_end_to_end_through_the_real_pipeline(qtbot, harness):
    """A head turn during focus becomes exactly one DB event per episode,
    across two focus blocks that each get a fresh pipeline."""
    sm, scene, source_log, states, logged = harness

    # ── Block 1: calibrate, focus, one event, rest ──────────────
    sm.start_session()
    qtbot.waitUntil(lambda: sm.state == SessionState.FOCUS, timeout=_WAIT_MS)
    assert states[:2] == ["CALIBRATING", "FOCUS"]

    _turn_away_until_logged(qtbot, scene, logged, 1)
    qtbot.wait(300)  # a duplicate connection would deliver a second copy
    events = queries.get_events_for_session(sm.session_id)
    assert [e.event_type for e in events] == [EVENT_DISTRACTION_POSE]
    assert len(logged) == 1
    first_block_id = events[0].block_id

    sm.request_rest()
    assert sm.state == SessionState.REST
    assert sm.pipeline is None
    assert source_log == ["open", "release"]  # camera released for the rest

    # ── Block 2: a new pipeline, still one log line per event ──
    sm.skip_rest()
    sm.begin_next_block()
    qtbot.waitUntil(lambda: sm.state == SessionState.FOCUS, timeout=_WAIT_MS)
    assert sm.block_number == 2

    _turn_away_until_logged(qtbot, scene, logged, 2)
    qtbot.wait(300)
    events = queries.get_events_for_session(sm.session_id)
    assert len(events) == 2
    assert len(logged) == 2
    assert events[1].block_id != first_block_id

    # ── End ─────────────────────────────────────────────────────
    session_id = sm.session_id
    sm.end_session()
    assert sm.state == SessionState.IDLE
    assert queries.get_session(session_id).end_time is not None
    assert source_log == ["open", "release", "open", "release"]


class _FakeReviewDialog:
    """Stands in for EventReviewDialog: records each request, then Skip."""

    requests: list[tuple[str, list[str]]] = []

    def __init__(self, events, status, parent=None) -> None:
        self.requests.append((status, [e.event_type for e in events]))

    def exec(self) -> QDialog.DialogCode:
        return QDialog.DialogCode.Rejected

    def labels(self) -> dict[int, str]:
        return {}


@pytest.mark.parametrize("review_events", [True, False])
def test_review_dialog_follows_the_review_events_setting(
    qtbot, monkeypatch, stubs, silent_sound_player, review_events
):
    """Rating alerts is opt-in: with review_events on, the block's events
    are put to the user when the rest starts and again at session end
    (a skipped review comes back); with it off, the user is never asked."""
    scene, _ = stubs
    monkeypatch.setattr(_FakeReviewDialog, "requests", [])
    monkeypatch.setattr(main_window, "EventReviewDialog", _FakeReviewDialog)
    ConfigManager().set("review_events", review_events)

    window = main_window.MainWindow()
    qtbot.addWidget(window)
    sm = window._session_mgr
    logged: list[str] = []
    sm.event_logged.connect(logged.append)

    sm.start_session()
    qtbot.waitUntil(lambda: sm.state == SessionState.FOCUS, timeout=_WAIT_MS)
    _turn_away_until_logged(qtbot, scene, logged, 1)

    sm.request_rest()
    qtbot.wait(100)  # the review opens one event-loop turn after REST
    sm.end_session()
    qtbot.wait(100)  # and the session-end review one turn after that

    if review_events:
        assert _FakeReviewDialog.requests == [
            ("Block 1 review.", [EVENT_DISTRACTION_POSE]),
            ("Session review.", [EVENT_DISTRACTION_POSE]),
        ]
    else:
        assert _FakeReviewDialog.requests == []
