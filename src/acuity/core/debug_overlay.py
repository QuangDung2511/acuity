# Real-time diagnostic overlay renderer for the debug overlay.
# ─────────────────────────────────────────────────────────────────
# A pure OpenCV drawing module — zero Qt dependency.
#
# Thread: CV worker (called from the pipeline loop).
#
# Takes a BGR frame + DebugSnapshot and returns the annotated BGR
# frame.  Also accepts optional face and pose landmark arrays so
# it can draw color-coded key landmarks directly on the body.
#
# Layout: two semi-transparent panels at the frame edges:
#   • Left  — Presence badge, head-pose baselines, fatigue, legend
#   • Right — Distraction (head pose) metrics
#
# The original's phone and posture sections, the phone bounding box
# and the torso readout are gone with their detectors.  The pose
# skeleton stays: the pose landmarker still feeds presence.
#
# Panels are built through the _Panel layout helper: rows are queued
# first, then rendered into a background box sized to the content.
# Labels sit on the left edge, values are right-aligned, and anything
# too wide is auto-shrunk (then ellipsized) so lines never collide.
#
# Text is deliberately ASCII-only.  cv2.putText draws Hershey vector
# fonts, which have no glyphs for box-drawing / check / degree / greek
# characters — those render as '?'.  Bullets, swatches and dividers are
# drawn as rectangles instead.
#
# Two color systems (never overlap, to eliminate ambiguity):
#   1. Status text  — Green/Yellow/Red (OK / warning / alert)
#   2. Landmarks    — Cyan, LightBlue, White, Orange, Magenta, Pink
#
# Producer: CVPipelineThread._build_debug_snapshot()
# Consumer:  CVPipelineThread._run_tracking() / _run_calibration()
#
# Frame-skip support: when frame_skip > 1 the renderer caches the
# last drawn overlay mask and re-composites it onto new frames,
# skipping the expensive text / drawing pass for N-1 out of every
# N frames.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import cv2
import numpy as np

from acuity.core.debug_snapshot import DebugSnapshot
from acuity.core.face_mesh import FaceResult
from acuity.core.pose_tracker import PoseResult

# ── Status color palette (BGR) — traffic-light system ───────────
_COLOR_OK = (0, 200, 0)  # green
_COLOR_WARN = (0, 220, 220)  # yellow (BGR: B=0,G=220,R=220)
_COLOR_ALERT = (0, 0, 220)  # red
_COLOR_WHITE = (230, 230, 230)
_COLOR_GRAY = (150, 150, 150)
_COLOR_DIM = (100, 100, 100)

# ── Landmark color palette (BGR) — body-space, never green/yellow/red
_LM_L_EYE = (255, 255, 0)  # Cyan     — left eye contour
_LM_R_EYE = (255, 191, 100)  # LightBlue — right eye contour
_LM_POSE_ANCHOR = (255, 255, 255)  # White — head-pose anchor pts
_LM_NOSE = (0, 165, 255)  # Orange   — nose point
_LM_MOUTH = (0, 255, 128)  # Spring Green — inner-lip contour
_LM_SHOULDER = (255, 0, 255)  # Magenta  — shoulders
_LM_SHLDR_NOSE_LINE = (147, 20, 255)  # Pink — shoulder-mid→nose

# ── MediaPipe FaceMesh landmark indices ─────────────────────────
# (based on 478-point FaceLandmarker model)
_LM_LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # subject's left
_LM_RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]  # subject's right
_LM_HEAD_POSE_IDX = [1, 10, 33, 263, 6, 168]  # pose anchors (expression-rigid)
# Inner-lip contour for MAR / yawn visualization.
# Ordered clockwise: left corner → upper lip → right corner → lower lip.
_LM_INNER_LIP_IDX = [
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
]

# ── MediaPipe Pose landmark indices ─────────────────────────────
_POSE_NOSE = 0
_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER = 12

# ── Layout constants ─────────────────────────────────────────────
_PANEL_ALPHA = 0.62  # opacity of semi-transparent panels
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_SMALL = 0.36
_FONT_SCALE_LABEL = 0.40
_FONT_SCALE_HEADER = 0.44
_FONT_SCALE_MIN = 0.28  # never shrink text below this
_LINE_HEIGHT = 15  # pixels between text lines (calibration view)
_MARGIN = 5  # inner margin from panel edge
_PANEL_W_FRAC = 0.33  # panel width as fraction of frame width

# Row metrics for the _Panel layout engine.
_PAD_X = 7  # horizontal padding inside a panel
_PAD_Y = 6  # vertical padding inside a panel
_ROW_H = 14  # pitch of a plain text / key-value row
_SECTION_H = 18  # pitch of a section-header row
_BAR_H = 7  # thickness of a progress / signal bar
_BAR_ROW_H = 11  # pitch of a bar row
_GUTTER = 6  # min gap between a label and its value
_SIG_LABEL_W = 50  # label column width for signal bars
_SIG_VALUE_W = 42  # value column width for signal bars (incl. gutter)

# ── Warning thresholds (fraction of limit) ───────────────────────
_WARN_FRAC = 0.70  # > 70 % of limit → yellow


# ──────────────────────────────────────────────────────────────────
# Text primitives
# ──────────────────────────────────────────────────────────────────


def _text_w(text: str, scale: float, thickness: int = 1) -> int:
    """Width in pixels of *text* rendered at *scale*."""
    return cv2.getTextSize(text, _FONT, scale, thickness)[0][0]


def _fit(text: str, max_w: int, scale: float) -> tuple[str, float]:
    """Shrink (then ellipsize) *text* so it fits within *max_w* pixels.

    Returns the possibly-truncated text and the scale to draw it at.
    Keeps the HUD readable on small frames without ever letting a long
    value spill over its neighbour.
    """
    if max_w <= 0 or _text_w(text, scale) <= max_w:
        return text, scale
    # Step the scale down before resorting to truncation.
    while scale > _FONT_SCALE_MIN:
        scale = round(scale - 0.02, 2)
        if _text_w(text, scale) <= max_w:
            return text, scale
    scale = _FONT_SCALE_MIN
    while len(text) > 1 and _text_w(text + "..", scale) > max_w:
        text = text[:-1]
    return text + "..", scale


def _put_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    bold: bool = False,
) -> None:
    """Draw one line of text with a thin dark outline for legibility."""
    thickness = 2 if bold else 1
    cv2.putText(
        frame,
        text,
        (pos[0] + 1, pos[1] + 1),
        _FONT,
        scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        pos,
        _FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _swatch(
    frame: np.ndarray,
    x: int,
    baseline_y: int,
    color: tuple[int, int, int],
    size: int = 6,
) -> None:
    """Draw a small filled square used as a bullet / legend swatch."""
    top = baseline_y - size
    cv2.rectangle(frame, (x, top), (x + size, top + size), color, -1)


# ──────────────────────────────────────────────────────────────────
# _Panel — vertical stack layout
# ──────────────────────────────────────────────────────────────────


class _Panel:
    """Collects HUD rows, then renders them into a self-sized box.

    Rows are queued with the builder methods (:meth:`title`,
    :meth:`section`, :meth:`kv`, …), each of which reserves a fixed
    vertical slot.  :meth:`render` draws the translucent background at
    exactly the accumulated height and then paints every row inside it.

    Labels are left-aligned at the panel's inner edge and values are
    right-aligned to the opposite edge, so numbers form a clean column;
    a value that would collide with its label is shrunk by :func:`_fit`.
    """

    def __init__(self, x: int, y: int, w: int) -> None:
        self._x = x
        self._y = y
        self._w = w
        self._rows: list[tuple] = []
        self._content_h = 0

    # ── Row builders ─────────────────────────────────────────────

    def title(self, text: str, color: tuple[int, int, int]) -> None:
        """Bulleted headline row (presence badge)."""
        self._add(("title", text, color), _ROW_H + 4)

    def section(
        self,
        text: str,
        chips: list[tuple[str, tuple[int, int, int]]] | None = None,
    ) -> None:
        """Divider + section label, with optional right-aligned chips."""
        self._add(("section", text, chips or []), _SECTION_H)

    def kv(
        self,
        label: str,
        value: str,
        value_color: tuple[int, int, int] = _COLOR_WHITE,
        label_color: tuple[int, int, int] = _COLOR_GRAY,
    ) -> None:
        """Label on the left, value right-aligned on the same row."""
        self._add(("kv", label, value, value_color, label_color), _ROW_H)

    def bar(
        self,
        fraction: float,
        color: tuple[int, int, int],
        gradient: bool = False,
    ) -> None:
        """Full-width progress bar row."""
        self._add(("bar", fraction, color, gradient), _BAR_ROW_H)

    def signal(
        self,
        label: str,
        fraction: float,
        color: tuple[int, int, int],
    ) -> None:
        """Label + mini bar + numeric value, in three fixed columns."""
        self._add(("signal", label, fraction, color), _BAR_ROW_H)

    def legend(self, items: list[tuple[str, tuple[int, int, int]]]) -> None:
        """Two-column color legend."""
        rows = (len(items) + 1) // 2
        self._add(("legend", items), rows * (_ROW_H - 1))

    def gap(self, px: int = 4) -> None:
        self._add(("gap",), px)

    # ── Geometry ─────────────────────────────────────────────────

    @property
    def height(self) -> int:
        return self._content_h + 2 * _PAD_Y

    def _add(self, row: tuple, height: int) -> None:
        self._rows.append(row)
        self._content_h += height

    # ── Rendering ────────────────────────────────────────────────

    def render(self, frame: np.ndarray) -> None:
        """Draw the background box and every queued row onto *frame*."""
        frame_h = frame.shape[0]
        box_h = min(self.height, max(0, frame_h - self._y))
        _draw_translucent_box(frame, self._x, self._y, self._w, box_h, _PANEL_ALPHA)

        left = self._x + _PAD_X
        right = self._x + self._w - _PAD_X
        inner_w = right - left
        y = self._y + _PAD_Y

        for row in self._rows:
            kind = row[0]

            if kind == "gap":
                y += row[1] if len(row) > 1 else 4

            elif kind == "title":
                _, text, color = row
                base = y + _ROW_H - 4
                _swatch(frame, left, base - 1, color, 7)
                text, scale = _fit(text, inner_w - 13, _FONT_SCALE_HEADER)
                _put_text(frame, text, (left + 12, base), color, scale, bold=True)
                y += _ROW_H + 4

            elif kind == "section":
                _, text, chips = row
                base = y + _SECTION_H - 6
                cv2.line(frame, (left, y + 1), (right, y + 1), (95, 95, 95), 1)
                # Chips are laid out from the right edge inwards.
                chip_x = right
                for chip_text, chip_color in reversed(chips):
                    cw = _text_w(chip_text, _FONT_SCALE_SMALL)
                    chip_x -= cw
                    _put_text(
                        frame, chip_text, (chip_x, base), chip_color, _FONT_SCALE_SMALL, bold=True
                    )
                    chip_x -= _GUTTER
                text, scale = _fit(text, chip_x - left - _GUTTER, _FONT_SCALE_LABEL)
                _put_text(frame, text, (left, base), (205, 205, 205), scale)
                y += _SECTION_H

            elif kind == "kv":
                _, label, value, value_color, label_color = row
                base = y + _ROW_H - 4
                # Value keeps priority; the label yields space if needed.
                value, v_scale = _fit(value, inner_w - 30, _FONT_SCALE_SMALL)
                v_w = _text_w(value, v_scale)
                label, l_scale = _fit(label, inner_w - v_w - _GUTTER, _FONT_SCALE_SMALL)
                _put_text(frame, label, (left, base), label_color, l_scale)
                _put_text(frame, value, (right - v_w, base), value_color, v_scale)
                y += _ROW_H

            elif kind == "bar":
                _, fraction, color, gradient = row
                if gradient:
                    _draw_gradient_bar(frame, left, y, inner_w, _BAR_H, fraction)
                else:
                    _draw_bar(frame, left, y, inner_w, _BAR_H, fraction, color)
                y += _BAR_ROW_H

            elif kind == "signal":
                _, label, fraction, color = row
                base = y + _BAR_H + 1
                bar_w = max(10, inner_w - _SIG_LABEL_W - _SIG_VALUE_W)
                lbl, l_scale = _fit(label, _SIG_LABEL_W - 4, _FONT_SCALE_SMALL)
                _put_text(frame, lbl, (left, base), _COLOR_GRAY, l_scale)
                _draw_bar(frame, left + _SIG_LABEL_W, y, bar_w, _BAR_H, fraction, color)
                val = f"{fraction:.2f}"
                _put_text(
                    frame,
                    val,
                    (right - _text_w(val, _FONT_SCALE_SMALL), base),
                    _COLOR_GRAY,
                    _FONT_SCALE_SMALL,
                )
                y += _BAR_ROW_H

            elif kind == "legend":
                _, items = row
                col_w = inner_w // 2
                for i, (label, color) in enumerate(items):
                    cx = left + (i % 2) * col_w
                    base = y + (i // 2) * (_ROW_H - 1) + _ROW_H - 5
                    _swatch(frame, cx, base - 1, color, 6)
                    text, scale = _fit(label, col_w - 12, _FONT_SCALE_SMALL)
                    _put_text(frame, text, (cx + 10, base), color, scale)
                y += ((len(items) + 1) // 2) * (_ROW_H - 1)


# ──────────────────────────────────────────────────────────────────
# Box / bar primitives
# ──────────────────────────────────────────────────────────────────


def _draw_translucent_box(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    alpha: float,
) -> None:
    """Darken a rectangular region so text on top stays readable."""
    sub = frame[y : y + h, x : x + w]
    if sub.size == 0:
        return
    black = np.zeros_like(sub)
    cv2.addWeighted(black, alpha, sub, 1.0 - alpha, 0, sub)
    frame[y : y + h, x : x + w] = sub


def _draw_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    fraction: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled horizontal progress bar."""
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 55, 55), -1)
    filled_w = int(w * fraction)
    if filled_w > 0:
        cv2.rectangle(frame, (x, y), (x + filled_w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)


def _draw_gradient_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    score: float,
) -> None:
    """Progress bar whose fill colour runs green → yellow → red.

    * score 0.0 – 0.5 : green  → yellow
    * score 0.5 – 1.0 : yellow → red
    """
    score = max(0.0, min(1.0, score))
    if score <= 0.5:
        t = score / 0.5
        color = (0, 200, int(220 * t))
    else:
        t = (score - 0.5) / 0.5
        color = (0, int(220 * (1.0 - t)), 220)
    _draw_bar(frame, x, y, w, h, score, color)


# ──────────────────────────────────────────────────────────────────
# DebugOverlayRenderer
# ──────────────────────────────────────────────────────────────────


class DebugOverlayRenderer:
    """Renders a semi-transparent diagnostic HUD onto BGR frames.

    Parameters
    ----------
    show_landmarks : bool
        When ``True`` (default), key MediaPipe landmarks are drawn on
        the body in group-specific colors.  Set to ``False`` to show
        only the text panels.
    frame_skip : int
        Draw a fresh overlay every *N* frames (default ``1`` = every
        frame).  When ``N > 1``, the last rendered overlay is cached
        and re-composited without recalculating text for the
        intermediate frames.  Useful if heavier rendering is added
        later (e.g., landmark trails).

    Usage
    -----
    ::

        renderer = DebugOverlayRenderer()
        annotated = renderer.draw(frame.copy(), snapshot,
                                  face_landmarks_px=face.landmarks_px,
                                  pose_landmarks_px=pose.landmarks)
    """

    def __init__(
        self,
        show_landmarks: bool = True,
        frame_skip: int = 1,
    ) -> None:
        self._show_landmarks = show_landmarks
        self._frame_skip = max(1, int(frame_skip))
        self._frame_count: int = 0
        # Cached overlay layer (same shape as frame, BGRA or BGR).
        # When frame_skip > 1 this is the last fully-rendered overlay
        # blended into the frame.
        self._cached_overlay: np.ndarray | None = None

    # ── Public API ───────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        snapshot: DebugSnapshot,
        face_landmarks_px: np.ndarray | None = None,
        pose_landmarks_px: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw the full debug overlay onto *frame* and return it.

        The frame is mutated in-place for efficiency; the same array
        is also returned so callers can chain calls.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (``H × W × 3``, ``uint8``).  Should be a copy
            if the original must be preserved.
        snapshot : DebugSnapshot
            Frozen per-frame diagnostic state (produced by the pipeline).
        face_landmarks_px : np.ndarray or None
            ``(478, 2)`` array of face-mesh landmark pixel coordinates,
            or ``None`` when no face was detected.
        pose_landmarks_px : np.ndarray or None
            ``(33, 2)`` array of pose landmark pixel coordinates,
            or ``None`` when no pose was detected.
        """
        self._frame_count += 1
        h, w = frame.shape[:2]

        should_render = (self._frame_count % self._frame_skip) == 1 or self._frame_skip == 1

        if should_render or self._cached_overlay is None:
            # ── Build a fresh overlay ────────────────────────────
            overlay = frame.copy()

            # Draw body-space landmarks first (they sit under the panels).
            if self._show_landmarks:
                if face_landmarks_px is not None:
                    self._draw_face_landmarks(overlay, face_landmarks_px)
                if pose_landmarks_px is not None:
                    self._draw_pose_skeleton(overlay, pose_landmarks_px)

            # ── Two text panels ──────────────────────────────────
            # Never let the pair cover more than the frame width.
            panel_w = max(150, min(int(w * _PANEL_W_FRAC), (w - 16) // 2))

            self._draw_left_panel(overlay, snapshot, 0, panel_w)
            self._draw_right_panel(overlay, snapshot, w - panel_w, panel_w)

            self._cached_overlay = overlay
            # Freshly rendered: take it verbatim so text stays crisp.
            frame[:] = overlay
            return frame

        # ── Composite cached overlay onto frame ──────────────────
        # The cache holds an older video frame under the HUD, so blend
        # it rather than overwriting the live image outright.
        cv2.addWeighted(
            self._cached_overlay,
            0.85,
            frame,
            0.15,
            0,
            frame,
        )
        return frame

    def draw_calibration(
        self,
        frame: np.ndarray,
        face: FaceResult,
        pose: PoseResult,
        face_sample_count: int,
        elapsed_s: float,
        duration_s: float,
    ) -> np.ndarray:
        """Lighter overlay for the calibration phase.

        Draws live EAR, pitch/yaw values, the face sample count, pose
        detection and the pose skeleton.  Does not require a ``DebugSnapshot`` since engines
        don't exist during calibration.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to annotate (mutated in-place).
        face : FaceResult
            Current frame's face analysis output.
        pose : PoseResult
            Current frame's pose analysis output.
        face_sample_count : int
            Number of valid face samples collected so far.
        elapsed_s : float
            Seconds elapsed since calibration started.
        duration_s : float
            Total calibration duration (seconds).
        """
        h, w = frame.shape[:2]
        panel_w = max(200, int(w * 0.35))
        panel_h = 130  # one line shorter: no pose sample count

        # Semi-transparent panel (top-left).
        self._draw_panel(frame, 0, 0, panel_w, panel_h, _PANEL_ALPHA)

        x, y = _MARGIN, _MARGIN + _LINE_HEIGHT
        progress = min(elapsed_s / max(duration_s, 0.001), 1.0)
        prog_color = _COLOR_OK if progress >= 1.0 else _COLOR_WARN

        _swatch(frame, x, y - 1, _COLOR_WHITE, 7)
        self._draw_text(
            frame, "CALIBRATING", (x + 12, y), _COLOR_WHITE, _FONT_SCALE_HEADER, bold=True
        )
        y += _LINE_HEIGHT + 3
        self._draw_progress_bar(frame, x, y, panel_w - 2 * _MARGIN, 8, progress, prog_color)
        y += 13
        self._draw_text(
            frame,
            f"  {elapsed_s:.1f}s / {duration_s:.0f}s",
            (x, y),
            _COLOR_GRAY,
            _FONT_SCALE_SMALL,
        )
        y += _LINE_HEIGHT
        self._draw_text(
            frame,
            f"  Face samples: {face_sample_count}",
            (x, y),
            _COLOR_WHITE,
            _FONT_SCALE_SMALL,
        )
        y += _LINE_HEIGHT

        # EAR
        if face.face_detected:
            self._draw_text(
                frame,
                f"  EAR: {face.ear_avg:.3f}  P:{face.pitch:.1f}  Y:{face.yaw:.1f}",
                (x, y),
                _COLOR_OK,
                _FONT_SCALE_SMALL,
            )
        else:
            self._draw_text(frame, "  Face: NOT DETECTED", (x, y), _COLOR_ALERT, _FONT_SCALE_SMALL)
        y += _LINE_HEIGHT

        if pose.pose_detected:
            self._draw_text(frame, "  Pose: detected", (x, y), _COLOR_OK, _FONT_SCALE_SMALL)
        else:
            self._draw_text(frame, "  Pose: NOT DETECTED", (x, y), _COLOR_ALERT, _FONT_SCALE_SMALL)

        # Landmarks on body.
        if self._show_landmarks:
            if face.landmarks_px is not None:
                self._draw_face_landmarks(frame, face.landmarks_px)
            if pose.landmarks is not None:
                self._draw_pose_skeleton(frame, pose.landmarks)

        return frame

    # ── Panel drawing helpers ────────────────────────────────────

    _draw_panel = staticmethod(_draw_translucent_box)
    _draw_text = staticmethod(_put_text)
    _draw_progress_bar = staticmethod(_draw_bar)

    # ── Status color helper ──────────────────────────────────────

    @staticmethod
    def _status_color(
        value: float,
        limit: float,
        invert: bool = False,
    ) -> tuple[int, int, int]:
        """Return Green/Yellow/Red based on *value* vs *limit*.

        Parameters
        ----------
        value : float
            Current metric value.
        limit : float
            The threshold at which the metric triggers an alert.
        invert : bool
            When ``True``, *higher* is better (e.g. EAR vs threshold).
            ``value < limit`` is then the bad condition.
        """
        if limit <= 0.0:
            return _COLOR_OK
        if invert:
            # Lower value is bad (e.g. EAR falling below blink threshold)
            ratio = value / limit if limit > 0 else 1.0
            if ratio >= 1.0:
                return _COLOR_OK
            if ratio >= (1.0 - _WARN_FRAC):
                return _COLOR_WARN
            return _COLOR_ALERT
        else:
            # Higher value is bad (e.g. timer elapsed vs threshold)
            ratio = value / limit
            if ratio >= 1.0:
                return _COLOR_ALERT
            if ratio >= _WARN_FRAC:
                return _COLOR_WARN
            return _COLOR_OK

    # ── Left panel ───────────────────────────────────────────────

    def _draw_left_panel(
        self,
        frame: np.ndarray,
        snap: DebugSnapshot,
        panel_x: int,
        panel_w: int,
    ) -> None:
        """Presence badge, head-pose baselines, fatigue model, legend."""
        p = _Panel(panel_x, 0, panel_w)

        # ── Presence badge ───────────────────────────────────────
        pcolor = {
            "PRESENT": _COLOR_OK,
            "PRESENT_NO_FACE": _COLOR_WARN,
            "ABSENT": _COLOR_ALERT,
        }.get(snap.presence, _COLOR_GRAY)
        p.title(snap.presence.replace("_", " "), pcolor)
        both = snap.face_detected and snap.pose_detected
        p.kv(
            "Face / Pose",
            f"{'yes' if snap.face_detected else 'no'} / {'yes' if snap.pose_detected else 'no'}",
            _COLOR_OK if both else _COLOR_WARN,
        )

        # ── Head-pose baselines ──────────────────────────────────
        p.section("HEAD POSE (deg)")
        p.kv("Base P / Y", f"{snap.baseline_pitch:+.1f} / {snap.baseline_yaw:+.1f}")
        p.kv("Live P / Y", f"{snap.pitch:+.1f} / {snap.yaw:+.1f}")

        # ── Fatigue ──────────────────────────────────────────────
        chips: list[tuple[str, tuple[int, int, int]]] = []
        if snap.pitch_gated:
            chips.append(("GATED", _COLOR_WARN))
        elif snap.ear_suppressed:
            # Suppressed for a reason other than pitch — a fast head turn,
            # a collapsed mesh solution, or the settling window after one.
            chips.append(("EAR?", _COLOR_WARN))
        if snap.fatigue_alert_active:
            chips.append(("ALERT", _COLOR_ALERT))
        elif snap.fatigue_warning_active:
            chips.append(("WARN", _COLOR_WARN))
        p.section("FATIGUE", chips)

        ear_color = self._status_color(snap.smoothed_ear, snap.blink_threshold, invert=True)
        p.kv("EAR / thr", f"{snap.smoothed_ear:.3f} / {snap.blink_threshold:.3f}", ear_color)
        # Per-eye split.  The average alone hides a one-eyed collapse, which
        # is what a wink looks like and what the microsleep bilateral share
        # exists to refuse — see MICROSLEEP_BILATERAL_SHARE in constants.py.
        one_eyed = (snap.ear_left < snap.blink_threshold) != (snap.ear_right < snap.blink_threshold)
        p.kv(
            "EAR L / R",
            f"{snap.ear_left:.3f} / {snap.ear_right:.3f}",
            _COLOR_WARN if one_eyed else ear_color,
        )
        # Three states, because they drive different things: SHUT is what
        # the microsleep timer runs on, NARROW is below the blink threshold
        # but not shut (it feeds PERCLOS and blink counting only).
        if snap.eyes_shut:
            p.kv("Eyes", "SHUT", _COLOR_ALERT)
        elif snap.eyes_closed:
            p.kv("Eyes", "NARROW", _COLOR_WARN)
        else:
            p.kv("Eyes", "OPEN", _COLOR_OK)
        if snap.microsleep_active:
            p.kv("Microsleep", f"{snap.microsleep_elapsed_s:.1f}s", _COLOR_ALERT)

        # Composite score + its five contributing signals.
        p.kv("Score", f"{snap.fatigue_score:.2f}", self._fatigue_signal_color(snap.fatigue_score))
        p.bar(snap.fatigue_score, _COLOR_OK, gradient=True)
        p.gap(2)
        for label, value in (
            ("PERCLOS", snap.signal_perclos),
            ("BlnkDur", snap.signal_blink_dur),
            ("Yawn", snap.signal_yawn),
            ("BlnkFrq", snap.signal_blink_freq),
            ("HeadNod", snap.signal_head_nod),
        ):
            p.signal(label, value, self._fatigue_signal_color(value))
        p.gap(3)

        p.kv(
            "MAR / thr",
            f"{snap.mar_smoothed:.3f} / {snap.yawn_threshold:.3f}"
            f"  {'YAWNING' if snap.yawn_active else 'ok'}",
            _COLOR_ALERT if snap.yawn_active else _COLOR_OK,
        )
        p.kv("Yawns / 5m", f"{snap.yawn_count_5min}")
        p.kv("Blink avg", f"{snap.avg_blink_duration_ms:.0f} ms")
        p.kv(
            "Nods / 1m",
            f"{snap.head_nod_count_1min}",
            _COLOR_WARN if snap.head_nod_count_1min > 0 else _COLOR_WHITE,
        )
        p.kv(
            "Cooldown",
            f"{snap.fatigue_cooldown_remaining:.1f}s",
            _COLOR_WARN if snap.fatigue_cooldown_remaining > 0 else _COLOR_DIM,
        )
        p.kv(
            "Desk writing",
            "ON" if snap.desk_writing_mode else "off",
            _COLOR_WARN if snap.desk_writing_mode else _COLOR_DIM,
        )

        # ── Landmark legend ──────────────────────────────────────
        if self._show_landmarks:
            p.section("LANDMARKS")
            p.legend(
                [
                    ("L-eye", _LM_L_EYE),
                    ("R-eye", _LM_R_EYE),
                    ("Pose", _LM_POSE_ANCHOR),
                    ("Nose", _LM_NOSE),
                    ("Mouth", _LM_MOUTH),
                    ("Shldr", _LM_SHOULDER),
                ]
            )

        p.render(frame)

    # ── Right panel ──────────────────────────────────────────────

    def _draw_right_panel(
        self,
        frame: np.ndarray,
        snap: DebugSnapshot,
        panel_x: int,
        panel_w: int,
    ) -> None:
        """Distraction (head pose) metrics."""
        p = _Panel(panel_x, 0, panel_w)

        # ── Distraction — head pose ──────────────────────────────
        p.section(
            "DISTRACTION: HEAD",
            [("FIRED", _COLOR_ALERT)] if snap.pose_timer_fired else [],
        )
        p.kv(
            "Deviated",
            "YES" if snap.head_pose_deviated else "no",
            _COLOR_ALERT if snap.head_pose_deviated else _COLOR_OK,
        )
        p.kv(
            "Pitch d / tol",
            f"{snap.pitch_delta:.1f} / {snap.head_pose_pitch_tolerance_deg:.0f}",
            self._status_color(snap.pitch_delta, snap.head_pose_pitch_tolerance_deg),
        )
        p.kv(
            "Yaw d / tol",
            f"{snap.yaw_delta:.1f} / {snap.head_pose_yaw_tolerance_deg:.0f}",
            self._status_color(snap.yaw_delta, snap.head_pose_yaw_tolerance_deg),
        )
        p.kv(
            "Rel / cal", f"{snap.body_relative_pitch:+.1f} / {snap.calibrated_relative_pitch:+.1f}"
        )

        pose_timer_color = self._status_color(
            snap.pose_timer_elapsed_s, snap.pose_timer_threshold_s
        )
        p.kv(
            "Timer",
            f"{snap.pose_timer_elapsed_s:.1f}s / {snap.pose_timer_threshold_s:.0f}s",
            pose_timer_color,
        )
        p.bar(snap.pose_timer_elapsed_s / max(snap.pose_timer_threshold_s, 0.001), pose_timer_color)

        p.render(frame)

    # ── Fatigue signal color helper ──────────────────────────────

    @staticmethod
    def _fatigue_signal_color(value: float) -> tuple[int, int, int]:
        """Return a BGR color for a normalized signal value [0, 1]."""
        if value >= 0.8:
            return _COLOR_ALERT  # red
        if value >= 0.5:
            return _COLOR_WARN  # yellow
        return _COLOR_OK  # green

    # ── Landmark drawing ─────────────────────────────────────────

    def _draw_face_landmarks(
        self,
        frame: np.ndarray,
        landmarks_px: np.ndarray,
    ) -> None:
        """Draw key face-mesh landmarks in group-specific colors.

        Groups drawn:
        * Left eye contour  — Cyan
        * Right eye contour — Light Blue
        * Head-pose anchors — White
        """
        num_lm = len(landmarks_px)

        def _safe_pt(idx: int) -> tuple[int, int] | None:
            if idx >= num_lm:
                return None
            x, y = landmarks_px[idx]
            return int(x), int(y)

        # Left eye contour (subject's left).
        prev_pt = None
        for idx in _LM_LEFT_EYE_IDX:
            pt = _safe_pt(idx)
            if pt is None:
                continue
            cv2.circle(frame, pt, 2, _LM_L_EYE, -1, cv2.LINE_AA)
            if prev_pt is not None:
                cv2.line(frame, prev_pt, pt, _LM_L_EYE, 1, cv2.LINE_AA)
            prev_pt = pt
        # Close the contour.
        if len(_LM_LEFT_EYE_IDX) >= 2:
            pt_first = _safe_pt(_LM_LEFT_EYE_IDX[0])
            pt_last = _safe_pt(_LM_LEFT_EYE_IDX[-1])
            if pt_first and pt_last:
                cv2.line(frame, pt_last, pt_first, _LM_L_EYE, 1, cv2.LINE_AA)

        # Right eye contour (subject's right).
        prev_pt = None
        for idx in _LM_RIGHT_EYE_IDX:
            pt = _safe_pt(idx)
            if pt is None:
                continue
            cv2.circle(frame, pt, 2, _LM_R_EYE, -1, cv2.LINE_AA)
            if prev_pt is not None:
                cv2.line(frame, prev_pt, pt, _LM_R_EYE, 1, cv2.LINE_AA)
            prev_pt = pt
        if len(_LM_RIGHT_EYE_IDX) >= 2:
            pt_first = _safe_pt(_LM_RIGHT_EYE_IDX[0])
            pt_last = _safe_pt(_LM_RIGHT_EYE_IDX[-1])
            if pt_first and pt_last:
                cv2.line(frame, pt_last, pt_first, _LM_R_EYE, 1, cv2.LINE_AA)

        # Head-pose anchor points.
        for idx in _LM_HEAD_POSE_IDX:
            pt = _safe_pt(idx)
            if pt is not None:
                cv2.circle(frame, pt, 3, _LM_POSE_ANCHOR, -1, cv2.LINE_AA)

        # Inner-lip contour (Spring Green) — shows MAR landmark positions.
        prev_pt = None
        for idx in _LM_INNER_LIP_IDX:
            pt = _safe_pt(idx)
            if pt is None:
                continue
            cv2.circle(frame, pt, 1, _LM_MOUTH, -1, cv2.LINE_AA)
            if prev_pt is not None:
                cv2.line(frame, prev_pt, pt, _LM_MOUTH, 1, cv2.LINE_AA)
            prev_pt = pt
        # Close the contour.
        if len(_LM_INNER_LIP_IDX) >= 2:
            pt_first = _safe_pt(_LM_INNER_LIP_IDX[0])
            pt_last = _safe_pt(_LM_INNER_LIP_IDX[-1])
            if pt_first and pt_last:
                cv2.line(frame, pt_last, pt_first, _LM_MOUTH, 1, cv2.LINE_AA)

    def _draw_pose_skeleton(
        self,
        frame: np.ndarray,
        pose_landmarks_px: np.ndarray,
    ) -> None:
        """Draw shoulder–nose skeleton on the body.

        Points drawn:
        * Nose            — Orange dot
        * Left shoulder   — Magenta dot
        * Right shoulder  — Magenta dot
        * Shoulder midpoint — Magenta hollow circle
        * Left–right shoulder line — Magenta
        * Shoulder-midpoint → nose line — Pink
        """
        if len(pose_landmarks_px) <= _POSE_RIGHT_SHOULDER:
            return

        def _pt(idx: int) -> tuple[int, int]:
            return int(pose_landmarks_px[idx][0]), int(pose_landmarks_px[idx][1])

        nose_pt = _pt(_POSE_NOSE)
        left_shldr = _pt(_POSE_LEFT_SHOULDER)
        right_shldr = _pt(_POSE_RIGHT_SHOULDER)

        mid_x = (left_shldr[0] + right_shldr[0]) // 2
        mid_y = (left_shldr[1] + right_shldr[1]) // 2
        mid_pt = (mid_x, mid_y)

        # Shoulder–shoulder line.
        cv2.line(frame, left_shldr, right_shldr, _LM_SHOULDER, 2, cv2.LINE_AA)

        # Shoulder midpoint → nose line (Pink).
        cv2.line(frame, mid_pt, nose_pt, _LM_SHLDR_NOSE_LINE, 2, cv2.LINE_AA)

        # Shoulder dots.
        cv2.circle(frame, left_shldr, 5, _LM_SHOULDER, -1, cv2.LINE_AA)
        cv2.circle(frame, right_shldr, 5, _LM_SHOULDER, -1, cv2.LINE_AA)

        # Shoulder midpoint (hollow circle).
        cv2.circle(frame, mid_pt, 4, _LM_SHOULDER, 1, cv2.LINE_AA)

        # Nose point (Orange).
        cv2.circle(frame, nose_pt, 5, _LM_NOSE, -1, cv2.LINE_AA)
