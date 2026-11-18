# Tests for circular (angular) math helpers.
# ─────────────────────────────────────────────────────────────────
# Regression coverage for the head-pose wraparound defect: pitch
# decomposes to ≈ ±180° for a user facing the camera, so every
# small head movement crossed the wrap boundary and plain
# arithmetic reported it as a ~360° swing.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from acuity.utils.angles import circular_median_deg, ema_angle_deg, wrap_deg

# ── wrap_deg ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        (0.0, 0.0),
        (10.0, 10.0),
        (-10.0, -10.0),
        (179.0, 179.0),
        (-180.0, -180.0),
        (180.0, -180.0),  # half-open range [-180, 180)
        (354.0, -6.0),
        (-354.0, 6.0),
        (720.0, 0.0),
        (-359.0, 1.0),
    ],
)
def test_wrap_deg_maps_into_half_open_range(raw, expected):
    assert wrap_deg(raw) == pytest.approx(expected, abs=1e-9)


def test_wrap_deg_output_always_in_range():
    for raw in np.linspace(-1080.0, 1080.0, 4001):
        assert -180.0 <= wrap_deg(raw) < 180.0


def test_wrap_deg_recovers_true_delta_across_boundary():
    """The original defect: baseline −176°, head at +178°.

    The real movement is 6°; naive subtraction reports −354°, which
    trips the 25° EAR gate, the 8° nod threshold and the 50° pose
    tolerance all at once.
    """
    baseline, current = -176.0, 178.0
    assert baseline - current == pytest.approx(-354.0)  # the bug
    assert wrap_deg(baseline - current) == pytest.approx(6.0)


# ── circular_median_deg ─────────────────────────────────────────


def test_circular_median_matches_plain_median_away_from_boundary():
    samples = [-4.0, -5.0, -5.5, -6.0, -5.2]
    assert circular_median_deg(samples) == pytest.approx(float(np.median(samples)), abs=1e-6)


def test_circular_median_survives_the_wrap_boundary():
    """Samples straddling ±180° must not collapse toward 0°."""
    samples = [179.0, -179.0, 178.0, -178.0, 180.0]
    assert float(np.median(samples)) == pytest.approx(178.0)  # plain median
    assert abs(circular_median_deg(samples)) == pytest.approx(180.0, abs=2.0)


def test_circular_median_is_outlier_resistant():
    samples = [-5.0] * 20 + [120.0, -140.0]
    assert circular_median_deg(samples) == pytest.approx(-5.0, abs=1.0)


def test_circular_median_single_sample():
    assert circular_median_deg([42.0]) == pytest.approx(42.0)


def test_circular_median_rejects_empty():
    with pytest.raises(ValueError):
        circular_median_deg([])


# ── ema_angle_deg ───────────────────────────────────────────────


def test_ema_angle_matches_linear_blend_away_from_boundary():
    assert ema_angle_deg(10.0, 20.0, 0.4) == pytest.approx(14.0, abs=1e-9)


def test_ema_angle_takes_short_arc_across_boundary():
    """A linear EMA sweeps the long way through 0°, injecting
    excursions far larger than any real head movement."""
    smoothed = ema_angle_deg(-179.0, 179.0, 0.4)
    # True movement is 2°; the step must stay within it.
    assert abs(wrap_deg(smoothed - (-179.0))) <= 2.0
    assert abs(smoothed) > 175.0  # stayed near the boundary


def test_ema_angle_converges_without_excursion():
    smoothed = -179.0
    for _ in range(25):
        smoothed = ema_angle_deg(smoothed, 179.0, 0.4)
        assert abs(smoothed) > 170.0  # never sweeps through 0°
    assert wrap_deg(smoothed - 179.0) == pytest.approx(0.0, abs=1.0)


def test_ema_angle_output_always_in_range():
    smoothed = 0.0
    for target in (170.0, -170.0, 179.9, -179.9, 5.0):
        for _ in range(10):
            smoothed = ema_angle_deg(smoothed, target, 0.3)
            assert -180.0 <= smoothed < 180.0
