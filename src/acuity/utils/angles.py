# Circular (angular) math helpers for head-pose handling.
# ─────────────────────────────────────────────────────────────────
# Head-pose Euler angles live on a circle, not a line: −179° and
# +179° are 2° apart, not 358°.  Plain arithmetic (subtraction,
# mean, median, EMA) silently produces garbage whenever a signal
# sits near the ±180° wrap boundary.
#
# Every comparison of a head-pose angle against a calibrated
# baseline must go through :func:`wrap_deg` so the result is the
# true shortest-arc difference.
#
# Used by:
#   - face_mesh.py         (pitch normalisation + smoothing)
#   - calibration.py       (baseline pitch/yaw)
#   - fatigue_engine.py    (EAR gate, head-nod detection)
#   - distraction_engine.py(head-pose deviation)
#   - cv_pipeline.py       (debug/CSV pitch_gated column)
#
# Thread: any (pure functions).
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["wrap_deg", "circular_median_deg", "ema_angle_deg"]


def wrap_deg(angle_deg: float) -> float:
    """Wrap an angle (degrees) into the half-open range ``[-180, 180)``.

    This is the shortest-arc representation.  Use it on every
    *difference* between two angles::

        drop = wrap_deg(baseline_pitch - head_pitch)

    Without it, a 6° head movement across the wrap boundary reads as
    a 354° movement and trips every threshold downstream.

    Parameters
    ----------
    angle_deg : float
        Any angle in degrees.

    Returns
    -------
    float
        Equivalent angle in ``[-180, 180)``.
    """
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def circular_median_deg(samples: Sequence[float]) -> float:
    """Robust median of angular samples (degrees), wrap-safe.

    A plain :func:`numpy.median` over angles straddling the ±180°
    boundary returns a value near 0° — the exact opposite of the
    true central direction.  This function instead:

    1. Computes the circular mean direction via ``atan2`` of the
       mean sine/cosine (immune to wrapping).
    2. Unwraps every sample relative to that direction.
    3. Takes the ordinary median of the unwrapped values (retaining
       the outlier resistance the median was chosen for).
    4. Wraps the result back into ``[-180, 180)``.

    Parameters
    ----------
    samples : Sequence[float]
        Angular samples in degrees.  Must be non-empty.

    Returns
    -------
    float
        The circular median in ``[-180, 180)``.
    """
    if len(samples) == 0:
        raise ValueError("circular_median_deg requires at least one sample.")

    radians = np.radians(np.asarray(samples, dtype=np.float64))
    mean_dir = math.atan2(
        float(np.mean(np.sin(radians))),
        float(np.mean(np.cos(radians))),
    )
    mean_dir_deg = math.degrees(mean_dir)

    # Unwrap each sample onto a line centred on the mean direction,
    # then take the plain median there.
    unwrapped = [mean_dir_deg + wrap_deg(s - mean_dir_deg) for s in samples]
    return wrap_deg(float(np.median(unwrapped)))


def ema_angle_deg(previous_deg: float, new_deg: float, alpha: float) -> float:
    """Wrap-safe exponential moving average for an angular signal.

    A linear EMA blended across the ±180° boundary sweeps *the long
    way round* — a 2° real movement drags the smoothed output through
    ~140°/frame excursions before it settles.  Blending the wrapped
    *difference* instead always takes the short arc.

    Parameters
    ----------
    previous_deg : float
        Previous smoothed value (degrees).
    new_deg : float
        Newest raw observation (degrees).
    alpha : float
        Smoothing factor in ``[0, 1]``.  Higher = more responsive.

    Returns
    -------
    float
        Updated smoothed angle in ``[-180, 180)``.
    """
    delta = wrap_deg(new_deg - previous_deg)
    return wrap_deg(previous_deg + alpha * delta)
