# Synthesise the bundled default alert sounds into assets/sounds/.
# ─────────────────────────────────────────────────────────────────
# Run from the repo root:
#
#     python tools/generate_default_sounds.py
#
# Writes five 16-bit / 44.1 kHz mono WAV files:
#
#     distraction.wav      two quick rising blips  — "hey, look back"
#     fatigue_warning.wav  soft descending chime   — gentle nudge
#     fatigue.wav          three urgent rising beeps — full alert
#     rest_start.wav       resolving major arpeggio — "you may stop"
#     rest_end.wav         the same arpeggio inverted — "come back"
#
# They are deliberately distinct along three axes (pitch contour,
# rhythm, and timbre) so they can be told apart without looking at the
# screen.  Warning-tier sounds are quieter and softer-edged than
# alert-tier ones.  The two rest sounds are each other's mirror image
# because they are heard in the same context and mean opposite things.
#
# Only the standard library is used, so this runs anywhere without the
# project's CV dependencies installed.  The generated files are what
# ships; regenerate them only if you want to retune the defaults.
#
# Thread: standalone CLI (no threads, no Qt).
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 44_100
BIT_DEPTH = 2  # bytes per sample (16-bit)

_OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "sounds"

# Harmonic recipes: (multiple of fundamental, relative amplitude).
# More upper harmonics = harder / more urgent edge.
_SOFT = ((1.0, 1.0), (2.0, 0.12))  # near-pure sine
_WARM = ((1.0, 1.0), (2.0, 0.35), (3.0, 0.12))  # mellow, marimba-ish
_BRIGHT = ((1.0, 1.0), (2.0, 0.5), (3.0, 0.3))  # crisp blip
_EDGY = ((1.0, 1.0), (3.0, 0.4), (5.0, 0.22))  # odd harmonics — urgent


def _tone(
    freq: float,
    duration_s: float,
    amp: float = 0.6,
    harmonics: tuple[tuple[float, float], ...] = _SOFT,
    attack_s: float = 0.006,
    release_s: float = 0.06,
    decay: float = 0.0,
    freq_end: float | None = None,
) -> list[float]:
    """Return one enveloped tone as a list of samples in [-1, 1].

    ``decay`` applies an exponential amplitude falloff (0 = sustained,
    higher = pluckier).  ``freq_end`` glides the pitch linearly from
    *freq* to *freq_end* across the tone.
    """
    n = int(SAMPLE_RATE * duration_s)
    attack_n = max(1, int(SAMPLE_RATE * attack_s))
    release_n = max(1, int(SAMPLE_RATE * release_s))
    norm = sum(a for _, a in harmonics)

    samples: list[float] = []
    phase = 0.0
    for i in range(n):
        t = i / n
        f = freq if freq_end is None else freq + (freq_end - freq) * t
        phase += 2.0 * math.pi * f / SAMPLE_RATE

        value = sum(a * math.sin(mult * phase) for mult, a in harmonics) / norm

        # Attack / release ramps keep the edges click-free.
        env = 1.0
        if i < attack_n:
            env *= i / attack_n
        if i > n - release_n:
            env *= (n - i) / release_n
        if decay:
            env *= math.exp(-decay * t)

        samples.append(value * env * amp)
    return samples


def _silence(duration_s: float) -> list[float]:
    return [0.0] * int(SAMPLE_RATE * duration_s)


def _write_wav(path: Path, samples: list[float], peak: float = 0.85) -> None:
    """Normalise to *peak* and write a 16-bit mono WAV."""
    loudest = max((abs(s) for s in samples), default=0.0)
    gain = (peak / loudest) if loudest > 0 else 0.0
    frames = b"".join(
        struct.pack("<h", int(max(-1.0, min(1.0, s * gain)) * 32767)) for s in samples
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BIT_DEPTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(frames)
    print(f"  {path.name:<22} {len(samples) / SAMPLE_RATE:.2f}s")


# ── The default sounds ───────────────────────────────────────────


def distraction() -> list[float]:
    """Two quick rising blips — short, crisp, easy to ignore once seen."""
    return (
        _tone(784.0, 0.085, amp=0.55, harmonics=_BRIGHT, release_s=0.03)
        + _silence(0.045)
        + _tone(1046.5, 0.10, amp=0.55, harmonics=_BRIGHT, release_s=0.04)
    )


def fatigue_warning() -> list[float]:
    """Soft descending two-note chime — a nudge, not an alarm."""
    return (
        _tone(587.33, 0.26, amp=0.42, harmonics=_SOFT, attack_s=0.035, release_s=0.11, decay=0.7)
        + _silence(0.03)
        + _tone(440.0, 0.42, amp=0.42, harmonics=_SOFT, attack_s=0.035, release_s=0.20, decay=0.7)
    )


def fatigue_alert() -> list[float]:
    """Three rising urgent beeps — the loudest, hardest-edged sound."""
    out: list[float] = []
    for freq in (880.0, 987.77, 1108.73):
        out += _tone(freq, 0.13, amp=0.9, harmonics=_EDGY, attack_s=0.004, release_s=0.03)
        out += _silence(0.065)
    # Final sustained tone so the alert reads as more than a triple blip.
    out += _tone(1108.73, 0.30, amp=0.9, harmonics=_EDGY, attack_s=0.004, release_s=0.12)
    return out


def rest_start() -> list[float]:
    """Resolving major arpeggio — 'block complete, you may stop'.

    The other three sounds are all warnings: they interrupt to correct
    something.  This one marks an earned transition, so it resolves
    upward and rings out instead of stopping short.  Longer and more
    consonant than any alert, which keeps it unmistakable when heard
    from across the room with the screen out of view.
    """
    return (
        _tone(523.25, 0.16, amp=0.5, harmonics=_SOFT, attack_s=0.012, release_s=0.05, decay=0.9)
        + _tone(659.25, 0.16, amp=0.5, harmonics=_SOFT, attack_s=0.012, release_s=0.05, decay=0.9)
        + _tone(783.99, 0.55, amp=0.55, harmonics=_WARM, attack_s=0.012, release_s=0.30, decay=1.1)
    )


def rest_end() -> list[float]:
    """Descending major arpeggio — 'rest is over, come back'.

    Deliberately the mirror of rest_start(): same three notes, same
    timbre, reversed contour (G-E-C instead of C-E-G) and a shorter,
    firmer final note instead of a long ring-out.  The two are heard in
    the same place at opposite ends of a rest, often from across the
    room with the screen out of view, so sharing a voice while inverting
    the gesture is what makes them tellable apart without being learned
    as two unrelated sounds.  Slightly brighter than rest_start so it
    carries — this one has to reach someone who walked away.
    """
    return (
        _tone(783.99, 0.16, amp=0.55, harmonics=_SOFT, attack_s=0.012, release_s=0.05, decay=0.9)
        + _tone(659.25, 0.16, amp=0.55, harmonics=_SOFT, attack_s=0.012, release_s=0.05, decay=0.9)
        + _tone(523.25, 0.34, amp=0.6, harmonics=_WARM, attack_s=0.012, release_s=0.14, decay=1.4)
    )


_SOUNDS = {
    "distraction.wav": distraction,
    "fatigue_warning.wav": fatigue_warning,
    "fatigue.wav": fatigue_alert,
    "rest_start.wav": rest_start,
    "rest_end.wav": rest_end,
}


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing default alert sounds to {_OUT_DIR}")
    for name, builder in _SOUNDS.items():
        _write_wav(_OUT_DIR / name, builder())
    print("Done.")


if __name__ == "__main__":
    main()
