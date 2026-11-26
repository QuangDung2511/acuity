# Acuity

A desktop app that watches the user through the webcam during focus sessions and detects fatigue and distraction, entirely on the user's machine.

## Camera

**Frame source**:
Where the pipeline's images come from: a webcam, or a video file standing in for one in tests and replays.
_Avoid_: capture, feed, stream

**Camera adapter**:
The platform-specific part that decides how the webcam is opened on each operating system, and whether it can be opened there at all.
_Avoid_: camera driver, backend (when meaning the whole adapter)
