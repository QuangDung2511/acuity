# Acuity

A desktop focus companion that watches you through your webcam during focus sessions and tells you when you show signs of fatigue or drift away from your work.

> **Status: pre-release.** The code is being ported from a private prototype. Nothing below is installable yet.

## What it does

- **Fatigue:** warns on signs of tiredness (eyelid closure over time, slow blinks, yawning), and alerts on sleep or a microsleep.
- **Distraction:** notices when you turn away from the screen for a sustained period, or switch to an app or site you haven't marked as focus work.
- **Sessions:** structured focus/rest cycles, a short calibration at the start of each focus block, and an Insights tab comparing events per focus hour.

## Privacy

- **Everything runs on your computer.** Acuity makes no network calls. Video frames are analysed in memory and never saved.
- **What is stored** (in your user data folder, e.g. `%APPDATA%\Acuity` on Windows):
  - a local database of sessions, focus blocks, rests and detected events (type, time, the detector's reason, and your rating if you give one);
  - for window tracking, the *app name* only (e.g. `chrome.exe`). Window titles are never stored unless you turn on "log window titles";
  - in debug mode only, per-frame numeric measurements (eye openness, head angles, scores) as CSV. These contain no images.
- Delete that folder and every trace of Acuity's data is gone.

## Caveat: tuned on one person (n=1)

Every detection threshold was tuned on **one person, one camera and one desk**. Calibration makes each threshold relative to *your* resting baseline, which helps, but different faces, glasses, lighting and camera angles will behave differently. Expect false alarms and misses until you check it on yourself (next section).

## Check its accuracy on yourself

Turn on **Settings → "Ask me to rate alerts at each rest"**. At each rest, and at the end of a session, Acuity lists the alerts from the block you just finished and asks whether each one was right or a false alarm. Once a detector has at least five ratings, the Insights tab shows its precision for you.

If a detector is consistently wrong for you, please open an issue with the precision figure and your setup (camera position, glasses, lighting). Ratings stay on your machine unless you choose to share them.

## Licence

MIT. See [LICENSE](LICENSE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for bundled components.
