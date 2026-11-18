# Third-party notices

Acuity is MIT-licensed. It depends on, and release builds bundle, the components below under their own licences.

| Component | Licence | Notes |
|---|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) and its Face Landmarker / Pose Landmarker models (`assets/models/*.task`) | Apache-2.0 | Model files redistributed unmodified |
| [Qt](https://www.qt.io/) via [PySide6](https://doc.qt.io/qtforpython-6/) | LGPL-3.0 | Dynamically linked; release builds ship the Qt libraries as separate, replaceable files |
| [OpenCV](https://opencv.org/) (`opencv-python`) | Apache-2.0 | |
| [pyqtgraph](https://www.pyqtgraph.org/) | MIT | |
| [NumPy](https://numpy.org/) | BSD-3-Clause | |

Full licence texts are included with each package's distribution and in release builds.
