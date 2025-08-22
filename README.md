## YOLOv8n-based Navigation Announcer (Laptop Camera + Offline TTS)

This project uses the laptop camera with YOLOv8n for continuous object detection and an offline text-to-speech engine to announce object positions like "pole on the right" or "person ahead". It only speaks when the scene changes to avoid spam.

### Features
- YOLOv8n object detection (auto-downloads `yolov8n.pt` on first run)
- Left / center / right spatial cues based on object positions
- Offline TTS via `pyttsx3` (works without Internet)
- Announces only on scene changes, with periodic refresh
- Simple on-screen visualization for debugging

### Requirements
- Python 3.10+
- A working webcam
- Windows (PowerShell) tested; should also work on Linux/macOS

### Setup (Windows PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If pip builds wheels slowly on Windows, consider installing the prebuilt `opencv-python` wheel first, then others.

### Run
```powershell
python -m src.main
```
Press `q` in the preview window to quit. You can also close with Ctrl+C in the terminal.

### Notes
- The first run will download `yolov8n.pt` automatically from Ultralytics.
- You can tune thresholds in `src/main.py`: `CONF_THRESHOLD`, `IOU_THRESHOLD`, and the `cooldown_seconds` in `SceneChangeAnnouncer`.
- The TTS voice can be changed in `src/tts.py` by selecting a different `voice_index`.

### Troubleshooting
- If the camera cannot be opened, try changing the index in `cv2.VideoCapture(0)` to `1` or `2`.
- If you get GPU/CUDA errors and you're on CPU, ensure the Ultralytics install uses CPU by default (no extra config needed); performance on CPU is still fine with `yolov8n`.
- If you don't hear speech, check your system sound output device and that `pyttsx3` is installed correctly.
