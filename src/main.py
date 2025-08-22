from __future__ import annotations
import time
import signal
import sys
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
	from .tts import TTSEngine
	from .scene_change import summarize_scene, SceneChangeAnnouncer
except ImportError:
	from tts import TTSEngine
	from scene_change import summarize_scene, SceneChangeAnnouncer


MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 640


def gather_detections(result) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
	detections: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
	if result is None or result.boxes is None or len(result.boxes) == 0:
		return detections
	names = result.names
	boxes = result.boxes
	xyxy = boxes.xyxy.cpu().numpy()
	confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((xyxy.shape[0],), dtype=float)
	clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)
	for i in range(xyxy.shape[0]):
		x1, y1, x2, y2 = xyxy[i].tolist()
		conf = float(confs[i])
		cls_id = int(clss[i])
		cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
		detections.append((cls_name, conf, (x1, y1, x2, y2)))
	return detections


def draw_overlays(frame, detections):
	h, w = frame.shape[:2]
	cv2.line(frame, (w // 3, 0), (w // 3, h), (80, 80, 80), 1)
	cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (80, 80, 80), 1)
	for cls, conf, (x1, y1, x2, y2) in detections:
		color = (0, 180, 255)
		cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
		label = f"{cls} {conf:.2f}"
		cv2.putText(frame, label, (int(x1), int(max(0, y1 - 6))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Error: Cannot access camera.")
		sys.exit(1)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	model = YOLO(MODEL_PATH)
	tts = TTSEngine(rate=170)
	announcer = SceneChangeAnnouncer(cooldown_seconds=3.0, refresh_seconds=15.0)
	def handle_sigint(sig, frame):
		raise KeyboardInterrupt
	try:
		signal.signal(signal.SIGINT, handle_sigint)
	except Exception:
		pass
	window_name = "YOLOv8n Navigation (Press Q to quit)"
	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				print("Warning: Failed to read frame from camera.")
				time.sleep(0.05)
				continue
			h, w = frame.shape[:2]
			results = model.predict(
				source=frame,
				imgsz=IMG_SIZE,
				conf=CONF_THRESHOLD,
				iou=IOU_THRESHOLD,
				verbose=False,
				stream=False,
			)
			if not results:
				detections = []
			else:
				detections = gather_detections(results[0])
			description = summarize_scene(detections, w, h)
			now_t = time.time()
			if announcer.should_announce(description, now_t):
				tts.speak(description)
			draw_overlays(frame, detections)
			cv2.putText(
				frame,
				description,
				(10, 28),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 255, 0),
				2,
			)
			cv2.imshow(window_name, frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
	except KeyboardInterrupt:
		pass
	finally:
		try:
			cap.release()
		except Exception:
			pass
		try:
			cv2.destroyAllWindows()
		except Exception:
			pass
		tts.stop()


if __name__ == "__main__":
	main()
