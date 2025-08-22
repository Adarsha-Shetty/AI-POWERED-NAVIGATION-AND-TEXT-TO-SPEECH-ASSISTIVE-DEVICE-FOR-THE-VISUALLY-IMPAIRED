from __future__ import annotations
from typing import List, Tuple, Dict


def assign_side(x_center: float, width: int) -> str:
	ratio = x_center / max(width, 1)
	if ratio < 1 / 3:
		return "left"
	if ratio > 2 / 3:
		return "right"
	return "center"


def distance_label(box_height: float, frame_height: int) -> str:
	"""Heuristic distance categories based on relative box height."""
	frac = box_height / max(frame_height, 1)
	if frac >= 0.60:
		return "extremely close"
	if frac >= 0.40:
		return "very close"
	if frac >= 0.25:
		return "close"
	if frac >= 0.12:
		return "medium distance"
	return "far away"


def summarize_scene(
	detections: List[Tuple[str, float, Tuple[float, float, float, float]]],
	frame_width: int,
	frame_height: int | None = None,
) -> str:
	if not detections:
		return "No objects ahead"
	# Keep top by confidence to limit noise
	top = sorted(detections, key=lambda d: d[1], reverse=True)[:12]
	# For each side, choose the nearest object (largest box height)
	best_per_side: Dict[str, Tuple[str, float, Tuple[float, float, float, float]]] = {}
	for cls, conf, (x1, y1, x2, y2) in top:
		cx = (x1 + x2) / 2.0
		side = assign_side(cx, frame_width)
		box_h = max(0.0, y2 - y1)
		prev = best_per_side.get(side)
		if prev is None:
			best_per_side[side] = (cls, box_h, (x1, y1, x2, y2))
		else:
			_, prev_h, _ = prev
			if box_h > prev_h:
				best_per_side[side] = (cls, box_h, (x1, y1, x2, y2))
	phrases: List[str] = []
	fh = frame_height if frame_height is not None else int(frame_width * 3 / 4)
	for side in ("left", "center", "right"):
		item = best_per_side.get(side)
		if not item:
			continue
		cls, box_h, _ = item
		dist = distance_label(box_h, fh)
		if side == "center":
			phrases.append(f"{cls} {dist} ahead")
		else:
			phrases.append(f"{cls} {dist} on the {side}")
	return "; ".join(phrases)


class SceneChangeAnnouncer:
	def __init__(self, cooldown_seconds: float = 4.0, refresh_seconds: float = 12.0):
		self._last_description: str | None = None
		self._last_time: float = 0.0
		self._cooldown = cooldown_seconds
		self._refresh = refresh_seconds

	def should_announce(self, description: str, now_time: float) -> bool:
		if not description:
			return False
		if self._last_description is None:
			self._last_description = description
			self._last_time = now_time
			return True
		if description != self._last_description and (now_time - self._last_time) >= self._cooldown:
			self._last_description = description
			self._last_time = now_time
			return True
		if (now_time - self._last_time) >= self._refresh and description == self._last_description:
			self._last_time = now_time
			return True
		return False
