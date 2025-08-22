import threading
import queue
import time
import pyttsx3


class TTSEngine:
	"""Threaded TTS engine that de-duplicates consecutive messages."""

	def __init__(self, rate: int = 180, volume: float = 1.0, voice_index: int | None = None):
		self._engine = pyttsx3.init()
		self._engine.setProperty("rate", rate)
		self._engine.setProperty("volume", volume)
		if voice_index is not None:
			voices = self._engine.getProperty("voices")
			if 0 <= voice_index < len(voices):
				self._engine.setProperty("voice", voices[voice_index].id)

		self._queue: "queue.Queue[str]" = queue.Queue()
		self._thread = threading.Thread(target=self._worker, daemon=True)
		self._last_spoken: str | None = None
		self._running = True
		self._thread.start()

	def _worker(self) -> None:
		while self._running:
			try:
				text = self._queue.get(timeout=0.1)
			except queue.Empty:
				continue
			if not text:
				continue
			if text == self._last_spoken:
				continue
			self._last_spoken = text
			try:
				self._engine.say(text)
				self._engine.runAndWait()
			except Exception:
				# Avoid crashing on TTS errors; wait briefly
				time.sleep(0.1)

	def speak(self, text: str) -> None:
		if not text:
			return
		self._queue.put(text)

	def stop(self) -> None:
		self._running = False
		# Drain to unblock thread
		self._queue.put("")
		try:
			self._thread.join(timeout=1.0)
		except Exception:
			pass
		try:
			self._engine.stop()
		except Exception:
			pass
