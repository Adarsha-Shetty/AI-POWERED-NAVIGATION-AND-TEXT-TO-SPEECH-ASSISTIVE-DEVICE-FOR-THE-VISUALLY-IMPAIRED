import cv2
import time
from vision.detector import detect_objects_frame
from vision.ocr import extract_text_frame
from speech.tts import speak

def position_label(x_center, frame_width):
    """Return position label based on object's x center."""
    if x_center < frame_width / 3:
        return "on the left"
    elif x_center > (2 * frame_width / 3):
        return "on the right"
    else:
        return "in the center"

def main():
    cap = cv2.VideoCapture(0)
    last_state = ""
    cooldown_time = 2  # seconds between spoken updates
    last_spoken_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects with positions
        objects_with_pos = detect_objects_frame(frame, return_positions=True)

        # OCR (optional, always runs here)
        detected_text = extract_text_frame(frame).strip()

        # Build description
        description_parts = []
        for obj, x_center in objects_with_pos:
            pos = position_label(x_center, frame.shape[1])
            description_parts.append(f"{obj} {pos}")

        if detected_text:
            description_parts.append(f"Text reads: {detected_text}")

        description = ". ".join(description_parts)

        # Speak only if scene changed & cooldown passed
        if description and description != last_state and (time.time() - last_spoken_time) > cooldown_time:
            print(description)
            speak(description)
            last_state = description
            last_spoken_time = time.time()

        # Show live feed
        cv2.imshow("Conversational Glasses", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
