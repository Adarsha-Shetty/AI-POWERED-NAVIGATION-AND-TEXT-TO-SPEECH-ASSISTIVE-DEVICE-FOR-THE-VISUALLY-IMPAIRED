import pytesseract
import cv2
from config import TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

def extract_text_frame(frame):
    text = pytesseract.image_to_string(frame)
    return text.strip()
