import re
from pathlib import Path

CONFIG_DIR = Path(__file__).parent

VEHICLE_MODEL_PATH = str(CONFIG_DIR / "yolo11n.pt")
PLATE_MODEL_PATH   = str(CONFIG_DIR / "license-plate-finetune-v1n.pt")
TESSERACT_PATH = r'C:\Users\Ariana\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

PLATE_REGEX = re.compile(r"^\d{4}[A-Z]{3}$")

# OCR config
OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

