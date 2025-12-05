import re
import pytesseract
import numpy as np
from conf.settings import TESSERACT_PATH, PLATE_REGEX, OCR_CONFIG
from processors.image_processor import ImageProcessor

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

class TextNormalizer:
    DIGIT_LIKE = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1',
        'Z': '2', 'S': '5', 'G': '6',
        'T': '7', 'B': '8',
    }
    
    LETTER_LIKE = {
        '0': 'O', '1': 'I', '2': 'Z',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B',
    }
    
    @staticmethod
    def normalize_text(raw: str) -> str | None:
        s = re.sub(r'[^A-Z0-9]', '', raw.upper())
        if len(s) < 5:
            return None
        
        s = s[:7]
        if len(s) != 7:
            return None
        
        chars = list(s)
        
        for i in range(4):
            if chars[i].isalpha() and chars[i] in TextNormalizer.DIGIT_LIKE:
                chars[i] = TextNormalizer.DIGIT_LIKE[chars[i]]
            elif not chars[i].isdigit():
                return None
        
        for i in range(4, 7):
            if chars[i].isdigit() and chars[i] in TextNormalizer.LETTER_LIKE:
                chars[i] = TextNormalizer.LETTER_LIKE[chars[i]]
            elif not chars[i].isalpha():
                return None
        
        return ''.join(chars)
    
class OCRProcessor:
    def __init__(self):
        self.image_processor = ImageProcessor()
    
    def read_plate(self, plate_crop: np.ndarray) -> str:        
        processed = self.image_processor.preprocess_plate(plate_crop)
        
        raw_text = pytesseract.image_to_string(
            processed, 
            config=OCR_CONFIG
        ).strip()
        
        if not raw_text:
            return "NO_PLATE_FOUND"
        
        normalized = TextNormalizer.normalize_text(raw_text)
        
        if not normalized:
            return "NO_PLATE_FOUND"
        
        if PLATE_REGEX.fullmatch(normalized):
            return normalized
        
        return "NO_PLATE_FOUND"    