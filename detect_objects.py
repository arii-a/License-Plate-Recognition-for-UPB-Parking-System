import cv2
import numpy as np
from ultralytics import YOLO
import re
import easyocr
from PIL import Image

reader = easyocr.Reader(['en'], gpu=False)

PLATE_REGEX = re.compile(r'^\d{4}[A-Z]{3}$')

model = YOLO("yolo11n.pt")

def normalize_text(raw: str) -> str | None:
    s = re.sub(r'[^A-Z0-9]', '', raw.upper())
    if len(s) < 5:  
        return None
    
    s = s[:7]
    chars = list(s)
    
    digit_like = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1',
        'Z': '2',
        'S': '5',
        'G': '6',
        'Z': '7', 'T': '7',
        'B': '8',
    }
    
    letter_like = {
        '0': 'O',
        '1': 'I', '1': 'L',
        '2': 'Z',
        '5': 'S',
        '6': 'G',
        '7': 'T', '7': 'Z',
        '8': 'B',
    }
    
    for i in range(min(4, len(chars))):
        if chars[i].isalpha() and chars[i] in digit_like:
            chars[i] = digit_like[chars[i]]

    for i in range(4, min(7, len(chars))):
        if chars[i].isdigit() and chars[i] in letter_like:
            chars[i] = letter_like[chars[i]]

    s_norm = ''.join(chars)

    if len(s_norm) != 7:
        return None

    return s_norm

def alpr_detection(vehicle_crop: np.ndarray) -> str:
    h, w = vehicle_crop.shape[:2]
    
    if h < 60 or w < 120:
        return "TOO_SMALL"
    
    # supposed roi
    y1 = int(h * 0.55)
    y2 = int(h * 0.95)
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    
    
    roi = vehicle_crop[y1:y2, x1:x2]
    
    if roi.size == 0:
        roi = vehicle_crop
        
    roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    results = reader.readtext(roi)    
    
    if not results:
        return "NO_PLATE_FOUND"
     
    print("\n[OCR RAW]")
    
    candidates = []
    
    for (bbox, raw_text, conf) in results:
        norm = normalize_text(raw_text)
        if not norm:
            continue
        
        score = conf
        
        if PLATE_REGEX.fullmatch(norm):
            score += 0.5
            
        candidates.append((score, norm))
        
    if not candidates:
        return "NO_PLATE_FOUND"
        
    best_score, best_text = max(candidates, key=lambda c: c[0])[:2]
    return best_text    

try:
    results = model.predict(
        source=0,
        show=False,
        stream=True,
        verbose=False,
        classes=[2, 3, 5, 6, 7] # detect only cars, motorcycles, buses, trucks, and bicycles
    )
    
    print("Starting webcam detection...")
    print("Press 'q' to quit")
    
    for r in results:
        frame = r.orig_img
        
        if frame is None:
            continue
        
        annotated_frame = r.plot() 
        
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) # vehicle coordinates
            
            # ensure coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            vehicle_crop = frame[y1:y2, x1:x2]
            
            plate_text = alpr_detection(vehicle_crop)
            
            if plate_text not in ["TOO_SMALL", "NO_PLATE_FOUND"]:
                cv2.putText(
                    annotated_frame,
                    plate_text,
                    (x1, y1 - 10), # text position above the box
                    cv2.FONT_HERSHEY_SIMPLEX, # font
                    0.7, # font scale
                    (0, 255, 0), # colour 
                    2 # thickness
                )
        
                
        cv2.imshow("Vehicle and ALPR Detection", annotated_frame)
                
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    cv2.destroyAllWindows()    
    print("\nWebcam detection ended.")
    
except Exception as e:
    print(f"Ocurrió un error durante la detección: {e}")    