import cv2
import numpy as np
from ultralytics import YOLO
import re
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Ariana\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

VEHICLE_MODEL_PATH = "yolo11n.pt"
PLATE_MODEL_PATH   = "license-plate-finetune-v1n.pt"

PLATE_REGEX = re.compile(r"^\d{4}[A-Z]{3}$")

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model   = YOLO(PLATE_MODEL_PATH)


def normalize_text(raw: str) -> str | None:
    s = re.sub(r'[^A-Z0-9]', '', raw.upper())

    if len(s) < 5:
        return None

    s = s[:7]
    if len(s) != 7:
        return None
    
    chars = list(s)

    digit_like = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1',
        'Z': '2',
        'S': '5',
        'G': '6',
        'T': '7',
        'B': '8',
    }

    letter_like = {
        '0': 'O',
        '1': 'I',  
        '2': 'Z',
        '5': 'S',
        '6': 'G',
        '7': 'T',
        '8': 'B',
    }

    for i in range(4):
        if chars[i].isalpha() and chars[i] in digit_like:
            chars[i] = digit_like[chars[i]]
        elif not chars[i].isdigit():
            return None

    for i in range(4, 7):
        if chars[i].isdigit() and chars[i] in letter_like:
            chars[i] = letter_like[chars[i]]
        elif not chars[i].isalpha():
            return None

    s_norm = ''.join(chars)
    return s_norm


def preprocess_plate(plate_crop: np.ndarray) -> np.ndarray:
    """Preprocesamiento optimizado para placas"""

    h, w = plate_crop.shape[:2]
    if h < 60:
        scale = 60 / h
        plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale, 
                                interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # binarización adaptativa
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    return binary


def alpr_from_plate_crop(plate_crop: np.ndarray) -> str:
    if plate_crop.size == 0:
        return "NO_PLATE_FOUND"

    proc = preprocess_plate(plate_crop)
    
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    raw_text = pytesseract.image_to_string(proc, config=custom_config).strip()
    
    if not raw_text:
        return "NO_PLATE_FOUND"
    
    norm = normalize_text(raw_text)
    
    if not norm:
        return "NO_PLATE_FOUND"
    
    if PLATE_REGEX.fullmatch(norm):
        return norm
    
    return "NO_PLATE_FOUND"


def main():
    try:
        results = vehicle_model.predict(
            source=0,
            show=False,
            stream=True,
            verbose=False,
            classes=[2, 3, 5, 6, 7],
            conf=0.35
        )

        print("Starting webcam detection...")
        print("Press 'q' to quit")
        
        frame_count = 0

        for r in results:
            frame = r.orig_img
            if frame is None:
                continue
            
            frame_count += 1
            annotated_frame = r.plot()

            for vbox in r.boxes:
                vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0].tolist())
                vx1 = max(0, vx1); vy1 = max(0, vy1)
                vx2 = min(frame.shape[1], vx2)
                vy2 = min(frame.shape[0], vy2)

                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                if vehicle_crop.size == 0:
                    continue

                if frame_count % 3 != 0:
                    continue

                plate_res = plate_model.predict(
                    source=vehicle_crop,
                    verbose=False,
                    conf=0.25, 
                    imgsz=640  
                )[0]

                print(f"Found {len(plate_res.boxes)} potential plates")

                for pbox in plate_res.boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                    px1 = max(0, px1); py1 = max(0, py1)
                    px2 = min(vehicle_crop.shape[1], px2)
                    py2 = min(vehicle_crop.shape[0], py2)

                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    
                    gx1, gy1 = vx1 + px1, vy1 + py1
                    gx2, gy2 = vx1 + px2, vy1 + py2
                    
                    cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                    
                    plate_text = alpr_from_plate_crop(plate_crop)
                    print(f"OCR Result: {plate_text}")

                    if plate_text != "NO_PLATE_FOUND":
                        cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
                        cv2.putText(
                            annotated_frame,
                            plate_text,
                            (gx1, gy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )

            cv2.imshow("Vehicle + Plate Detection (ALPR)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("\nWebcam detection ended.")

    except Exception as e:
        print(f"Ocurrió un error durante la detección: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()