import cv2
import numpy as np
from ultralytics import YOLO
import re
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

model = YOLO("yolo11n.pt")

def alpr_detection(vehicle_crop: np.ndarray) -> str:
    h, w = vehicle_crop.shape[:2]
    
    if h < 60 or w < 120:
        return "TOO_SMALL"
    
    results = reader.readtext(vehicle_crop)
    
    if not results:
        return "TOO_SMALL"
     
    print("\n[OCR RAW]")
    for (bbox, text, conf) in results:
        print(f"  '{text}'  conf={conf:.2f}")
        
    best = max(results, key=lambda r: r[2])
    text = best[1]    
    
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    if len(text) < 4:
        return "NO_PLATE_FOUND"
    
    return text

try:
    results = model.predict(
    source=0,
    show=True,
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
            vehicle_crop = frame[y1:y2, x1:x2] # crop vehicle for ALPR processing
            
            plate_number = alpr_detection(vehicle_crop)
            
            if plate_number != "TOO_SMALL":
                cv2.putText(
                    annotated_frame,
                    plate_number,
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
    print("Por favor, verifica tu URL RTSP, credenciales de la cámara y conectividad de red.")    
    