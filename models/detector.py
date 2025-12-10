from ultralytics import YOLO
from conf.settings import VEHICLE_MODEL_PATH, PLATE_MODEL_PATH

import cv2
import numpy as np

class VehicleDetector:
    def __init__(self, model_path=VEHICLE_MODEL_PATH):
        self.model = YOLO(model_path)
    
    def track(self, source, conf, classes, **kwargs):
        return self.model.track(
            source=source,
            persist=True,
            show=False,
            stream=True,
            verbose=False,
            classes=[2, 3, 5, 6, 7],
            conf=0.35
        )
        

class PlateDetector:
    def __init__(self, model_path=PLATE_MODEL_PATH):
        self.model = YOLO(model_path)
    
    def detect(self, source):
        results = self.model.predict(
            source=source,
            verbose=False,
            conf=0.25, 
            imgsz=640  
        )[0]
        
        return results    
    
class ColorDetector:
    def __init__(self, color_map=None):
        if color_map is None:
            self.color_map = {
                'Blanco': [200, 200, 200],  
                'Negro': [10, 10, 10],
                'Gris': [100, 100, 100],
                'Rojo': [0, 0, 200],
                'Azul': [200, 0, 0],
                'Verde': [0, 200, 0],
                'Amarillo': [0, 200, 200]
            }
        else:
            self.color_map = color_map    
            
    def detect_color(self, vehicle_crop: np.ndarray) -> str:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "Desconocido"
        
        h, w, _ = vehicle_crop.shape
        
        start_x, end_x = int(w * 0.25), int(w * 0.75)
        start_y, end_y = int(h * 0.25), int(h * 0.75)
        
        roi = vehicle_crop[start_y:end_y, start_x:end_x]
        
        if roi.size == 0:
             return "Desconocido"
         
        avg_color = cv2.mean(roi)[:3] # [B, G, R]
        
        min_distance = float('inf')
        detected_color = "Otro"
        
        avg_color_np = np.array(avg_color)
        
        for name, color_bgr in self.color_map.items():
            color_np = np.array(color_bgr)
            
            distance = np.linalg.norm(avg_color_np - color_np)
            
            if distance < min_distance:
                min_distance = distance
                detected_color = name

        if min_distance > 100:
             return "Otro"
             
        return detected_color         

    