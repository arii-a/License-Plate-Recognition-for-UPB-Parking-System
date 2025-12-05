from ultralytics import YOLO
from conf.settings import VEHICLE_MODEL_PATH, PLATE_MODEL_PATH

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

    