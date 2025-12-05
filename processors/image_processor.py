import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def preprocess_plate(plate_crop: np.ndarray) -> np.ndarray:
        if plate_crop.size == 0:
            return plate_crop
        
        h, w = plate_crop.shape[:2]
        if h < 60:
            scale = 60 / h
            plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale, 
                                    interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # binarización adaptativa
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return binary
    
    @staticmethod
    def crop_vehicle(frame: np.ndarray, box: tuple) -> np.ndarray:
        vx1, vy1, vx2, vy2 = box
        vx1 = max(0, vx1)
        vy1 = max(0, vy1)
        vx2 = min(frame.shape[1], vx2)
        vy2 = min(frame.shape[0], vy2)
        
        return frame[vy1:vy2, vx1:vx2]
    
    @staticmethod
    def crop_plate(vehicle_crop: np.ndarray, box: tuple) -> np.ndarray:
        px1, py1, px2, py2 = box
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(vehicle_crop.shape[1], px2)
        py2 = min(vehicle_crop.shape[0], py2)
        
        return vehicle_crop[py1:py2, px1:px2]
        