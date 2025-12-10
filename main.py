import cv2

from models.detector import VehicleDetector, PlateDetector, ColorDetector
from processors.image_processor import ImageProcessor
from processors.ocr_processor import OCRProcessor
from tracking.vehicle_tracking import VehicleTracking
from utils.visualization import Visualizer
from utils.notifier import notificar_placa_detectada


def main():
    try:
        vehicle_detector = VehicleDetector()
        plate_detector = PlateDetector()
        image_processor = ImageProcessor()
        ocr_processor = OCRProcessor()
        tracker = VehicleTracking()
        visualizer = Visualizer()
        color_detector = ColorDetector()
        
        results = vehicle_detector.track(
            source=0,
            conf=0.35,
            classes=[2, 3, 5, 6, 7]
        )
        
        print("Sistema ALPR iniciado")
        print("Presiona 'q' para salir")
        
        frame_count = 0
        cleanup_counter = 0
        
        for r in results:
            frame = r.orig_img
            if frame is None:
                continue
            
            frame_count += 1
            annotated_frame = r.plot()
            
            cleanup_counter += 1
            if cleanup_counter >= 60:
                tracker.cleanup_old_vehicles()
                cleanup_counter = 0
            
            if r.boxes is None or r.boxes.id is None:
                cv2.imshow("ALPR System", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            for i, vbox in enumerate(r.boxes):
                track_id = int(r.boxes.id[i].item())
                box = tuple(map(int, vbox.xyxy[0].tolist()))
                
                vehicle_crop = image_processor.crop_vehicle(frame, box)
                if vehicle_crop.size == 0:
                    continue
                
                color = color_detector.detect_color(vehicle_crop)
                
                info = tracker.get_vehicle_info(track_id)
                if info:
                    visualizer.draw_vehicle_info(annotated_frame, track_id, info, box)
                
                if frame_count % 3 != 0:
                    continue
                
                plate_results = plate_detector.model.predict(
                    source=vehicle_crop,
                    verbose=False,
                    conf=0.25, 
                    imgsz=640  
                )[0]
                
                for pbox in plate_results.boxes:
                    plate_box = tuple(map(int, pbox.xyxy[0].tolist()))
                    plate_crop = image_processor.crop_plate(vehicle_crop, plate_box)
                    
                    gx1 = box[0] + plate_box[0]
                    gy1 = box[1] + plate_box[1]
                    gx2 = box[0] + plate_box[2]
                    gy2 = box[1] + plate_box[3]
                    global_box = (gx1, gy1, gx2, gy2)
                    
                    plate_text = ocr_processor.read_plate(plate_crop)
                    
                    tracker.update(track_id, plate_text, color)
                        
                    
                    detected = plate_text != "NO_PLATE_FOUND"
                    visualizer.draw_plate_box(annotated_frame, global_box, detected)
                    
                    if detected:
                        visualizer.draw_plate_text(
                            annotated_frame, 
                            plate_text, 
                            (gx1, gy1 - 8)
                        )
                        if tracker.vehicles[track_id]['final_plate'] is not None and tracker.vehicles[track_id]['notified'] == False:
                            notificar_placa_detectada(
                                tracker.vehicles[track_id]['final_plate'],
                                color
                            )
            
            cv2.imshow("ALPR System", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("\nSistema ALPR finalizado")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()