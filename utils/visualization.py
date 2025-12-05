import cv2

class Visualizer: 
    COLOR_CONFIRMED = (0, 255, 0)    # Verde
    COLOR_READING = (255, 255, 0)     # Amarillo
    COLOR_PLATE = (0, 0, 255)         # Rojo
    COLOR_BLACK = (0, 0, 0)
    
    @staticmethod
    def draw_vehicle_info(frame, track_id: int, info: dict, box: tuple) -> None:
        vx1, vy1, vx2, vy2 = box
        
        # determinar color y texto
        if info['status'] == 'confirmed':
            color = Visualizer.COLOR_CONFIRMED
            text = f"ID:{track_id} | {info['plate']} ({info['confidence']:.0f}%)"
        else:
            color = Visualizer.COLOR_READING
            text = f"ID:{track_id} | {info['plate']}"
        
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, 
            (vx1, vy1 - text_h - 10), 
            (vx1 + text_w + 10, vy1), 
            color, -1
        )
        
        cv2.putText(
            frame, text,
            (vx1 + 5, vy1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, Visualizer.COLOR_BLACK, 2
        )
    
    @staticmethod
    def draw_plate_box(frame, box: tuple, detected: bool) -> None:
        color = Visualizer.COLOR_CONFIRMED if detected else Visualizer.COLOR_PLATE
        thickness = 3 if detected else 2
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)
    
    @staticmethod
    def draw_plate_text(frame, text: str, position: tuple) -> None:
        cv2.putText(
            frame, text, position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, Visualizer.COLOR_CONFIRMED, 2
        )