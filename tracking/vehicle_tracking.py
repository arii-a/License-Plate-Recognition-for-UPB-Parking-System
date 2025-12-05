import time
from collections import Counter

class VehicleTracking: 
    
    def __init__(self, max_readings=10, timeout=5.0):
        self.vehicles = {}
        self.max_readings = max_readings
        self.timeout = timeout
    
    def update(self, track_id: int, plate_text: str) -> None:
        current_time = time.time()
        
        if track_id not in self.vehicles:
            self.vehicles[track_id] = {
                'plates': [],
                'last_seen': current_time,
                'final_plate': None,
                'confidence': 0
            }
        
        vehicle = self.vehicles[track_id]
        vehicle['last_seen'] = current_time
        
        if vehicle['final_plate'] is None and plate_text != "NO_PLATE_FOUND":
            vehicle['plates'].append(plate_text)
            
            if len(vehicle['plates']) >= self.max_readings:
                self._finalize_plate(track_id)
    
    def _finalize_plate(self, track_id: int) -> None:
        vehicle = self.vehicles[track_id]
        
        if not vehicle['plates']:
            return
        
        counter = Counter(vehicle['plates'])
        most_common = counter.most_common(1)[0]
        plate, count = most_common
        
        vehicle['final_plate'] = plate
        vehicle['confidence'] = (count / len(vehicle['plates'])) * 100
        
        print(f"\n{'='*50}")
        print(f"Vehículo ID: {track_id}")
        print(f"Placa: {plate}")
        print(f"Confianza: {vehicle['confidence']:.1f}% ({count}/{len(vehicle['plates'])})")
        print(f"{'='*50}\n")
    
    def get_vehicle_info(self, track_id: int) -> dict | None:
        if track_id not in self.vehicles:
            return None
        
        vehicle = self.vehicles[track_id]
        
        if vehicle['final_plate']:
            return {
                'plate': vehicle['final_plate'],
                'confidence': vehicle['confidence'],
                'status': 'confirmed'
            }
        else:
            readings = len(vehicle['plates'])
            return {
                'plate': f"Leyendo... {readings}/{self.max_readings}",
                'confidence': 0,
                'status': 'reading'
            }
    
    def cleanup_old_vehicles(self) -> None:
        current_time = time.time()
        to_remove = [
            track_id for track_id, vehicle in self.vehicles.items()
            if current_time - vehicle['last_seen'] > self.timeout
        ]
        
        for track_id in to_remove:
            del self.vehicles[track_id]
            print(f"Vehículo {track_id} removido")