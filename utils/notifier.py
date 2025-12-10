import requests
from datetime import datetime

BACKEND_URL = "http://localhost:8000/api/v1/event/placa-detectada"

def notificar_placa_detectada(placa: str, color: str):
    payload = {
        "placa": placa,
        "color": color,
        "modelo": "N/A"
    }
    
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=5)
        response.raise_for_status()
        print(f"Notificación de placa {placa} enviada con éxito.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERROR al notificar: {e}")
        return False