from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from datetime import datetime

class PlacaDetectadaEvent(BaseModel):
    placa: str
    confianza: float
    timestamp: datetime
    

router = APIRouter()

@router.post("/event/placa-detectada")
async def recibir_placa_detectada(event: PlacaDetectadaEvent):
    try:
        
        print(f"EVENTO RECIBIDO: Placa {event.placa}")
        
        return {"message": "Evento procesado correctamente"}

    except Exception as e:
        print(f"Error procesando el evento: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fallo al registrar el evento de placa en la DB."
        )