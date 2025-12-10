from fastapi import FastAPI
from .routers import alpr_router

app = FastAPI(
    title="ALPR Detector Service",
    description="API para la detección y reconocimiento de matrículas usando YOLOv11x.",
    version="1.0.0",
)

#app.include_router(alpr_router.router, prefix="/api/v1", tags=["detection"])

@app.get("/")
def read_root():
    return {"message": "Bienvenido al servicio ALPR Detector"}