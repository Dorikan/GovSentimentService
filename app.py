from fastapi import FastAPI
from src.endpoints import api_prediction_router

app = FastAPI(title="ML Service", version="1.0")

app.include_router(api_prediction_router, prefix="/api/v1", tags=["prediction"])
