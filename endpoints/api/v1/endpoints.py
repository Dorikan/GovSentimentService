from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.prediction_service import prediction_service

router = APIRouter()


class ReviewItem(BaseModel):
    id: int
    text: str


class IdeaItem(BaseModel):
    description: str
    source_ids: List[int]


class PredictionRequest(BaseModel):
    reviews: List[ReviewItem]
    use_few_shot: bool = False


class PredictionResponse(BaseModel):
    reviews: Dict[int, Dict[str, str]]
    ideas: Dict[str, List[IdeaItem]]


@router.post("/predict", response_model=PredictionResponse)
async def predict_reviews(request: PredictionRequest):
    """
    Эндпоинт для классификации отзывов.
    Принимает список отзывов с ID, возвращает классификацию, тональность и идеи.
    """
    if not request.reviews:
        raise HTTPException(status_code=400, detail="List of reviews cannot be empty")

    # Convert Pydantic models to list of dicts for the service
    reviews_dicts = [r.model_dump() for r in request.reviews]

    try:
        reviews_map, ideas_map = await prediction_service.predict(
            reviews=reviews_dicts,
            use_few_shot=request.use_few_shot
        )
        return PredictionResponse(reviews=reviews_map, ideas=ideas_map)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
