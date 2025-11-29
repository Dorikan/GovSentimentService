from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.prediction_service import prediction_service

router = APIRouter()


class ReviewItem(BaseModel):
    id: int
    text: str


class CategorySentiment(BaseModel):
    name: str
    sentiment: int


class ReviewResponse(BaseModel):
    id: int
    categories: List[CategorySentiment]
    overall: int


class IdeaResponse(BaseModel):
    category: str
    description: str
    source_ids: List[int]


class PredictionRequest(BaseModel):
    reviews: List[ReviewItem]
    use_few_shot: bool = False


class PredictionResponse(BaseModel):
    reviews: List[ReviewResponse]
    ideas: List[IdeaResponse]


def map_sentiment_to_int(sentiment: str) -> int:
    s = sentiment.lower().strip()
    if s == "отрицательно":
        return 2
    elif s == "положительно":
        return 1
    else:
        return 0


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

        # Transform reviews
        transformed_reviews = []
        for review_id, sentiments_data in reviews_map.items():
            # Extract overall sentiment
            overall_str = sentiments_data.pop("overall", "нейтрально")
            overall_val = map_sentiment_to_int(overall_str)

            categories_list = []
            for cat_name, sent_str in sentiments_data.items():
                categories_list.append(
                    CategorySentiment(
                        name=cat_name,
                        sentiment=map_sentiment_to_int(sent_str)
                    )
                )

            transformed_reviews.append(
                ReviewResponse(
                    id=review_id,
                    categories=categories_list,
                    overall=overall_val
                )
            )

        # Transform ideas
        transformed_ideas = []
        for category, ideas_list in ideas_map.items():
            for idea in ideas_list:
                transformed_ideas.append(
                    IdeaResponse(
                        category=category,
                        description=idea.get("description", ""),
                        source_ids=idea.get("source_ids", [])
                    )
                )

        return PredictionResponse(reviews=transformed_reviews, ideas=transformed_ideas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
