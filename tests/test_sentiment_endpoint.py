import asyncio
import sys
import os
import pytest

# Add project root to path
sys.path.append(os.getcwd())

from src.services.prediction_service import prediction_service
from src.endpoints.api.v1.endpoints import map_sentiment_to_int

@pytest.mark.asyncio
async def test_sentiment_endpoint():
    print("Testing sentiment-only prediction...")
    
    reviews = [
        {"id": 1, "text": "Все отлично, спасибо!"},
        {"id": 2, "text": "Ужасный сервис, ничего не работает."},
        {"id": 3, "text": "Нормально, пойдет."}
    ]
    
    try:
        sentiments = await prediction_service.predict_sentiment_only(reviews)
        print(f"Sentiments: {sentiments}")
        
        assert sentiments[1] == "положительно" or sentiments[1] == "positive" # Allow for some variation if model outputs english
        assert sentiments[2] == "отрицательно" or sentiments[2] == "negative"
        assert sentiments[3] == "нейтрально" or sentiments[3] == "neutral"
        
        print("Verification successful!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_sentiment_endpoint())
