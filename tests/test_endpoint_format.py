import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from src.endpoints.api.v1.endpoints import router

# Create a test app since the main app might not include the router directly in the same way or for isolation
from fastapi import FastAPI
test_app = FastAPI()
test_app.include_router(router)

client = TestClient(test_app)

@pytest.mark.asyncio
async def test_predict_endpoint_format():
    # Mock data returned by prediction_service.predict
    mock_reviews_map = {
        123: {
            "Здравоохранение": "отрицательно",
            "overall": "отрицательно"
        },
        124: {
            "Мфц/Госуслуги": "положительно",
            "overall": "положительно"
        },
        125: {
            "Мфц/Госуслуги": "положительно",
            "overall": "положительно"
        }
    }
    
    mock_ideas_map = {
        "Здравоохранение": [
            {
                "description": "Clean up",
                "source_ids": [123]
            }
        ],
        "Мфц/Госуслуги": [
            {
                "description": "Photo service",
                "source_ids": [125]
            }
        ]
    }

    with patch("src.services.prediction_service.prediction_service.predict", new_callable=AsyncMock) as mock_predict:
        mock_predict.return_value = (mock_reviews_map, mock_ideas_map)
        
        response = client.post("/predict", json={
            "reviews": [
                {"id": 123, "text": "Bad hospital"},
                {"id": 124, "text": "Good MFC"},
                {"id": 125, "text": "Good MFC photo"}
            ]
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify structure
        assert "reviews" in data
        assert "ideas" in data
        assert isinstance(data["reviews"], list)
        assert isinstance(data["ideas"], list)
        
        # Verify reviews transformation
        reviews = sorted(data["reviews"], key=lambda x: x["id"])
        assert len(reviews) == 3
        
        r1 = reviews[0]
        assert r1["id"] == 123
        assert r1["overall"] == 2 # отрицательно
        assert len(r1["categories"]) == 1
        assert r1["categories"][0]["name"] == "Здравоохранение"
        assert r1["categories"][0]["sentiment"] == 2
        
        r2 = reviews[1]
        assert r2["id"] == 124
        assert r2["overall"] == 1 # положительно
        assert len(r2["categories"]) == 1
        assert r2["categories"][0]["name"] == "Мфц/Госуслуги"
        assert r2["categories"][0]["sentiment"] == 1
        
        # Verify ideas transformation
        ideas = sorted(data["ideas"], key=lambda x: x["category"])
        assert len(ideas) == 2
        
        i1 = ideas[0]
        assert i1["category"] == "Здравоохранение"
        assert i1["description"] == "Clean up"
        assert i1["source_ids"] == [123]
        
        i2 = ideas[1]
        assert i2["category"] == "Мфц/Госуслуги"
        assert i2["description"] == "Photo service"
        assert i2["source_ids"] == [125]

if __name__ == "__main__":
    # Manually run the test function if executed directly (helper for debugging)
    import asyncio
    try:
        asyncio.run(test_predict_endpoint_format())
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
