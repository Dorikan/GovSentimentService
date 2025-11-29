import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.prediction_service import PredictionService
from settings import settings

@pytest.fixture
def prediction_service():
    return PredictionService()

@pytest.mark.asyncio
async def test_predict_realistic_scenario(prediction_service):
    # Realistic reviews with IDs
    reviews = [
        {"id": 101, "text": "Автобус 55 постоянно опаздывает, приходится ждать по 20 минут."},
        {"id": 102, "text": "В поликлинике №10 отличные врачи, но в регистратуре хамят."},
        {"id": 103, "text": "Парк Горького стал очень чистым и красивым, спасибо!"},
        {"id": 104, "text": "Невозможно записаться к стоматологу через госуслуги, постоянно ошибка."},
        {"id": 105, "text": "Во дворе дома 5 по улице Ленина не вывозят мусор уже неделю."}
    ]
    
    # Set batch size to 2 to test batching (5 reviews -> 3 batches: 2, 2, 1)
    settings.BATCH_SIZE = 2

    # Mock agent response logic
    async def mock_ainvoke(state):
        batch_reviews = state["reviews"]
        
        # Prepare mock responses based on the content of reviews in the batch
        sentiments = []
        ideas = []
        
        for review in batch_reviews:
            r_id = review["id"]
            text = review["text"]
            
            if "Автобус" in text:
                sentiments.append({"id": r_id, "sentiments": {"Транспорт": "отрицательно", "overall": "отрицательно"}})
                ideas.append({
                    "category": "Транспорт",
                    "ideas": [{"description": "Увеличить частоту движения автобуса 55", "source_ids": [r_id]}]
                })
            elif "поликлинике" in text:
                sentiments.append({
                    "id": r_id, 
                    "sentiments": {"Здравоохранение": "положительно", "МФЦ/Госуслуги": "отрицательно", "overall": "нейтрально"}
                })
                ideas.append({
                    "category": "МФЦ/Госуслуги",
                    "ideas": [{"description": "Провести обучение персонала регистратуры", "source_ids": [r_id]}]
                })
            elif "Парк" in text:
                sentiments.append({"id": r_id, "sentiments": {"Благоустройство": "положительно", "overall": "положительно"}})
                # Positive review, usually no ideas
            elif "стоматологу" in text:
                sentiments.append({"id": r_id, "sentiments": {"МФЦ/Госуслуги": "отрицательно", "Здравоохранение": "нейтрально", "overall": "отрицательно"}})
                ideas.append({
                    "category": "МФЦ/Госуслуги",
                    "ideas": [{"description": "Исправить технические ошибки при записи к врачу", "source_ids": [r_id]}]
                })
            elif "мусор" in text:
                sentiments.append({"id": r_id, "sentiments": {"ЖКХ": "отрицательно", "overall": "отрицательно"}})
                ideas.append({
                    "category": "ЖКХ",
                    "ideas": [{"description": "Обеспечить регулярный вывоз мусора по ул. Ленина, 5", "source_ids": [r_id]}]
                })
                
        return {
            "sentiments": sentiments,
            "ideas": ideas
        }

    # Patch the agent
    with patch("services.prediction_service.classification_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        
        # Run prediction
        reviews_map, ideas_map = await prediction_service.predict(reviews)
        
        # Verify batch calls
        # 5 reviews, batch size 2 -> 3 calls (2, 2, 1)
        assert mock_agent.ainvoke.call_count == 3
        
        # Verify results aggregation
        assert len(reviews_map) == 5
        
        # Check specific reviews by ID
        # Review 101: Bus
        assert reviews_map[101]["Транспорт"] == "отрицательно"
        
        # Review 102: Clinic (mixed)
        assert reviews_map[102]["Здравоохранение"] == "положительно"
        
        # Verify ideas aggregation
        assert "Транспорт" in ideas_map
        assert ideas_map["Транспорт"][0]["description"] == "Увеличить частоту движения автобуса 55"
        assert ideas_map["Транспорт"][0]["source_ids"] == [101]
        
        assert "ЖКХ" in ideas_map
        assert ideas_map["ЖКХ"][0]["description"] == "Обеспечить регулярный вывоз мусора по ул. Ленина, 5"
        assert ideas_map["ЖКХ"][0]["source_ids"] == [105]

@pytest.mark.asyncio
async def test_predict_empty_reviews(prediction_service):
    with patch("services.prediction_service.classification_agent") as mock_agent:
        reviews_map, ideas_map = await prediction_service.predict([])
        assert reviews_map == {}
        assert ideas_map == {}
        mock_agent.ainvoke.assert_not_called()
