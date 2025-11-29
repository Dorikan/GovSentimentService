import asyncio
from typing import Any, Dict, List, Tuple

from agent import agent as classification_agent
from agent.sentiment_agent import sentiment_agent
from settings import settings


class PredictionService:
    """Сервис для классификации отзывов с использованием агента."""

    def __init__(self):
        self.available_categories = [
            "Благоустройство",
            "ЖКХ",
            "Транспорт",
            "Здравоохранение",
            "Образование",
            "Социальная поддержка",
            "Безопасность",
            "Связь и интернет",
            "МФЦ/Госуслуги",
            "Прочее"
        ]

    async def predict(
        self, reviews: List[Dict[str, Any]], use_few_shot: bool = False
    ) -> Tuple[Dict[int, Dict[str, str]], Dict[str, List[Dict[str, Any]]]]:
        """
        Обрабатывает список отзывов, разбивая их на батчи и запуская агент.

        Args:
            reviews: Список словарей отзывов [{'id': 1, 'text': '...'}].
            use_few_shot: Использовать ли few-shot промпты.

        Returns:
            Tuple из двух словарей:
            1. reviews_with_sentiments_and_categories: {review_id: {category: sentiment, overall: sentiment}}
            2. ideas: {category_name: [{description: str, source_ids: list[int]}]}
        """
        all_sentiments_and_categories: Dict[int, Dict[str, str]] = {}
        all_ideas: Dict[str, List[Dict[str, Any]]] = {}

        batch_size = settings.BATCH_SIZE
        
        # Обработка по батчам
        for i in range(0, len(reviews), batch_size):
            batch_reviews = reviews[i : i + batch_size]
            
            initial_state = {
                "reviews": batch_reviews,
                "available_categories": self.available_categories,
                "categories": [],
                "sentiments": [],
                "ideas": [],
                "use_few_shot": use_few_shot
            }
            
            # Запуск агента (синхронно для каждого батча)
            final_state = await classification_agent.ainvoke(initial_state)
            
            # 1. Сбор результатов классификации и тональности
            batch_sentiments = final_state.get("sentiments", [])
            
            for item in batch_sentiments:
                r_id = item.get("id")
                sents = item.get("sentiments")
                if r_id is not None:
                     all_sentiments_and_categories[r_id] = sents

            # 2. Сбор идей
            batch_ideas = final_state.get("ideas", [])
            
            for idea_block in batch_ideas:
                category = idea_block.get("category")
                ideas_list = idea_block.get("ideas", [])
                
                if category and ideas_list:
                    if category not in all_ideas:
                        all_ideas[category] = []
                    all_ideas[category].extend(ideas_list)

        return all_sentiments_and_categories, all_ideas

    async def predict_sentiment_only(
        self, reviews: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        """
        Обрабатывает список отзывов для получения только общей тональности.

        Args:
            reviews: Список словарей отзывов [{'id': 1, 'text': '...'}].

        Returns:
            Dict[int, str]: {review_id: overall_sentiment}
        """
        all_sentiments: Dict[int, str] = {}
        batch_size = settings.BATCH_SIZE

        for i in range(0, len(reviews), batch_size):
            batch_reviews = reviews[i : i + batch_size]

            initial_state = {
                "reviews": batch_reviews,
                "sentiments": []
            }

            final_state = await sentiment_agent.ainvoke(initial_state)
            
            batch_sentiments = final_state.get("sentiments", [])
            for item in batch_sentiments:
                r_id = item.get("id")
                overall = item.get("overall")
                if r_id is not None:
                    all_sentiments[r_id] = overall

        return all_sentiments

prediction_service = PredictionService()
