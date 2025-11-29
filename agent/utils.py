import json
import re
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from settings import settings

# Настройка логгера
logger = logging.getLogger(__name__)


class LLM:
    """
    Обертка над OpenRouter (через интерфейс OpenAI)
    """

    def __init__(self) -> None:
        """
        Инициализация клиента.
        """
        try:
            self._llm = ChatOpenAI(
                model=settings.LLM_NAME,
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.BASE_URL,
                temperature=0.0,
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка конфигурации OpenRouter: {e}") from e

    async def check_connection(self) -> bool:
        """Проверка соединения с OpenRouter."""
        try:
            await self._llm.ainvoke("Hi")
            return True
        except Exception as e:
            logger.error(f"OpenRouter connection failed: {e}")
            return False

    # Retry логика: ждет 2с, 4с, 8с... при ошибках сети или перегрузке API
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _execute_runnable(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполнение методов LangChain с автоматическим ретраем."""
        try:
            return await method(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "insufficient_quota" in error_msg:
                logger.warning(f"Rate limit exceeded (OpenRouter), retrying... Error: {e}")
            raise e

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        """Привязка инструментов (function calling)."""
        return self._llm.bind_tools(*args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """
        Структурированный вывод.
        """
        return self._llm.with_structured_output(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._execute_runnable(self._llm.ainvoke, *args, **kwargs)

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        return await self._execute_runnable(self._llm.astream, *args, **kwargs)


def format_reviews(reviews: list[dict[str, Any]]) -> str:
    output_parts = []
    separator = "-" * 100
    for i, review in enumerate(reviews, 1):
        review_id = review.get("id", i)
        text = review.get("text", "")
        output_parts.append(f"\n{i}. Отзыв (ID={review_id}):\n{text}\n{separator}\n")
    return "".join(output_parts)


def format_reviews_with_categories(reviews: list[dict[str, Any]], categories: list[list[str]]) -> str:
    output_parts = []
    separator = "-" * 100
    for i, (review, cats) in enumerate(zip(reviews, categories, strict=True), 1):
        review_id = review.get("id", i)
        text = review.get("text", "")
        cats_str = ', '.join(cats)
        block = (
            f"\n{i}. Отзыв (ID={review_id}):\n"
            f"Категории: {cats_str}\n"
            f"Текст: {text}\n"
            f"{separator}\n"
        )
        output_parts.append(block)
    return "".join(output_parts)


def format_reviews_with_categories_and_sentiments(
    reviews: list[dict[str, Any]],
    categories: list[list[str]],
    sentiments: list[dict[str, str]]
) -> str:
    output_parts = []
    separator = "-" * 100
    for i, (review, cats, sents) in enumerate(zip(reviews, categories, sentiments, strict=True), 1):
        review_id = review.get("id", i)
        text = review.get("text", "")
        cats_str = ', '.join(cats)
        sents_str = ', '.join([f"{k}: {v}" for k, v in sents.items()])
        block = (
            f"\n{i}. Отзыв (ID={review_id}):\n"
            f"Категории: {cats_str}\n"
            f"Тональность: {sents_str}\n"
            f"Текст: {text}\n"
            f"{separator}\n"
        )
        output_parts.append(block)
    return "".join(output_parts)


def _extract_json_data(response: AIMessage) -> dict:
    content = response.content.strip()
    # Пытаемся найти JSON блок
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        content = json_match.group(0)
    # Очистка markdown
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON Decode Error. Content: {content[:50]}...") from e


def parse_review_categories(response: AIMessage) -> list[list[str]]:
    try:
        data = _extract_json_data(response)
        reviews = sorted(data.get("reviews", []), key=lambda x: x.get("review_id", 0))

        result = []
        for review in reviews:
            cats = [c.strip().title() for c in review.get("categories", [])]
            result.append(cats)
        return result
    except Exception as e:
        raise ValueError(f"Category parsing error: {e}") from e


def parse_review_sentiments(response: AIMessage) -> list[dict[str, Any]]:
    valid_sentiments = {"положительно", "нейтрально", "отрицательно"}
    try:
        data = _extract_json_data(response)
        reviews = sorted(data.get("reviews", []), key=lambda x: x.get("review_id", 0))

        result = []
        for review in reviews:
            review_id = review.get("review_id")
            raw_sentiments = review.get("sentiments", {})
            normalized = {}
            for cat, sent in raw_sentiments.items():
                s_norm = sent.lower().strip()
                if s_norm not in valid_sentiments:
                    s_norm = "нейтрально"
                normalized[cat] = s_norm
            
            # Extract overall sentiment
            overall = review.get("overall", "нейтрально").lower().strip()
            if overall not in valid_sentiments:
                overall = "нейтрально"
            normalized["overall"] = overall
            
            # Return dict with id and sentiments
            result.append({"id": review_id, "sentiments": normalized})
        return result
    except Exception as e:
        raise ValueError(f"Sentiment parsing error: {e}") from e


def parse_ideas(response: AIMessage) -> list[dict[str, Any]]:
    try:
        data = _extract_json_data(response)
        
        # New format: "ideas_by_category": [{"category": "...", "items": [{"description": "...", "source_ids": []}]}]
        if "ideas_by_category" in data:
            result = []
            for category_block in data["ideas_by_category"]:
                category = category_block.get("category", "Unknown")
                items = category_block.get("items", [])
                
                # Return full items
                if items:
                    result.append({
                        "category": category,
                        "ideas": items # List of dicts {description, source_ids}
                    })
            return result
            
        # Fallback to old format if model hallucinates or uses old format
        return data.get("ideas", [])
    except Exception as e:
        raise ValueError(f"Ideas parsing error: {e}") from e


def parse_overall_sentiments(response: AIMessage) -> list[dict[str, Any]]:
    valid_sentiments = {"положительно", "нейтрально", "отрицательно"}
    try:
        data = _extract_json_data(response)
        reviews = sorted(data.get("reviews", []), key=lambda x: x.get("review_id", 0))

        result = []
        for review in reviews:
            review_id = review.get("review_id")
            overall = review.get("overall", "нейтрально").lower().strip()
            if overall not in valid_sentiments:
                overall = "нейтрально"
            
            # Return dict with id and overall sentiment
            result.append({"id": review_id, "overall": overall})
        return result
    except Exception as e:
        raise ValueError(f"Overall sentiment parsing error: {e}") from e



llm_client = LLM()
