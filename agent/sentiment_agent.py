"""Агент для анализа только тональности отзывов."""

from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .prompts import CLASSIFY_OVERALL_SENTIMENT_PROMPT
from .utils import (
    format_reviews,
    llm_client,
    parse_overall_sentiments,
)


class SentimentState(TypedDict):
    reviews: List[Dict[str, Any]]
    sentiments: List[Dict[str, Any]]


async def classify_overall_sentiment(state: SentimentState) -> SentimentState:
    """Классификация общей тональности для каждого отзыва

    Args:
        state (SentimentState): Состояние агента

    Returns:
        SentimentState: Обновленное состояние с тональностями
    """
    reviews = state["reviews"]
    formatted_reviews = format_reviews(reviews)

    prompt = CLASSIFY_OVERALL_SENTIMENT_PROMPT.format(
        reviews=formatted_reviews
    )

    response = await llm_client.ainvoke(prompt)
    sentiments = parse_overall_sentiments(response)

    return {"sentiments": sentiments}


workflow = StateGraph(SentimentState)

workflow.add_node("classify_overall_sentiment", classify_overall_sentiment)

workflow.add_edge(START, "classify_overall_sentiment")
workflow.add_edge("classify_overall_sentiment", END)

sentiment_agent = workflow.compile()
