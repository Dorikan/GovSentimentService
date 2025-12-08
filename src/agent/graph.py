"""Граф агента для классификации отзывов."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .prompts import (
    CLASSIFY_CATEGORY_MULTIPLE_REVIEWS_PROMPT,
    CLASSIFY_SENTIMENT_MULTIPLE_REVIEWS_PROMPT,
    MAKE_IDEAS_MULTIPLE_REVIEWS_PROMPT,
    CLASSIFY_CATEGORY_MULTIPLE_REVIEWS_FEW_SHOT_PROMPT,
    CLASSIFY_SENTIMENT_MULTIPLE_REVIEWS_FEW_SHOT_PROMPT,
    MAKE_IDEAS_MULTIPLE_REVIEWS_FEW_SHOT_PROMPT,

)
from .state import ClassificationState
from .utils import (
    format_reviews,
    format_reviews_with_categories,
    format_reviews_with_categories_and_sentiments,
    llm_client,
    parse_review_categories,
    parse_review_sentiments,
    parse_ideas,
)


async def classify_category(state: ClassificationState) -> ClassificationState:
    """Классификация категорий для каждого отзыва

    Args:
        state (ClassificationState): Состояние агента

    Returns:
        ClassificationState: Обновленное состояние с категориями
    """
    reviews = state["reviews"]
    formatted_reviews = format_reviews(reviews)
    available_categories = state["available_categories"]
    formatted_available_categories = ", ".join(available_categories)

    if state.get("use_few_shot", False):
        prompt_template = CLASSIFY_CATEGORY_MULTIPLE_REVIEWS_FEW_SHOT_PROMPT
    else:
        prompt_template = CLASSIFY_CATEGORY_MULTIPLE_REVIEWS_PROMPT

    prompt = prompt_template.format(
        reviews=formatted_reviews,
        available_categories=formatted_available_categories,
    )

    response = await llm_client.ainvoke(prompt)
    categories = parse_review_categories(response)

    return {"categories": categories}


async def classify_sentiments(state: ClassificationState) -> ClassificationState:
    """Классификация тональности для каждой категории в каждом отзыве

    Args:
        state (ClassificationState): Состояние агента

    Returns:
        ClassificationState: Обновленное состояние с тональностями
    """
    reviews = state["reviews"]
    categories = state["categories"]
    reviews_with_categories = format_reviews_with_categories(reviews, categories)

    if state.get("use_few_shot", False):
        prompt_template = CLASSIFY_SENTIMENT_MULTIPLE_REVIEWS_FEW_SHOT_PROMPT
    else:
        prompt_template = CLASSIFY_SENTIMENT_MULTIPLE_REVIEWS_PROMPT

    prompt = prompt_template.format(reviews_with_categories=reviews_with_categories)

    response = await llm_client.ainvoke(prompt)
    sentiments = parse_review_sentiments(response)

    return {"sentiments": sentiments}


async def extract_ideas(state: ClassificationState) -> ClassificationState:
    """Извлечение идей по улучшению сервисов

    Args:
        state (ClassificationState): Состояние агента

    Returns:
        ClassificationState: Обновленное состояние с идеями
    """
    reviews = state["reviews"]
    categories = state["categories"]
    sentiments = state["sentiments"]
    
    # sentiments is list[dict] with 'id' and 'sentiments' keys
    # We need to extract just the sentiments dict for formatting
    formatted_sentiments = [s.get("sentiments", {}) for s in sentiments]
    
    reviews_with_cats_sents = format_reviews_with_categories_and_sentiments(
        reviews, categories, formatted_sentiments
    )

    if state.get("use_few_shot", False):
        prompt_template = MAKE_IDEAS_MULTIPLE_REVIEWS_FEW_SHOT_PROMPT
    else:
        prompt_template = MAKE_IDEAS_MULTIPLE_REVIEWS_PROMPT

    prompt = prompt_template.format(
        reviews_with_categories_and_sentiments=reviews_with_cats_sents
    )

    response = await llm_client.ainvoke(prompt)
    ideas = parse_ideas(response)

    return {"ideas": ideas}


workflow = StateGraph(ClassificationState)

workflow.add_node("classify_category", classify_category)
workflow.add_node("classify_sentiments", classify_sentiments)
workflow.add_node("extract_ideas", extract_ideas)

workflow.add_edge(START, "classify_category")
workflow.add_edge("classify_category", "classify_sentiments")
workflow.add_edge("classify_sentiments", "extract_ideas")
workflow.add_edge("extract_ideas", END)

classification_agent: CompiledStateGraph = workflow.compile()
