from typing import TypedDict, Any


class ClassificationState(TypedDict):
    """Состояние агента для классификации отзывов"""
    available_categories: list[str]
    reviews: list[dict[str, Any]]
    categories: list[list[str]]
    sentiments: list[dict[str, str]]
    ideas: list[dict[str, Any]]
    use_few_shot: bool