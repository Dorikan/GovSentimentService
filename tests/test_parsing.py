import pytest
from langchain_core.messages import AIMessage
from agent.utils import parse_review_sentiments, parse_ideas

def test_parse_sentiments_with_overall():
    json_content = """
    {
        "reviews": [
            {
                "review_id": 1,
                "sentiments": {
                    "Transport": "negative"
                },
                "overall": "negative"
            }
        ]
    }
    """
    response = AIMessage(content=json_content)
    result = parse_review_sentiments(response)
    
    assert len(result) == 1
    assert result[0]["Transport"] == "нейтрально" # "negative" not in valid set (russian), so defaults to neutral?
    # Wait, valid sentiments are russian: положительно, нейтрально, отрицательно.
    # Let's fix the test data to be Russian.

def test_parse_sentiments_with_overall_russian():
    json_content = """
    {
        "reviews": [
            {
                "review_id": 1,
                "sentiments": {
                    "Транспорт": "отрицательно"
                },
                "overall": "отрицательно"
            }
        ]
    }
    """
    response = AIMessage(content=json_content)
    result = parse_review_sentiments(response)
    
    assert len(result) == 1
    assert result[0]["Транспорт"] == "отрицательно"
    assert result[0]["overall"] == "отрицательно"

def test_parse_ideas_new_format():
    json_content = """
    {
        "ideas_by_category": [
            {
                "category": "Транспорт",
                "items": [
                    {
                        "description": "Починить автобус",
                        "source_ids": [1, 2]
                    },
                    {
                        "description": "Уволить водителя",
                        "source_ids": [3]
                    }
                ]
            }
        ]
    }
    """
    response = AIMessage(content=json_content)
    result = parse_ideas(response)
    
    assert len(result) == 1
    assert result[0]["category"] == "Транспорт"
    assert len(result[0]["ideas"]) == 2
    assert "Починить автобус" in result[0]["ideas"]
    assert "Уволить водителя" in result[0]["ideas"]

def test_parse_ideas_fallback():
    # Test backward compatibility
    json_content = """
    {
        "ideas": [
            {
                "category": "Транспорт",
                "ideas": ["Починить автобус"]
            }
        ]
    }
    """
    response = AIMessage(content=json_content)
    result = parse_ideas(response)
    
    assert len(result) == 1
    assert result[0]["category"] == "Транспорт"
    assert result[0]["ideas"] == ["Починить автобус"]
