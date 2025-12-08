import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis CLI")
    parser.add_argument("file_path", nargs="?", default="reviews.json", help="Path to the JSON file with reviews")
    parser.add_argument("--few-shot", action="store_true", help="Enable few-shot mode")
    args_cli = parser.parse_args()

    file_path = args_cli.file_path
    use_few_shot = args_cli.few_shot

    if use_few_shot:
        logger.info("Running in FEW-SHOT mode.")
    else:
        logger.info("Running in ZERO-SHOT mode.")

    path = Path(file_path)
    if not path.exists():
        logger.error(f"File {file_path} not found. Please create a json file with a list of reviews.")
        logger.info("Example: [\"Review 1\", \"Review 2\"] or [{\"id\": 1, \"text\": \"Review 1\"}]")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_reviews = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        return

    if not isinstance(raw_reviews, list):
        logger.error("JSON must contain a list of reviews (strings or objects).")
        return

    # Adapt input to List[Dict]
    reviews = []
    for i, item in enumerate(raw_reviews):
        if isinstance(item, str):
            reviews.append({"id": i + 1, "text": item})
        elif isinstance(item, dict):
             # Ensure dict has 'id' and 'text'
             if "text" not in item:
                 logger.warning(f"Skipping item {i}: missing 'text' field")
                 continue
             if "id" not in item:
                 item["id"] = i + 1
             reviews.append(item)
        else:
            logger.warning(f"Skipping item {i}: unknown format")

    if not reviews:
        logger.error("No valid reviews found.")
        return

    from src.services.prediction_service import prediction_service
    
    logger.info(f"Loaded {len(reviews)} reviews. Starting classification...")
    
    try:
        reviews_map, ideas_map = await prediction_service.predict(reviews, use_few_shot=use_few_shot)

        print("\nClassification Results:")
        print("-" * 50)
        
        for r_id, sentiments in reviews_map.items():
            # Find original text for display (optional, can be expensive if large list)
            # For CLI we can just print ID
            print(f"\nReview ID: {r_id}")
            
            categories = list(sentiments.keys())
            
            print(f"Categories: {', '.join(categories)}")
            print("Sentiments:")
            for cat, sentiment in sentiments.items():
                print(f"  - {cat}: {sentiment}")
            print("-" * 50)

        if ideas_map:
            print("\nImprovement Ideas:")
            print("=" * 50)
            for category, ideas_list in ideas_map.items():
                print(f"\nCategory: {category}")
                for idea in ideas_list:
                    print(f"  - {idea}")
            print("=" * 50)
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
