import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


async def main():
    file_path = "reviews.json"
    use_few_shot = False

    args = sys.argv[1:]
    if "--few-shot" in args:
        use_few_shot = True
        args.remove("--few-shot")
        print("Running in FEW-SHOT mode.")
    else:
        print("Running in ZERO-SHOT mode.")
        
    if len(args) > 0:
        file_path = args[0]

    path = Path(file_path)
    if not path.exists():
        print(f"File {file_path} not found. Please create a json file with a list of reviews.")
        print("Example: [\"Review 1\", \"Review 2\"]")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            reviews = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}")
        return

    if not isinstance(reviews, list):
        print("JSON must contain a list of strings (reviews).")
        return

    from services.prediction_service import prediction_service
    
    print(f"Loaded {len(reviews)} reviews. Starting classification...")
    
    reviews_map, ideas_map = await prediction_service.predict(reviews, use_few_shot=use_few_shot)

    print("\nClassification Results:")
    print("-" * 50)
    
    for i, review in enumerate(reviews):
        print(f"\nReview {i+1}: {review}")
        
        sentiments = reviews_map.get(i, {})

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

if __name__ == "__main__":
    asyncio.run(main())
