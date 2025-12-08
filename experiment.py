import argparse
import time
import logging
import pandas as pd
import requests
import mlflow
from src.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run experiment for Sentiment Service")
    parser.add_argument("--service-url", required=True, help="URL of the predict endpoint")
    parser.add_argument("--mlflow-url", required=True, help="URL of the MLflow server")
    parser.add_argument("--model-name", required=True, help="Name of the model being tested")
    parser.add_argument("--csv-path", required=True, help="Path to the CSV file with reviews")
    parser.add_argument("--few-shot", action="store_true", help="Enable few-shot mode")

    args = parser.parse_args()

    # Setup MLflow
    mlflow.set_tracking_uri(args.mlflow_url)
    mlflow.set_experiment("Sentiment Service Experiment")

    # Load Data
    try:
        df = pd.read_csv(args.csv_path)

    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return

    if "text" not in df.columns or "label" not in df.columns:
        logger.error("CSV must contain 'text' and 'label' columns.")
        return

    y_true = []
    y_pred = []
    processing_times = []

    logger.info(f"Starting experiment with model: {args.model_name}")
    # Limit to 2000 reviews
    if len(df) > 100:
        print(f"Limiting dataset from {len(df)} to 100 reviews.")
        df = df.iloc[:100]

    logger.info(f"Processing {len(df)} reviews in batches of {settings.BATCH_SIZE}...")

    with mlflow.start_run():
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("few_shot", args.few_shot)
        mlflow.log_param("batch_size", 5)

        # Process in batches
        batch_size = settings.BATCH_SIZE
        errors = []

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            
            reviews_payload = []
            batch_true_labels = {} # id -> label
            batch_reviews_text = {} # id -> text
            
            for index, row in batch_df.iterrows():
                reviews_payload.append({"id": index, "text": row["text"]})
                batch_true_labels[index] = row["label"]
                batch_reviews_text[index] = row["text"]

            payload = {
                "reviews": reviews_payload,
                "use_few_shot": args.few_shot
            }

            start_time = time.time()
            try:
                response = requests.post(args.service_url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Calculate processing time per batch
                duration = time.time() - start_time
                processing_times.append(duration)

                # Parse prediction
                # New format: {"reviews": [{"id": ..., "overall": ..., "categories": ...}]}
                reviews_resp = data.get("reviews", [])
                
                # Map responses back to true labels
                for review_item in reviews_resp:
                    r_id = review_item.get("id")
                    predicted_sentiment = review_item.get("overall", 0) # Default to 1 (Neutral)
                    
                    if r_id in batch_true_labels:
                        true_label = int(batch_true_labels[r_id])
                        y_true.append(true_label)
                        y_pred.append(predicted_sentiment)
                        
                        if predicted_sentiment != true_label:
                            review_text = batch_reviews_text.get(r_id, "N/A")
                            print(f"\n[MISMATCH] Review ID: {r_id}")
                            print(f"Text: {review_text}")
                            print(f"Predicted: {predicted_sentiment} | True: {true_label}")
                            print("-" * 50)
                            
                            errors.append({
                                "id": r_id,
                                "text": review_text,
                                "predicted": predicted_sentiment,
                                "true": true_label
                            })

                    else:
                        print(f"Warning: Received ID {r_id} not in sent batch")

            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue

            macro_f1 = f1_score(y_true, y_pred, average='macro')

            print(f'processed {i}\nmacro F1: {macro_f1}')

        # Calculate Metrics
        if not y_true:
            print("No successful predictions.")
            return

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

        print(f"Macro F1: {macro_f1}")
        print(f"Avg Processing Time: {avg_time:.4f}s")

        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("avg_processing_time", avg_time)
        
        if errors:
            errors_df = pd.DataFrame(errors)
            errors_csv_path = "errors.csv"
            errors_df.to_csv(errors_csv_path, index=False)
            mlflow.log_artifact(errors_csv_path)
            print(f"Logged {len(errors)} errors to MLflow artifact: {errors_csv_path}")

if __name__ == "__main__":
    main()
