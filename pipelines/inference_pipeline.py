from utils.data_cleaning import load_data, clean_data
import mlflow
import joblib

def run_inference_pipeline():
    mlflow.start_run()
    # Load and clean data
    df = load_data('data/test.csv')
    df = clean_data(df)
    
    # Load the trained model from MLflow
    model_uri = "runs:/<RUN_ID>/random_forest_model"  # Replace <RUN_ID> with the actual run ID
    model = mlflow.sklearn.load_model(model_uri)
    
    # Make predictions
    predictions = model.predict(df)
    print(predictions)
    mlflow.end_run()
