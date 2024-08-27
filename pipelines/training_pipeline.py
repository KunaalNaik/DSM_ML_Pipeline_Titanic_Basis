from utils.data_cleaning import load_data, clean_data, split_data
from utils.model_training import train_model, evaluate_model, log_model
import mlflow

def run_training_pipeline():
    mlflow.start_run()
    # Load and clean data
    df = load_data('data/train.csv')
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train and evaluate model
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Log model and accuracy
    log_model(model, accuracy)
    mlflow.end_run()
