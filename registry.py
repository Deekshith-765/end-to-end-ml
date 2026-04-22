import mlflow
from mlflow.tracking import MlflowClient
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_experiment("churn_prediction_experiment")

def log_model_to_registry(model, model_name, X_train, X_test, y_train, y_test, metrics):
    with mlflow.start_run():
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", len(X_train))
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"models/{model_name}",
            registered_model_name=model_name
        )
        
    print(f"Model '{model_name}' logged to MLflow registry")

def promote_model_stage(model_name, version, stage):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    print(f"Model '{model_name}' v{version} promoted to {stage}")

def get_production_model(model_name):
    client = MlflowClient()
    try:
        model_uri = f"models:/{model_name}/production"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"No production model found: {e}")
        return None

def get_latest_model(model_name, stage="production"):
    client = MlflowClient()
    try:
        latest = client.get_latest_versions(model_name, stages=[stage])
        if latest:
            return latest[0]
    except:
        pass
    return None

def compare_models(model_name):
    client = MlflowClient()
    versions = client.get_model_version_by_name(model_name, 1)
    return versions

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    X = df[["age", "salary"]]
    y = df["bought"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }
    
    log_model_to_registry(model, "churn_model", X_train, X_test, y_train, y_test, metrics)
    
    try:
        promote_model_stage("churn_model", version=1, stage="Staging")
        promote_model_stage("churn_model", version=1, stage="Production")
    except:
        print("Model promotion skipped (may already be in registry)")
