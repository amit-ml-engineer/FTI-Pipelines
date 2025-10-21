from fastapi import FastAPI
import mlflow.pyfunc
from feast import FeatureStore

app = FastAPI()
store = FeatureStore(repo_path="features_repo/feature_repo")

model_name = "ChurnPredictionModel"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

@app.post("/predict")
def predict(customer_id: int):
    # Get latest features

    feature_vector = store.get_online_features(
        features=["customer_features:total_spent",
                  "customer_features:num_transactions"],
        entity_rows=[{"customer_id": customer_id}]
    ).to_df()

    # Predict
    preds = model.predict(feature_vector[["total_spent", "num_transactions"]])
    return {"customer_id": customer_id, "prediction": int(preds[0])}
# To run the API, use the command:
# uvicorn inference_pipeline:app --host 0.0.0.0 --port 8000
# curl -X POST "http://localhost:8000/predict?customer_id=your_customer_id"