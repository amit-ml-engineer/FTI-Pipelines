from feast import FeatureStore
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pyarrow.parquet as pq

store = FeatureStore(repo_path="features_repo/feature_repo")
# cutomer ids and timestamps from parquet file for which features are to be extracted
parquet_path = "features_repo/feature_repo/data/transactions.parquet"
table = pq.read_table(parquet_path, columns=["customer_id", "event_timestamp"])
entity_df = table.to_pandas()

# Get features from the feature store
features = [
    "customer_features:total_spent",
    "customer_features:num_transactions",
    "customer_features:churn_label"
]

# Retrieve feature data
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=features
).to_df()

X = training_df[["total_spent", "num_transactions"]]
y = training_df["churn_label"]

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    
    mlflow.log_metric("train_accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model")
    # Register the model in MLflow Model Registry
    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name="ChurnPredictionModel"
    )

