# feature_repo/feature_store.py
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Int64
import pandas as pd
# Define entity
customer = Entity(name="customer_id", join_keys=["customer_id"])

# Define data source
transactions_source = FileSource(
    path="data/transactions.parquet",
    timestamp_field="event_timestamp",
)

# Load data and calculate calculate churn_label feature
df = pd.read_parquet("data/transactions.parquet")
df["churn_label"] = ((df["total_spent"] < 2000) & (df["num_transactions"] < 10)).astype(int)
df.to_parquet("data/transactions.parquet", index=False)  # Overwrite with churn_label

# Define feature view
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=None,
    schema=[
        Field(name="total_spent", dtype=Float32),
        Field(name="num_transactions", dtype=Int64),
        Field(name="churn_label", dtype=Int64), # Add churn_label as a feature
    ],
    source=transactions_source,
)
