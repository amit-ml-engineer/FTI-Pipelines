import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic data
np.random.seed(42)
num_customers = 100
num_records = 500

data = {
    "customer_id": np.random.randint(1, num_customers + 1, num_records),
    "total_spent": np.random.uniform(100, 10000, num_records).round(2),
    "num_transactions": np.random.randint(1, 50, num_records),
    "event_timestamp": [
        datetime.now() - timedelta(days=np.random.randint(0, 30))
        for _ in range(num_records)
    ],
}

df = pd.DataFrame(data)
df.to_parquet("data_raw/transactions.parquet")
