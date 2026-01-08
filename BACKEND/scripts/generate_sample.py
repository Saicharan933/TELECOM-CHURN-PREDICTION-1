import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

n = 10000
np.random.seed(42)

user_ids = [f"user_{i:06d}" for i in range(n)]
tenure = np.random.randint(0, 72, size=n)
monthly_charge = np.round(np.random.uniform(20, 120, size=n), 2)
total_data = np.round(np.random.uniform(100, 50000, size=n), 2)
avg_call = np.round(np.random.uniform(0.5, 20.0, size=n), 2)
complaints = np.random.poisson(0.2, size=n)
contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.6,0.25,0.15])
payment = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n)
internet = np.random.choice(["DSL","Fiber optic","No"], size=n, p=[0.4,0.45,0.15])
roaming = np.round(np.random.uniform(0, 200, size=n), 2)
last_interaction = np.random.randint(0, 365, size=n)
churn = np.random.binomial(1, 0.18, size=n)

start_date = datetime(2025, 1, 1)
record_dates = [start_date + timedelta(days=int(x)) for x in np.random.randint(0,30,size=n)]

df = pd.DataFrame({
    "user_id": user_ids,
    "customer_tenure_months": tenure,
    "monthly_charge": monthly_charge,
    "total_data_usage_mb": total_data,
    "avg_call_duration_mins": avg_call,
    "total_complaints": complaints,
    "contract_type": contract,
    "payment_method": payment,
    "internet_service": internet,
    "device_protection": np.random.choice(["Yes","No"], size=n),
    "tech_support": np.random.choice(["Yes","No"], size=n),
    "streaming_tv": np.random.choice(["Yes","No"], size=n),
    "streaming_movies": np.random.choice(["Yes","No"], size=n),
    "roaming_minutes": roaming,
    "last_interaction_days": last_interaction,
    "churn": churn,
    "record_date": pd.to_datetime(record_dates)
})

out_dir = Path("/opt/spark-data/cleaned")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "sample.parquet"

df.to_parquet(out_path, index=False)

print(f"Wrote {len(df)} rows to {out_path}")
