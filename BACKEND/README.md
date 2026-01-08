# Telecom Big Data Churn Prediction System

Production-ready backend for telecom customer churn prediction using Apache Spark and Machine Learning.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Flask REST API                       │
│              (Prediction Service Layer)                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  Spark ML Pipeline                       │
│  (Feature Engineering + Model Training/Prediction)       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│               Distributed ETL Layer                      │
│        (Data Ingestion, Cleaning, Validation)            │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Parquet Data Storage                        │
│    (Raw → Cleaned → Features → Predictions)              │
└──────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Python 3.10**: Core language
- **Apache Spark 3.5.0**: Distributed data processing
- **PySpark MLlib**: Machine learning
- **Flask 3.0**: REST API framework
- **Docker**: Containerization
- **Parquet**: Columnar storage format

## Features

### 1. Distributed ETL Pipeline
- Ingest CSV/Parquet files at scale
- Schema validation and type checking
- Missing value imputation
- Duplicate removal
- Partitioned storage for query optimization

### 2. Feature Engineering
- Spark SQL aggregations (90-day rolling averages)
- Derived features:
  - Data usage per month
  - Charge per tenure
  - Complaint rate
- String indexing for categorical variables
- Feature scaling and normalization

### 3. ML Models
- **Logistic Regression**: Baseline model
- **Random Forest**: Ensemble model (100 trees)
- Binary classification evaluation
- Model persistence and versioning

### 4. Batch Prediction Engine
- Scalable prediction jobs
- Risk categorization (High/Medium/Low)
- Partitioned output by risk level
- Job statistics and monitoring

### 5. REST API
```
GET  /api/health              - Health check
GET  /api/metrics             - Aggregated metrics
GET  /api/predictions         - Paginated predictions
POST /api/run-batch           - Trigger batch job
GET  /api/batch-status        - Job status
```

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python main.py generate-data --records 100000

# Train model
python main.py train

# Run predictions
python main.py predict

# Start API server
python main.py api
```

### Docker Deployment

```bash
# Build image
make build

# Start cluster
make run

# Run full pipeline
make full-pipeline

# Check services
docker-compose ps

# View logs
make logs

# Stop services
make stop
```

## API Usage

### Get Metrics
```bash
curl http://localhost:5001/api/metrics
```

Response:
```json
{
  "total_users": 10000,
  "high_risk_count": 1523,
  "medium_risk_count": 2876,
  "low_risk_count": 5601,
  "avg_churn_probability": 0.342,
  "high_risk_percentage": 15.23,
  "top_high_risk_users": [...]
}
```

### Get Predictions (Paginated)
```bash
curl "http://localhost:5001/api/predictions?page=1&size=100&risk_category=High"
```

### Trigger Batch Prediction
```bash
curl -X POST http://localhost:5001/api/run-batch \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/opt/spark-data/cleaned",
    "model_path": "/opt/spark-data/models/churn_model"
  }'
```

## Data Schema

### Input Data
```python
- user_id: String (unique identifier)
- customer_tenure_months: Integer
- monthly_charge: Double
- total_data_usage_mb: Double
- avg_call_duration_mins: Double
- total_complaints: Integer
- contract_type: String (Month-to-month, One year, Two year)
- payment_method: String
- internet_service: String
- roaming_minutes: Double
- last_interaction_days: Integer
- churn: Integer (0 or 1)
```

### Prediction Output
```python
- user_id: String
- customer_tenure_months: Integer
- monthly_charge: Double
- total_complaints: Integer
- churn_probability: Double (0.0 to 1.0)
- prediction: Integer (0 or 1)
- risk_category: String (High, Medium, Low)
- prediction_timestamp: Timestamp
```

## Model Performance

Example metrics (Random Forest on 10K records):

```
AUC-ROC: 0.87
Accuracy: 0.82
Precision: 0.79
Recall: 0.76
F1-Score: 0.77
```

## Production Considerations

### Scalability
- Horizontal scaling via Spark cluster
- Data partitioning by date/risk category
- Configurable worker nodes
- Adaptive query execution

### Performance
- Parquet columnar format for fast queries
- Predicate pushdown optimization
- Broadcast joins for small tables
- Cached intermediate results

### Monitoring
- Spark Web UI (port 8080)
- Worker monitoring (ports 8081, 8082)
- Application logs
- API health checks

### Error Handling
- Comprehensive try-catch blocks
- Graceful degradation
- Detailed logging
- Job status tracking

## Configuration

Environment variables:

```bash
SPARK_MASTER=spark://spark-master:7077
SPARK_DRIVER_MEMORY=4g
SPARK_EXECUTOR_MEMORY=4g
API_HOST=0.0.0.0
API_PORT=5000
MODEL_TYPE=random_forest  # or logistic_regression
```

## Directory Structure

```
/opt/spark-data/
├── raw/              # Raw CSV/Parquet input
├── cleaned/          # Cleaned data
├── features/         # Engineered features
├── models/           # Trained ML models
└── predictions/      # Prediction outputs
```

## Testing

```bash
# Unit tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=main --cov-report=html

# Integration tests
make test
```

## Troubleshooting

### Issue: Out of Memory
```bash
# Increase executor memory
export SPARK_EXECUTOR_MEMORY=8g
```

### Issue: Slow predictions
```bash
# Increase parallelism
spark.conf.set("spark.sql.shuffle.partitions", "200")
```

### Issue: API timeout
```bash
# Run batch job asynchronously
POST /api/run-batch  # Returns immediately with job ID
```

## License

MIT License - Production use approved

## Contact

Senior Backend & Data Engineering Team
