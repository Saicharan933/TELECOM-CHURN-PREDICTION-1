"""
Telecom Big Data Churn Prediction System
Production-Ready Backend with Apache Spark & Flask

Author: Senior Backend & Data Engineer
Tech Stack: PySpark, Spark MLlib, Flask, Docker, Parquet
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# Spark Imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, TimestampType, BooleanType
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Flask Imports
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the churn prediction system"""
    
    # Spark Configuration
    SPARK_APP_NAME = "TelecomChurnPrediction"
    SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
    SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
    SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
    
    # Data Paths
    BASE_DIR = Path("/opt/spark-data")
    RAW_DATA_PATH = BASE_DIR / "raw"
    CLEANED_DATA_PATH = BASE_DIR / "cleaned"
    FEATURES_PATH = BASE_DIR / "features"
    MODELS_PATH = BASE_DIR / "models"
    PREDICTIONS_PATH = BASE_DIR / "predictions"
    
    # Model Configuration
    MODEL_TYPE = os.getenv("MODEL_TYPE", "random_forest")  # or "logistic_regression"
    CHURN_THRESHOLD = 0.5
    HIGH_RISK_THRESHOLD = 0.7
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "5000"))
    API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"
    
    # Feature Engineering
    LOOKBACK_DAYS = 90
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for path in [cls.RAW_DATA_PATH, cls.CLEANED_DATA_PATH, 
                     cls.FEATURES_PATH, cls.MODELS_PATH, cls.PREDICTIONS_PATH]:
            path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SPARK SESSION MANAGER
# ============================================================================

class SparkSessionManager:
    """Singleton Spark Session Manager"""
    
    _instance = None
    _spark = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SparkSessionManager, cls).__new__(cls)
        return cls._instance
    
    def get_spark(self) -> SparkSession:
        """Get or create Spark session"""
        if self._spark is None:
            logger.info("Creating new Spark session")
            self._spark = SparkSession.builder \
                .appName(Config.SPARK_APP_NAME) \
                .master(Config.SPARK_MASTER) \
                .config("spark.driver.memory", Config.SPARK_DRIVER_MEMORY) \
                .config("spark.executor.memory", Config.SPARK_EXECUTOR_MEMORY) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .getOrCreate()
            
            self._spark.sparkContext.setLogLevel("WARN")
            logger.info(f"Spark session created: {self._spark.version}")
        
        return self._spark
    
    def stop_spark(self):
        """Stop Spark session"""
        if self._spark is not None:
            logger.info("Stopping Spark session")
            self._spark.stop()
            self._spark = None


# ============================================================================
# DATA SCHEMAS
# ============================================================================

class TelecomSchemas:
    """Define schemas for telecom data"""
    
    @staticmethod
    def get_raw_schema() -> StructType:
        """Schema for raw telecom data"""
        return StructType([
            StructField("user_id", StringType(), False),
            StructField("customer_tenure_months", IntegerType(), True),
            StructField("monthly_charge", DoubleType(), True),
            StructField("total_data_usage_mb", DoubleType(), True),
            StructField("avg_call_duration_mins", DoubleType(), True),
            StructField("total_complaints", IntegerType(), True),
            StructField("contract_type", StringType(), True),  # Month-to-month, One year, Two year
            StructField("payment_method", StringType(), True),
            StructField("internet_service", StringType(), True),
            StructField("device_protection", StringType(), True),
            StructField("tech_support", StringType(), True),
            StructField("streaming_tv", StringType(), True),
            StructField("streaming_movies", StringType(), True),
            StructField("roaming_minutes", DoubleType(), True),
            StructField("last_interaction_days", IntegerType(), True),
            StructField("churn", IntegerType(), True),  # 0 or 1
            StructField("record_date", TimestampType(), True)
        ])
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of feature column names"""
        return [
            "customer_tenure_months",
            "monthly_charge",
            "total_data_usage_mb",
            "avg_call_duration_mins",
            "total_complaints",
            "roaming_minutes",
            "last_interaction_days",
            "data_usage_per_month",
            "charge_per_tenure",
            "complaint_rate",
            "contract_type_indexed",
            "payment_method_indexed",
            "internet_service_indexed"
        ]


# ============================================================================
# ETL LAYER
# ============================================================================

class SparkETL:
    """ETL operations using PySpark"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def ingest_data(self, 
                    input_path: str, 
                    file_format: str = "csv") -> DataFrame:
        """
        Ingest data from CSV or Parquet
        
        Args:
            input_path: Path to input data
            file_format: 'csv' or 'parquet'
        
        Returns:
            DataFrame with loaded data
        """
        self.logger.info(f"Ingesting data from {input_path} (format: {file_format})")
        
        try:
            if file_format.lower() == "csv":
                df = self.spark.read \
                    .option("header", "true") \
                    .option("inferSchema", "false") \
                    .schema(TelecomSchemas.get_raw_schema()) \
                    .csv(input_path)
            elif file_format.lower() == "parquet":
                df = self.spark.read.parquet(input_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            row_count = df.count()
            self.logger.info(f"Loaded {row_count} rows with {len(df.columns)} columns")
            return df
        
        except Exception as e:
            self.logger.error(f"Error ingesting data: {str(e)}")
            raise
    
    def validate_schema(self, df: DataFrame) -> bool:
        """Validate DataFrame schema"""
        expected_schema = TelecomSchemas.get_raw_schema()
        
        if len(df.columns) != len(expected_schema.fields):
            self.logger.error(f"Column count mismatch: {len(df.columns)} vs {len(expected_schema.fields)}")
            return False
        
        self.logger.info("Schema validation passed")
        return True
    
    def clean_data(self, df: DataFrame) -> DataFrame:
        """
        Clean and validate data
        - Handle missing values
        - Remove duplicates
        - Filter invalid records
        """
        self.logger.info("Starting data cleaning")
        initial_count = df.count()
        
        # Remove duplicates
        df = df.dropDuplicates(["user_id"])
        
        # Handle missing values
        # For numeric columns, fill with median or 0
        numeric_cols = ["monthly_charge", "total_data_usage_mb", 
                       "avg_call_duration_mins", "roaming_minutes"]
        
        for col in numeric_cols:
            df = df.fillna({col: 0.0})
        
        # For categorical columns, fill with 'Unknown'
        categorical_cols = ["contract_type", "payment_method", 
                           "internet_service", "device_protection",
                           "tech_support", "streaming_tv", "streaming_movies"]
        
        for col in categorical_cols:
            df = df.fillna({col: "Unknown"})
        
        # Filter invalid records
        df = df.filter(
            (F.col("customer_tenure_months") >= 0) &
            (F.col("monthly_charge") >= 0) &
            (F.col("total_complaints") >= 0)
        )
        
        final_count = df.count()
        self.logger.info(f"Cleaning complete: {initial_count} -> {final_count} rows")
        
        return df
    
    def save_cleaned_data(self, df: DataFrame, output_path: str):
        """Save cleaned data as Parquet"""
        self.logger.info(f"Saving cleaned data to {output_path}")
        
        df.write \
            .mode("overwrite") \
            .partitionBy("record_date") \
            .parquet(output_path)
        
        self.logger.info("Data saved successfully")


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Feature engineering using Spark SQL and DataFrame operations"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_features(self, df: DataFrame) -> DataFrame:
        """
        Create engineered features
        
        Features:
        - data_usage_per_month: Usage normalized by tenure
        - charge_per_tenure: Monthly charge relative to tenure
        - complaint_rate: Complaints per month of tenure
        - Encoded categorical variables
        """
        self.logger.info("Starting feature engineering")
        
        # Create derived features
        df = df.withColumn(
            "data_usage_per_month",
            F.when(F.col("customer_tenure_months") > 0,
                   F.col("total_data_usage_mb") / F.col("customer_tenure_months"))
            .otherwise(0.0)
        )
        
        df = df.withColumn(
            "charge_per_tenure",
            F.when(F.col("customer_tenure_months") > 0,
                   F.col("monthly_charge") / F.col("customer_tenure_months"))
            .otherwise(F.col("monthly_charge"))
        )
        
        df = df.withColumn(
            "complaint_rate",
            F.when(F.col("customer_tenure_months") > 0,
                   F.col("total_complaints") / F.col("customer_tenure_months"))
            .otherwise(F.col("total_complaints"))
        )
        
        self.logger.info("Feature engineering complete")
        return df
    
    def aggregate_features(self, df: DataFrame) -> DataFrame:
        """
        Aggregate features using Spark SQL
        Calculate rolling averages and totals
        """
        self.logger.info("Aggregating features")
        
        # Register as temp view for SQL operations
        df.createOrReplaceTempView("telecom_data")
        
        # SQL aggregations
        aggregated = self.spark.sql(f"""
            SELECT 
                user_id,
                customer_tenure_months,
                monthly_charge,
                AVG(total_data_usage_mb) OVER (
                    PARTITION BY user_id 
                    ORDER BY record_date 
                    ROWS BETWEEN {Config.LOOKBACK_DAYS} PRECEDING AND CURRENT ROW
                ) as avg_data_usage_90d,
                SUM(total_complaints) OVER (
                    PARTITION BY user_id 
                    ORDER BY record_date
                ) as cumulative_complaints,
                AVG(avg_call_duration_mins) OVER (
                    PARTITION BY user_id 
                    ORDER BY record_date 
                    ROWS BETWEEN {Config.LOOKBACK_DAYS} PRECEDING AND CURRENT ROW
                ) as avg_call_duration_90d,
                total_data_usage_mb,
                avg_call_duration_mins,
                total_complaints,
                contract_type,
                payment_method,
                internet_service,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
                roaming_minutes,
                last_interaction_days,
                churn,
                record_date
            FROM telecom_data
        """)
        
        self.logger.info("Aggregation complete")
        return aggregated
    
    def prepare_ml_features(self, df: DataFrame) -> Tuple[DataFrame, Pipeline]:
        """
        Prepare features for ML with VectorAssembler
        Returns: (transformed_df, pipeline)
        """
        self.logger.info("Preparing ML features")
        
        # First apply basic feature engineering
        df = self.create_features(df)
        
        # String indexers for categorical variables
        indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep")
            for col in ["contract_type", "payment_method", "internet_service"]
        ]
        
        # Feature columns for vector assembler
        feature_cols = [
            "customer_tenure_months",
            "monthly_charge",
            "total_data_usage_mb",
            "avg_call_duration_mins",
            "total_complaints",
            "roaming_minutes",
            "last_interaction_days",
            "data_usage_per_month",
            "charge_per_tenure",
            "complaint_rate",
            "contract_type_indexed",
            "payment_method_indexed",
            "internet_service_indexed"
        ]
        
        # Vector assembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        
        # Standard scaler
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=False
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=indexers + [assembler, scaler])
        
        self.logger.info("ML feature preparation complete")
        return df, pipeline


# ============================================================================
# ML PIPELINE
# ============================================================================

class ChurnModelTrainer:
    """Train and evaluate churn prediction models"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(self.__class__.__name__)
        self.evaluator = BinaryClassificationEvaluator(
            labelCol="churn",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
    
    def train_logistic_regression(self, training_df: DataFrame, pipeline: Pipeline) -> PipelineModel:
        """Train a logistic regression model and return the fitted PipelineModel."""
        self.logger.info("Training logistic regression model")

        lr = LogisticRegression(featuresCol="features", labelCol="churn", maxIter=50)

        full_pipeline = Pipeline(stages=pipeline.getStages() + [lr]) if hasattr(pipeline, "getStages") else Pipeline(stages=[lr])

        model = full_pipeline.fit(training_df)
        self.logger.info("Logistic regression training complete")
        return model

    def train_random_forest(self, training_df: DataFrame, pipeline: Pipeline) -> PipelineModel:
        """Train a random forest classifier and return the fitted PipelineModel."""
        self.logger.info("Training random forest model")

        rf = RandomForestClassifier(featuresCol="features", labelCol="churn", numTrees=100)

        full_pipeline = Pipeline(stages=pipeline.getStages() + [rf]) if hasattr(pipeline, "getStages") else Pipeline(stages=[rf])
        model = full_pipeline.fit(training_df)
        self.logger.info("Random forest training complete")
        return model

    def train(self, cleaned_data_path: str, model_output_path: str):
        """High-level train entry: loads cleaned data, prepares features, trains selected model, and saves it."""
        if not Path(cleaned_data_path).exists():
            raise FileNotFoundError(f"Cleaned data not found at {cleaned_data_path}")

        df = self.spark.read.parquet(cleaned_data_path)
        fe = FeatureEngineer(self.spark)
        df, feature_pipeline = fe.prepare_ml_features(df)

        if Config.MODEL_TYPE == "logistic_regression":
            model = self.train_logistic_regression(df, feature_pipeline)
        else:
            model = self.train_random_forest(df, feature_pipeline)

        # Persist model
        model_output_dir = Path(model_output_path)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model.write().overwrite().save(str(model_output_dir))
        self.logger.info(f"Model saved to {model_output_dir}")


class BatchPredictor:
    """Run batch predictions and persist results"""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict(self, input_path: str, model_path: str, output_path: str):
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input data not found at {input_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        df = self.spark.read.parquet(input_path)
        model = PipelineModel.load(str(model_path))
        preds = model.transform(df)

        # Simplify output
        output = preds.select("user_id", "churn", F.col("probability").alias("churn_probability"))
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output.write.mode("overwrite").parquet(output_path)
        self.logger.info(f"Predictions saved to {output_path}")


# ============================================================================
# FLASK API (simple endpoints)
# ============================================================================

def create_app(spark: SparkSession):
    app = Flask(__name__)
    CORS(app)

    trainer = ChurnModelTrainer(spark)
    predictor = BatchPredictor(spark)

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "spark_version": spark.version})

    @app.route("/api/metrics", methods=["GET"])
    def metrics():
        return jsonify({"status": "ok", "message": "metrics endpoint not implemented in this lightweight build"})

    @app.route("/api/predict", methods=["POST"])
    def run_prediction():
        payload = request.json or {}
        input_path = payload.get("input_path") or str(Config.CLEANED_DATA_PATH)
        model_path = payload.get("model_path") or str(Config.MODELS_PATH / "churn_model")
        output_path = payload.get("output_path") or str(Config.PREDICTIONS_PATH)

        try:
            predictor.predict(input_path, model_path, output_path)
            return jsonify({"status": "ok", "output_path": output_path})
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({"status": "error", "message": str(e)}), 500

    return app


def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Telecom Churn Prediction CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Train model using cleaned data")
    sub.add_parser("predict", help="Run batch predictions")
    sub.add_parser("api", help="Run API server")

    args = parser.parse_args()
    spark_mgr = SparkSessionManager()
    spark = spark_mgr.get_spark()

    if args.cmd == "train":
        trainer = ChurnModelTrainer(spark)
        trainer.train(str(Config.CLEANED_DATA_PATH), str(Config.MODELS_PATH / "churn_model"))
    elif args.cmd == "predict":
        predictor = BatchPredictor(spark)
        predictor.predict(str(Config.FEATURES_PATH), str(Config.MODELS_PATH / "churn_model"), str(Config.PREDICTIONS_PATH))
    elif args.cmd == "api":
        app = create_app(spark)
        app.run(host=Config.API_HOST, port=Config.API_PORT, debug=Config.API_DEBUG)
    else:
        parser.print_help()


if __name__ == "__main__":
    Config.create_directories()
    main_cli()