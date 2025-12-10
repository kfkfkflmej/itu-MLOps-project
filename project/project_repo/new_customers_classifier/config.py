import os
import datetime

#directories and paths
DATA_DIR = "data"
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim") 
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external") #currently not used?
MODELS_DIR = "models"
OUTPUT_DIR = "output"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "raw_data.csv")
TRAINING_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
TRAIN_DATA_GOLD_PATH = os.path.join(PROCESSED_DATA_DIR, "train_data_gold.csv")

#interim files
DATE_LIMITS_PATH = os.path.join(INTERIM_DATA_DIR, "date_limits.json")
OUTLIER_SUMMARY_PATH = os.path.join(INTERIM_DATA_DIR, "outlier_summary.csv")
CAT_MISSING_IMPUTE_PATH = os.path.join(INTERIM_DATA_DIR, "cat_missing_impute.csv")
SCALER_PATH = os.path.join(INTERIM_DATA_DIR, "scaler.pkl")
COLUMNS_DRIFT_PATH = os.path.join(INTERIM_DATA_DIR, "columns_drift.json")

#test sets
X_TEST_PATH = os.path.join(INTERIM_DATA_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(INTERIM_DATA_DIR, "y_test.csv")

#model related
XGBOOST_MODEL_PKL = os.path.join(MODELS_DIR, "lead_model_xgb.pkl")
XGBOOST_MODEL_JSON = os.path.join(MODELS_DIR, "lead_model_xgboost.json")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "lead_model_lr.pkl")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")

#results, configs
COLUMNS_LIST_PATH = os.path.join(MODELS_DIR, "columns_list.json")
MODEL_RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.json")
DEPLOYMENT_CONFIG_PATH = os.path.join(MODELS_DIR, "deployment_config.json")

#mlflow stuff
CURRENT_DATE = datetime.datetime.now().strftime("%Y_%B_%d")
EXPERIMENT_NAME = CURRENT_DATE
MODEL_NAME = "lead_model"
ARTIFACT_PATH_NAME = "model" 
MAX_DATE_STR = "2024-01-31"
MIN_DATE_STR = "2024-01-01"
DEPLOYMENT_STAGE = "Staging"