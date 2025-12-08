import os
#import datetime

#directories
ARTIFACTS_DIR = "artifacts"
OUTPUT_DIR = "output"

RAW_DATA_PATH = os.path.join(ARTIFACTS_DIR, "raw_data.csv")
TRAINING_DATA_PATH = os.path.join(ARTIFACTS_DIR, "training_data.csv")
TRAIN_DATA_GOLD_PATH = os.path.join(ARTIFACTS_DIR, "train_data_gold.csv")

#test sets
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "y_test.csv")

#----
DATE_LIMITS_PATH = os.path.join(ARTIFACTS_DIR, "date_limits.json")
OUTLIER_SUMMARY_PATH = os.path.join(ARTIFACTS_DIR, "outlier_summary.csv")
CAT_MISSING_IMPUTE_PATH = os.path.join(ARTIFACTS_DIR, "cat_missing_impute.csv")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
COLUMNS_DRIFT_PATH = os.path.join(ARTIFACTS_DIR, "columns_drift.json")
COLUMNS_LIST_PATH = os.path.join(ARTIFACTS_DIR, "columns_list.json")
MODEL_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "model_results.json")



# MODELS

#XGBoost
XGBOOST_MODEL_PKL = os.path.join(ARTIFACTS_DIR, "lead_model_xgboost.pkl")
XGBOOST_MODEL_JSON = os.path.join(ARTIFACTS_DIR, "lead_model_xgboost.json")

#logistic regression
LR_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lead_model_lr.pkl")

#best model 
BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
BEST_EXPERIMENT_PATH = os.path.join(ARTIFACTS_DIR, "best_experiment.pkl")



#Mlflow stuff:
CURRENT_DATE = datetime.datetime.now().strftime("%Y_%B_%d")
EXPERIMENT_NAME = CURRENT_DATE
MODEL_NAME = "lead_model"
ARTIFACT_PATH_NAME = "model" # The name of the folder inside MLflow artifacts
