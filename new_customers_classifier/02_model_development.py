"""
This script is used to develop and track machine learning models for new customer classification 
using XGBoost and Logistic Regression. It preprocesses the data, performs hyperparameter tuning, 
evaluates model performance, and logs the experiments using MLflow.

Usage:
    python model_dev.py <data_gold_path>

Arguments:
    data_gold_path: Path to the CSV file containing the preprocessed data.
"""
# edit imports as needed
import os
import pandas as pd
from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import mlflow.pyfunc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
import datetime
import json
import numpy as np
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import sys
from new_customers_classifier import utils
from new_customers_classifier import config

# --- Setup ---
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

mlflow.set_experiment(config.EXPERIMENT_NAME)
mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME).experiment_id

# --- Load Data ---
if len(sys.argv) > 1:
    data_gold_path = sys.argv[1]
else:
    data_gold_path = config.TRAIN_DATA_GOLD_PATH
    
data = pd.read_csv(data_gold_path)

model_features = [
    'purchases',
    'time_spent',
    'n_visits',
    'customer_group_2',
    'customer_group_3',
    'customer_group_4',
    'customer_group_5',
    'customer_group_6',
    'customer_group_7',
    'customer_group_8',
    'customer_group_9',
    'onboarding_True'
]

# 2. Add missing feature columns with 0
for col in model_features:
    if col not in data.columns:
        data[col] = 0


# 4. Final Selection
final_columns = model_features + ['lead_indicator']

# Filter data.
available_cols = [c for c in final_columns if c in data.columns]
data = data[available_cols]

print(f"Data successfully aligned. Shape: {data.shape}")

# --- Split Data ---
X = data.drop(columns=['lead_indicator'])
y = data['lead_indicator']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)

print(f"Training features: {list(X_train.columns)}")

# Dictionary to store results to ensure we don't overwrite them
model_results = {}

# --- Track XGB experiment ---
with mlflow.start_run(experiment_id=experiment_id) as run:
    model = XGBRFClassifier(random_state=42)
    xgb_model_path = config.XGBOOST_MODEL_PKL

    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }
    
    model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)
    model_grid.fit(X_train, y_train)

    best_xgb_model = model_grid.best_estimator_

    y_pred_train_xgb = model_grid.predict(X_train)
    y_pred_test_xgb = model_grid.predict(X_test)
 
    # log artifacts
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test_xgb))
    mlflow.log_artifacts(config.MODELS_DIR, artifact_path=config.ARTIFACT_PATH_NAME)
    mlflow.log_param("data_version", "00000")
    
    # Store the BEST model
    joblib.dump(value=best_xgb_model, filename=xgb_model_path)
        
    # Custom python model for predicting probability 
    mlflow.pyfunc.log_model('model', python_model=utils.XGBWrapper(best_xgb_model))

    # Save results specifically for XGB
    model_results[xgb_model_path] = classification_report(y_train, y_pred_train_xgb, output_dict=True)


# --- Track LR experiment ---
with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()
    lr_model_path = config.LR_MODEL_PATH

    params = {
              'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              'penalty':  ["l2"], # Restricted to l2 to ensure solver compatibility
              'C' : [100, 10, 1.0, 0.1, 0.01]
    }
    
    model_grid = RandomizedSearchCV(model, param_distributions=params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    # Use the BEST trained model
    best_lr_model = model_grid.best_estimator_

    y_pred_train_lr = model_grid.predict(X_train)
    y_pred_test_lr = model_grid.predict(X_test)

    # log artifacts
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test_lr))
    mlflow.log_artifacts(config.MODELS_DIR, artifact_path=config.ARTIFACT_PATH_NAME)
    mlflow.log_param("data_version", "00000")
    
    # Store the BEST model
    joblib.dump(value=best_lr_model, filename=lr_model_path)
        
    # Custom python model for predicting probability 
    mlflow.pyfunc.log_model('model', python_model=utils.lr_wrapper(best_lr_model))

    # Save results specifically for LR
    model_results[lr_model_path] = classification_report(y_test, y_pred_test_lr, output_dict=True)


# --- Save Output ---
column_list_path = config.COLUMNS_LIST_PATH
with open(column_list_path, 'w+') as columns_file:
    # This saves the feature columns used for training
    columns = {'column_names': list(X_train.columns)}
    json.dump(columns, columns_file)

model_results_path = config.MODEL_RESULTS_PATH
with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)

print(f"Model development complete. Results saved to {config.MODEL_RESULTS_PATH}")
