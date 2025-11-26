"""This script is used to develop and track machine learning models for new customer classification using XGBoost and Logistic Regression.
It preprocesses the data, performs hyperparameter tuning, evaluates model performance, and logs the experiments
using MLflow.

Usage:
    python model_dev.py <data_gold_path>

Arguments:
    data_gold_path: Path to the CSV file containing the preprocessed data.

Requirements:
    - pandas
    - xgboost
    - scikit-learn
    - mlflow
    - scipy
    - joblib

    
Note:    Ensure that the necessary directories for artifacts and MLflow tracking are created before running the script.

Add: What do we want to print during the run?
    What do we move into a utilities.py script?
    Do we handle error messages with GO TOs or try/except?
    Do we want to change the naming methods for experiments/models/artifacts?

"""
# edit imports as needed
import os
import shutil
from pprint import pprint
import pandas as pd
import warnings
from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import mlflow.pyfunc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import joblib
import datetime
import json
import numpy as np
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import sys

#Setting up information for thraching experiments
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
artifact_path = "model"
model_name = "lead_model"
experiment_name = current_date

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# XGB wrapper
class XGBWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]
# LR wrapper
class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

#Helper functions
def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

# Load data
data_gold_path=sys.argv[1]
data = pd.read_csv(data_gold_path)
print(f"Training data length: {len(data)}")
data.head(5)

# Preprocess data
print("Preprocessing data...")
data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
cat_vars = data[cat_cols]

other_vars = data.drop(cat_cols, axis=1)

for col in cat_vars:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)

data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")
    print(f"Changed column {col} to float")

# Split data
X = data.drop(columns=['lead_indicator'])
y = data['lead_indicator']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)

print("Training...")
# Track XGB experiment    
with mlflow.start_run(experiment_id=experiment_id) as run:
    model = XGBRFClassifier(random_state=42)
    xgb_model_path = "./artifacts/lead_model_xgb.pkl"

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

    best_model = model_grid.best_estimator_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    #print("Best xgboost params")
    #pprint(model_grid.best_params_)
    #print("Accuracy train", accuracy_score(y_pred_train, y_train ))
    #print("Accuracy test", accuracy_score(y_pred_test, y_test))
    

    # log artifacts
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")
    
    # store model for model interpretability
    joblib.dump(value=model, filename=xgb_model_path)
        
    # Custom python model for predicting probability 
    mlflow.pyfunc.log_model('model', python_model=XGBWrapper(model))

# Track LR experiment 
with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()
    lr_model_path = "./artifacts/lead_model_lr.pkl"

    params = {
              'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              'penalty':  ["none", "l1", "l2", "elasticnet"],
              'C' : [100, 10, 1.0, 0.1, 0.01]
    }
    model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    print("Best lr params")
    pprint(model_grid.best_params_)
    print("Accuracy train", accuracy_score(y_pred_train, y_train ))
    print("Accuracy test", accuracy_score(y_pred_test, y_test))
    # log artifacts
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")
    
    # store model for model interpretability
    joblib.dump(value=model, filename=lr_model_path)
        
    # Custom python model for predicting probability 
    mlflow.pyfunc.log_model('model', python_model=lr_wrapper(model))

# Retrieve experiment results
client=MlflowClient()
client.search_experiments()

model_results = {
    xgb_model_path: classification_report(y_train, y_pred_train, output_dict=True)
}

model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

best_model_lr_params = model_grid.best_params_

print("Best lr params")
pprint(best_model_lr_params)

print("Accuracy train:", accuracy_score(y_pred_train, y_train ))
print("Accuracy test:", accuracy_score(y_pred_test, y_test))

conf_matrix = confusion_matrix(y_test, y_pred_test)
y_test = np.ravel(y_test)
y_pred_test = np.ravel(y_pred_test)
y_train = np.ravel(y_train)
y_pred_train = np.ravel(y_pred_train)
print("Test actual/predicted\n")
print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_test, y_pred_test),'\n')

conf_matrix = confusion_matrix(y_train, y_pred_train)
print("Train actual/predicted\n")
print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_train, y_pred_train),'\n')

model_results[lr_model_path] = model_classification_report
print(model_classification_report["weighted avg"]["f1-score"])

# Save column list and model results
print("Saving column list and model results...")
column_list_path = './artifacts/columns_list.json'
with open(column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)}
    pprint(columns)
    json.dump(columns, columns_file)

print('Saved column list to ', column_list_path)

model_results_path = "./artifacts/model_results.json"
with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)