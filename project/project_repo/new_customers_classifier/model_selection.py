import datetime
import json
import time

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import utils
from new_customers_classifier import config



# Get best run in today's experiment (by f1_score)
experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
experiment_ids = [experiment.experiment_id]

experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids,
    order_by=["metrics.f1_score DESC"],
    max_results=1,
).iloc[0]

train_model_score = experiment_best["metrics.f1_score"]


# Load per-model results and identify best model by weighted F1
with open(config.MODEL_RESULTS_PATH, "r") as f:
    model_results = json.load(f)

results_df = pd.DataFrame(
    {model: val["weighted avg"] for model, val in model_results.items()}
).T

best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name


# Inspect current production model, if any
client = MlflowClient()

prod_models = [
    mv
    for mv in client.search_model_versions(f"name='{config.MODEL_NAME}'")
    if dict(mv)["current_stage"] == "Production"
]
prod_model_exists = len(prod_models) > 0

if prod_model_exists:
    prod_model = dict(prod_models[0])
    prod_model_version = prod_model["version"]
    prod_model_run_id = prod_model["run_id"]


# Decide whether to register a new model
model_status = {}
run_id = None

if prod_model_exists:
    prod_run = mlflow.get_run(prod_model_run_id)
    prod_model_score = prod_run.data.metrics["f1_score"]

    model_status["current"] = train_model_score
    model_status["prod"] = prod_model_score

    if train_model_score > prod_model_score:
        run_id = experiment_best["run_id"]
else:
    run_id = experiment_best["run_id"]


# Register best model (if needed) and wait until ready
if run_id is not None:
    model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=run_id,
        artifact_path=config.ARTIFACT_PATH_NAME,
    )

    model_details = mlflow.register_model(model_uri=model_uri, name=config.MODEL_NAME)
    utils.wait_until_ready(model_details.name, model_details.version)

    model_details = dict(model_details)

with open("output/results.txt", "w") as f:
    f.write("Success!")


deployment_config = {
    "model_name": model_details['name'],
    "model_version": model_details['version']
}

with open("deployment_config.json", "w") as f:
    json.dump(deployment_config, f)