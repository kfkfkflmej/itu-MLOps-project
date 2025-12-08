
import json
from mlflow.tracking import MlflowClient
import utils
from new_customers_classifier import config

client = MlflowClient()

with open(config.DEPLOYMENT_CONFIG_PATH, "r") as f:
        config = json.load(f)
        model_name = config["model_name"]
        model_version = config["model_version"]

model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
model_status = True
if model_version_details['current_stage'] != config.DEPLOYMENT_STAGE:
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=config.DEPLOYMENT_STAGE, 
        archive_existing_versions=True
    )
    model_status = utils.wait_for_deployment(client, model_name, model_version, config.DEPLOYMENT_STAGE)