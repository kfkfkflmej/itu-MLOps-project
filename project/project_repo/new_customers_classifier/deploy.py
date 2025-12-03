
import json
from mlflow.tracking import MlflowClient
import utils

client = MlflowClient()

with open("deployment_config.json", "r") as f:
        config = json.load(f)
        model_name = config["model_name"]
        model_version = config["model_version"]

model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
model_status = True
if model_version_details['current_stage'] != 'Staging':
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,stage="Staging", 
        archive_existing_versions=True
    )
    model_status = utils.wait_for_deployment(client, model_name, model_version, 'Staging')