import json
from mlflow.tracking import MlflowClient
import mlflow
import utils
from new_customers_classifier import config

client = MlflowClient()

with open(config.DEPLOYMENT_CONFIG_PATH, "r") as f:
        deployment_conf = json.load(f)
        model_name = deployment_conf["model_name"]
        model_version = deployment_conf["model_version"]

model_version_details = dict(client.get_model_version(name=model_name, version=model_version))
model_status = True


if model_version_details['current_stage'] != config.DEPLOYMENT_STAGE:
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=config.DEPLOYMENT_STAGE, 
        archive_existing_versions=True
    )

model_status = utils.wait_for_deployment(client, model_name, model_version, config.DEPLOYMENT_STAGE)

# export the model pickle file
if model_status:
    model_root_uri = f"models:/{model_name}/{model_version}"

    pickle_file_uri = f"{model_root_uri}/python_model.pkl"
    
    mlflow.artifacts.download_artifacts(
        artifact_uri=pickle_file_uri, 
        dst_path=f"{config.MODELS_DIR}/deployed_model" 
    )