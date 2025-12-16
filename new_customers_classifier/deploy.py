import json
import os
import pickle
import shutil
import tempfile
from mlflow.tracking import MlflowClient
import mlflow
import utils  # Needed to unpickle the custom wrapper
from new_customers_classifier import config

# Initialize client
client = MlflowClient()

# Load deployment configuration
with open(config.DEPLOYMENT_CONFIG_PATH, "r") as f:
    deployment_conf = json.load(f)
    model_name = deployment_conf["model_name"]
    model_version = deployment_conf["model_version"]

model_version_details = dict(client.get_model_version(name=model_name, version=model_version))

# Manage Model Stage Transition
if model_version_details['current_stage'] != config.DEPLOYMENT_STAGE:
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=config.DEPLOYMENT_STAGE, 
        archive_existing_versions=True
    )

# Wait for deployment readiness
if utils.wait_for_deployment(client, model_name, model_version, config.DEPLOYMENT_STAGE):
    print(f"Model {model_name} v{model_version} is ready. Extracting pure artifact...")

    model_root_uri = f"models:/{model_name}/{model_version}"
    wrapper_uri = f"{model_root_uri}/python_model.pkl"
    
    # Use a temporary directory that cleans itself up automatically
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download wrapper to temp location
        wrapper_path = mlflow.artifacts.download_artifacts(
            artifact_uri=wrapper_uri, 
            dst_path=temp_dir
        )
        
        # Load the wrapper (requires 'utils' import to be active)
        with open(wrapper_path, "rb") as f:
            wrapper_instance = pickle.load(f)
        
        # Extract the inner pure model (sklearn/xgboost object)
        # Fallback to full object if .model attribute is missing
        pure_model = getattr(wrapper_instance, "model", wrapper_instance)

        # Define final destination
        final_dest_dir = os.path.join(config.MODELS_DIR, "deployed_model")
        os.makedirs(final_dest_dir, exist_ok=True)
        final_model_path = os.path.join(final_dest_dir, "model.pkl")
        
        # Save pure model
        with open(final_model_path, "wb") as f:
            pickle.dump(pure_model, f)
            
