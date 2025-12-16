import os
import sys

artifacts_dir = "models/"
files_to_check = [
    "deployed_model/model.pkl"
    "columns_list.json",
    "lead_model_lr.pkl",
    "lead_model_xgb.pkl",
    "model_results.json",
    "deployment_config.json"
]

missing_files = [f for f in files_to_check if not os.path.isfile(os.path.join(artifacts_dir, f))]

if missing_files:
    print(f"Missing files: {', '.join(missing_files)}")
    sys.exit(1)

else:
    print("All required artifacts exist!")
    sys.exit(0)