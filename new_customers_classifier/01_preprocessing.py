import pandas as pd
import os
import warnings
import datetime
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from new_customers_classifier import utils
import sys
from new_customers_classifier import config

# --- Setup ---
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

os.makedirs(config.INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

# --- Load Data ---
if len(sys.argv) > 1:
    data = pd.read_csv(sys.argv[1])
else:
    data = pd.read_csv(config.RAW_DATA_PATH)

# --- Time Limit Data ---
max_date = config.MAX_DATE_STR
min_date = config.MIN_DATE_STR

if not max_date:
    max_date = pd.to_datetime(datetime.datetime.now().date()).date()
else:
    max_date = pd.to_datetime(max_date).date()

min_date = pd.to_datetime(min_date).date()
data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

# Save date limits
date_limits = {
    "min_date": str(data["date_part"].min()), 
    "max_date": str(data["date_part"].max())
}
with open(config.DATE_LIMITS_PATH, "w") as f:
    json.dump(date_limits, f)

# --- Drop Irrelevant Columns ---
data = data.drop(
    ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"],
    axis=1
)

# --- Data Cleaning ---
data["lead_indicator"].replace("", np.nan, inplace=True)
data["lead_id"].replace("", np.nan, inplace=True)
data["customer_code"].replace("", np.nan, inplace=True)
data = data.dropna(axis=0, subset=["lead_indicator"])
data = data.dropna(axis=0, subset=["lead_id"])

# --- Categorical Conversions ---
vars_to_obj = [
    "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
]
for col in vars_to_obj:
    data[col] = data[col].astype("object")


bool_cols = data.select_dtypes(include=['bool']).columns
if len(bool_cols) > 0:
    data[bool_cols] = data[bool_cols].astype(int)
    print(f"Recovered boolean columns: {list(bool_cols)}")


cont_vars = data.select_dtypes(include=['float64', 'int64', 'int32'])
cat_vars = data.select_dtypes(include=['object'])

# --- Outlier Handling ---
cont_vars = cont_vars.apply(lambda x: x.clip(lower=(x.mean() - 2 * x.std()),
                                             upper=(x.mean() + 2 * x.std())))
outlier_summary = cont_vars.apply(utils.describe_numeric_col).T
outlier_summary.to_csv(config.OUTLIER_SUMMARY_PATH)

# --- Imputation ---
cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
cat_missing_impute.to_csv(config.CAT_MISSING_IMPUTE_PATH)

cat_vars.loc[cat_vars['customer_code'].isna(), 'customer_code'] = 'None'
cat_vars = cat_vars.apply(utils.impute_missing_values)

cont_vars = cont_vars.apply(utils.impute_missing_values)

# --- Scaling ---
scaler = MinMaxScaler()
scaler.fit(cont_vars)
joblib.dump(value=scaler, filename=config.SCALER_PATH)

cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

# --- Recombine Data ---
cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)
data = pd.concat([cat_vars, cont_vars], axis=1)

# --- Feature Engineering (Bin Source) ---
data['bin_source'] = data['source']
values_list = ['li', 'organic', 'signup', 'fb']
data.loc[~data['source'].isin(values_list), 'bin_source'] = 'Others'

mapping = {'li': 'socials', 'fb': 'socials', 'organic': 'group1', 'signup': 'group1'}
data['bin_source'] = data['source'].map(mapping).fillna('Others')


encode_cols = ['customer_group', 'onboarding', 'source', 'bin_source']
data = pd.get_dummies(data, columns=encode_cols, drop_first=True)


model_features = [
    'visited_learn_more_before_booking', 'visited_faq', 'purchases', 'time_spent', 'n_visits', 
    'customer_group_2', 'customer_group_3', 'customer_group_4', 'customer_group_5', 
    'customer_group_6', 'customer_group_7', 'customer_group_8', 'customer_group_9', 
    'onboarding_True', 'bin_source_socials', 'source_li', 'source_organic', 'source_signup'
]

for col in model_features:
    if col not in data.columns:
        data[col] = 0

# 3. Define Metadata/Target columns to KEEP (Don't delete these!)
meta_cols = ['lead_indicator', 'lead_id', 'customer_code', 'date_part', 'domain', 'country']

# 4. Final Selection
final_columns = meta_cols + model_features

# Filter data.
available_cols = [c for c in final_columns if c in data.columns]
data = data[available_cols]

print(f"Data successfully aligned. Shape: {data.shape}")

# --- Save Data ---
data_columns = list(data.columns)
with open(config.COLUMNS_DRIFT_PATH, 'w+') as f:
    json.dump(data_columns, f)

data.to_csv(config.TRAIN_DATA_GOLD_PATH, index=False)
print(f"Gold data saved to {config.TRAIN_DATA_GOLD_PATH}")
