# Part D - refit the final pipeline and save it so predict_delay.py can reuse it
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# settings chosen in part C
final_config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
if str(final_config["max_depth"]) == "unlimited":
    max_depth = None
else:
    max_depth = int(final_config["max_depth"])
threshold = float(final_config["threshold"])

# same stratified split as part C
table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

final_pipeline = Pipeline(
    [
        ("selector", SelectKBest(f_classif, k=int(final_config["selector_k"]))),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_estimators=int(final_config["n_estimators"]),
                max_depth=max_depth,
                min_samples_leaf=int(final_config["min_samples_leaf"]),
            ),
        ),
    ]
)
final_pipeline.fit(X_train, y_train)

# save everything predict_delay.py needs
joblib.dump(final_pipeline, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv(
    "models/delay_threshold.csv", index=False
)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
impute_table = pd.DataFrame(
    {
        "column": list(medians.keys()),
        "median": [round(v, 4) for v in medians.values()],
    }
)
impute_table.to_csv("models/impute_values.csv", index=False)

# make sure the saved model gives the same answers after loading
loaded_model = joblib.load("models/delay_model.pkl")
proba_loaded = loaded_model.predict_proba(X_test)[:, 1]
proba_memory = final_pipeline.predict_proba(X_test)[:, 1]
max_diff = np.abs(proba_loaded - proba_memory).max()

reload_check = pd.DataFrame(
    {
        "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
        "value": [len(X_test), round(float(max_diff), 4), threshold],
    },
    dtype=object,
)
reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)
print(reload_check.to_string(index=False))
