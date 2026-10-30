import os

import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute_table = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute_table["column"], impute_table["median"].astype(float)))

cleaned, _ = clean_shipments(new_shipments, medians=medians)
feature_table = build_features(cleaned, hubs)

result = feature_table[["shipment_id", "hub_id"]].copy()
delay_proba = model.predict_proba(feature_table[feature_order])[:, 1]
result["delay_probability"] = delay_proba.round(4)
result["flagged"] = (result["delay_probability"] >= threshold).astype(int)
result = result.merge(segments, on="hub_id", how="left")


def pick_action(row: pd.Series) -> str:
    if row["flagged"] == 0:
        return "No action"
    if row["segment"] == "Escalate":
        return "Escalate to hub manager"
    return "Proactive customer call"


result["action"] = result.apply(pick_action, axis=1)

output_columns = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
result = result[output_columns]
result.to_csv("outputs/new_shipment_actions.csv", index=False)

print(result.to_string(index=False))
print("flagged:", int(result["flagged"].sum()),
      "| escalations:", int((result["action"] == "Escalate to hub manager").sum()))
