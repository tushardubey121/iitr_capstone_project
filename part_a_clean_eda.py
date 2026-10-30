import os

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
dedup = raw.drop_duplicates()
clean_table, medians = clean_shipments(raw)
table = build_features(clean_table, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv("data/modelling_table.csv", index=False)

q1 = table["delivery_hours"].quantile(0.25)
q3 = table["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
too_low = table["delivery_hours"] < lower
too_high = table["delivery_hours"] > upper
n_outliers = int((too_low | too_high).sum())

spellings_after = dedup["service_type"].str.strip().str.title().nunique()
check_cols = FEATURES + ["delivery_hours", "is_delayed"]
weight_before = pd.to_numeric(dedup["parcel_weight_kg"], errors="coerce")
check_values = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", dedup["distance_km"].isna().sum()),
    ("missing_parcel_weight_before", weight_before.isna().sum()),
    ("sentinel_weight_rows", (weight_before > 100).sum()),
    ("missing_declared_value_before", dedup["declared_value"].isna().sum()),
    ("missing_payment_mode_before", dedup["payment_mode"].isna().sum()),
    ("service_type_spellings_before", dedup["service_type"].nunique()),
    ("service_type_spellings_after", spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", table[check_cols].isna().sum().sum()),
    ("delayed_count", table["is_delayed"].sum()),
    ("delay_rate", table["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", n_outliers),
    ("declared_value_mean_inr", table["declared_value_inr"].mean()),
    ("declared_value_median_inr", table["declared_value_inr"].median()),
    ("log_declared_value_mean", table["log_declared_value"].mean()),
    ("log_declared_value_median", table["log_declared_value"].median()),
]

check_rows = []
for name, value in check_values:
    if isinstance(value, (float, np.floating)):
        value = round(float(value), 4)
    else:
        value = int(value)
    check_rows.append((name, value))
checks_df = pd.DataFrame(check_rows, columns=["metric", "value"], dtype=object)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

table["distance_band"] = pd.cut(
    table["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
)

dim_tables = []
for dimension in ["service_type", "hub_tier", "distance_band"]:
    dim_table = (
        table.groupby(dimension, observed=True)
        .agg(
            shipments=("is_delayed", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        )
        .reset_index()
    )
    dim_table = dim_table.rename(columns={dimension: "group"})
    dim_table.insert(0, "dimension", dimension)
    dim_table["group"] = dim_table["group"].astype(str)
    dim_tables.append(dim_table)

by_group = pd.concat(dim_tables, ignore_index=True)
by_group["delay_rate"] = by_group["delay_rate"].round(4)
by_group["mean_hours"] = by_group["mean_hours"].round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

numeric_table = table.drop(columns=["distance_band"]).select_dtypes(include=[np.number])
correlations = numeric_table.corr().round(4)
correlations.to_csv("outputs/part_a_correlations.csv", index=True, index_label="column")

print(checks_df.to_string(index=False))
print(by_group.to_string(index=False))
top3_columns = (
    correlations["delivery_hours"]
    .drop("delivery_hours")
    .abs()
    .sort_values(ascending=False)
    .head(3)
)
print("top 3 correlations with delivery_hours:")
print(correlations.loc[top3_columns.index, "delivery_hours"].to_string())
