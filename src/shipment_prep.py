# shared constants and the two cleaning / feature functions used in every part

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

FILL_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(
    raw: pd.DataFrame, medians: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    shipments = raw.drop_duplicates().reset_index(drop=True)

    shipments["service_type"] = shipments["service_type"].str.strip().str.title()

    shipments["payment_mode"] = shipments["payment_mode"].fillna("Unknown")

    declared_text = shipments["declared_value"].astype("string")
    declared_text = declared_text.str.replace("₹", "", regex=False)
    declared_text = declared_text.str.replace(",", "", regex=False).str.strip()
    shipments["declared_value_inr"] = pd.to_numeric(
        declared_text, errors="coerce"
    ).astype(float)
    shipments = shipments.drop(columns=["declared_value"])

    shipments["dispatch_time"] = pd.to_datetime(shipments["dispatch_time"])

    shipments["parcel_weight_kg"] = shipments["parcel_weight_kg"].astype(float)
    shipments.loc[shipments["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    if medians is None:
        medians = {}
        for column in FILL_COLUMNS:
            medians[column] = float(shipments[column].median())
    else:
        medians = {column: float(medians[column]) for column in FILL_COLUMNS}

    for column in FILL_COLUMNS:
        shipments[column] = shipments[column].astype(float).fillna(medians[column])

    return shipments, medians


def build_features(clean_table: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    shipments = clean_table.merge(hubs[hub_cols], on="hub_id", how="left")

    shipments["promised_hours"] = (
        shipments["service_type"].map(PROMISED_HOURS).astype(int)
    )
    shipments["is_cod"] = (shipments["payment_mode"] == "COD").astype(int)
    shipments["is_weekend"] = (shipments["dispatch_time"].dt.dayofweek >= 5).astype(int)
    shipments["dispatch_hour"] = shipments["dispatch_time"].dt.hour.astype(int)
    shipments["log_declared_value"] = np.log(shipments["declared_value_inr"])
    return shipments
