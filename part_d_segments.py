import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

PROFILE_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]   # lowest -> highest delay rate
TIER_TO_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

hub_profile = table.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index()
hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)
hub_profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

hub_features_scaled = StandardScaler().fit_transform(hub_profile[PROFILE_COLS])


def across_over_within(points: np.ndarray, labels: np.ndarray) -> float:
    across = []
    within = []
    for i, j in combinations(range(len(points)), 2):
        distance = float(np.linalg.norm(points[i] - points[j]))
        if labels[i] == labels[j]:
            within.append(distance)
        else:
            across.append(distance)
    return float(np.mean(across) / np.mean(within))


separation_rows = []
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(hub_features_scaled)
    sizes = sorted(np.bincount(labels), reverse=True)
    separation_rows.append({
        "K": k,
        "across_over_within": across_over_within(hub_features_scaled, labels),
        "cluster_sizes": "-".join(str(s) for s in sizes),
    })
separation_df = pd.DataFrame(separation_rows).round(4)
separation_df.to_csv("outputs/part_d_separation.csv", index=False)

final_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
final_kmeans.fit(hub_features_scaled)
hub_profile["cluster"] = final_kmeans.labels_

clusters_by_delay = hub_profile.groupby("cluster")["delay_rate"].mean().sort_values().index
segment_of_cluster = {}
for i, cluster_id in enumerate(clusters_by_delay):
    segment_of_cluster[cluster_id] = SEGMENT_NAMES[i]
hub_profile["segment"] = hub_profile["cluster"].map(segment_of_cluster)

segments = hub_profile[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments.to_csv("outputs/hub_segments.csv", index=False)

segment_profile = hub_profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean"),
).reindex(SEGMENT_NAMES).reset_index().round(4)
segment_profile.to_csv("outputs/segment_profile.csv", index=False)

expected_segment = segments["hub_tier"].map(TIER_TO_SEGMENT)
mismatch = segments[segments["segment"] != expected_segment]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print(hub_profile.round(4).to_string(index=False))
print(separation_df.to_string(index=False))
print(segment_profile.to_string(index=False))
print(mismatch.to_string(index=False))
