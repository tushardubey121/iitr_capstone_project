import os

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["delivery_hours"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)

split_checks = pd.DataFrame(
    {
        "metric": [
            "train_rows",
            "val_rows",
            "test_rows",
            "train_scaled_max_abs_mean",
            "test_scaled_max_abs_mean",
        ],
        "value": [
            len(X_train),
            len(X_val),
            len(X_test),
            round(float(np.abs(train_scaled.mean(axis=0)).max()), 4),
            round(float(np.abs(test_scaled.mean(axis=0)).max()), 4),
        ],
    },
    dtype=object,
)
split_checks.to_csv("outputs/part_b_checks.csv", index=False)


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def get_scores(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(actual, predicted),
    }


baseline_pred = np.full(len(y_val), y_train.mean())
metric_rows = [
    {"model": "mean_baseline", "split": "val", **get_scores(y_val, baseline_pred)}
]

depth_list = [1, 2, 3, 4, 6, 8, 12, None]
depth_rows = []
best_depth = None
best_val_rmse = None
for depth in depth_list:
    regression_tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    regression_tree.fit(X_train, y_train)
    train_rmse = rmse(y_train, regression_tree.predict(X_train))
    val_rmse = rmse(y_val, regression_tree.predict(X_val))
    depth_rows.append(
        {
            "max_depth": "unlimited" if depth is None else depth,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        }
    )
    if best_val_rmse is None or round(val_rmse, 10) < round(best_val_rmse, 10):
        best_val_rmse = val_rmse
        best_depth = depth

depth_df = pd.DataFrame(depth_rows).round(4)
depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

candidates = {
    "linear_regression": Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    ),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}
for model_name, candidate in candidates.items():
    candidate.fit(X_train, y_train)
    metric_rows.append(
        {
            "model": model_name,
            "split": "val",
            **get_scores(y_val, candidate.predict(X_val)),
        }
    )

learning_rate_rows = []
for boost_rate in [0.01, 0.1, 1.0]:
    boosting_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=boost_rate, random_state=42
    )
    boosting_model.fit(X_train, y_train)
    val_pred = boosting_model.predict(X_val)
    learning_rate_rows.append(
        {
            "learning_rate": boost_rate,
            "val_rmse": rmse(y_val, val_pred),
            "val_r2": r2_score(y_val, val_pred),
        }
    )
learning_rate_df = pd.DataFrame(learning_rate_rows).round(4)
learning_rate_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

val_metrics = pd.DataFrame(metric_rows)
candidate_metrics = val_metrics[val_metrics["model"].isin(candidates)]
ranked_candidates = candidate_metrics.sort_values("RMSE", kind="stable")
top_two = ranked_candidates["model"].head(2).tolist()

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []
for model_name in top_two:
    cv_scores = cross_val_score(
        candidates[model_name], X_train_val, y_train_val, cv=kfold
    )
    cv_rows.append(
        {"model": model_name, "mean_r2": cv_scores.mean(), "std_r2": cv_scores.std()}
    )
cv_df = pd.DataFrame(cv_rows).round(4)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)

final_model_name = top_two[0]
metric_rows.append(
    {
        "model": final_model_name,
        "split": "test",
        **get_scores(y_test, candidates[final_model_name].predict(X_test)),
    }
)
all_metrics = pd.DataFrame(metric_rows).round(4)
all_metrics.to_csv("outputs/part_b_metrics.csv", index=False)

print(split_checks.to_string(index=False))
print(depth_df.to_string(index=False))
print(all_metrics.to_string(index=False))
print(learning_rate_df.to_string(index=False))
print(cv_df.to_string(index=False))
print("best tree depth:", best_depth, "| final model:", final_model_name)
