import os

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

N_FEATURES_KEPT = 7
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
TIE_ORDER = ["default_forest_same_7_features", "grid_search", "randomized_search"]

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(
    "sizes:",
    len(X_train),
    len(X_val),
    len(X_test),
    "| delayed:",
    y_train.sum(),
    y_val.sum(),
    y_test.sum(),
)


def score_row(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
    }


model_rows = []

blind_predictions = np.zeros(len(y_val), dtype=int)
model_rows.append({"model": "blind_all_on_time", **score_row(y_val, blind_predictions)})

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
logreg.fit(X_train, y_train)
model_rows.append(
    {"model": "logistic_regression", **score_row(y_val, logreg.predict(X_val))}
)

best_depth = None
best_f1 = -1
best_tree_scores = None
for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree_model = DecisionTreeClassifier(
        criterion="gini", max_depth=depth, random_state=42
    )
    tree_model.fit(X_train, y_train)
    depth_scores = score_row(y_val, tree_model.predict(X_val))
    if depth_scores["f1"] > best_f1:
        best_depth, best_f1, best_tree_scores = depth, depth_scores["f1"], depth_scores
model_rows.append(
    {"model": "decision_tree_depth_" + str(best_depth), **best_tree_scores}
)

default_forest = RandomForestClassifier(n_estimators=100, random_state=42)
default_forest.fit(X_train, y_train)
model_rows.append(
    {
        "model": "random_forest_default",
        **score_row(y_val, default_forest.predict(X_val)),
    }
)

models_df = pd.DataFrame(model_rows).round(4)
models_df.to_csv("outputs/part_c_models.csv", index=False)

f_score_selector = SelectKBest(f_classif, k="all")
f_score_selector.fit(X_train, y_train)

feature_scores_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "f_score": f_score_selector.scores_,
        "rf_importance": default_forest.feature_importances_,
    }
)
feature_scores_df["f_rank"] = (
    feature_scores_df["f_score"].rank(ascending=False, method="first").astype(int)
)
feature_scores_df["is_noise"] = feature_scores_df["feature"].isin(NOISE_COLUMNS)
feature_scores_df = feature_scores_df[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]
feature_scores_df = feature_scores_df.sort_values("f_rank").round(4)
feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    selectk_pipeline = Pipeline(
        [
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    selectk_pipeline.fit(X_train, y_train)
    selectk_rows.append(
        {"k": k, "val_f1": f1_score(y_val, selectk_pipeline.predict(X_val))}
    )
selectk_df = pd.DataFrame(selectk_rows).round(4)
selectk_df.to_csv("outputs/part_c_selectk.csv", index=False)

kbest_selector = SelectKBest(f_classif, k=N_FEATURES_KEPT)
kbest_selector.fit(X_train, y_train)
X_train_kbest = kbest_selector.transform(X_train)
X_val_kbest = kbest_selector.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_kbest, y_train)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42
)
random_search.fit(X_train_kbest, y_train)

untuned_forest = RandomForestClassifier(n_estimators=100, random_state=42)
untuned_forest.fit(X_train_kbest, y_train)


def ordered_params(final_params: dict) -> dict:
    return {
        "n_estimators": final_params["n_estimators"],
        "max_depth": final_params["max_depth"],
        "min_samples_leaf": final_params["min_samples_leaf"],
    }


grid_candidates = len(grid_search.cv_results_["params"])
random_candidates = len(random_search.cv_results_["params"])
default_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

tuning_df = pd.DataFrame(
    [
        {
            "search": "grid_search",
            "candidates": grid_candidates,
            "cv_folds": 5,
            "total_fits": grid_candidates * 5,
            "best_params": str(ordered_params(grid_search.best_params_)),
            "best_cv_accuracy": grid_search.best_score_,
            "val_f1": f1_score(y_val, grid_search.predict(X_val_kbest)),
        },
        {
            "search": "randomized_search",
            "candidates": random_candidates,
            "cv_folds": 5,
            "total_fits": random_candidates * 5,
            "best_params": str(ordered_params(random_search.best_params_)),
            "best_cv_accuracy": random_search.best_score_,
            "val_f1": f1_score(y_val, random_search.predict(X_val_kbest)),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": np.nan,
            "val_f1": f1_score(y_val, untuned_forest.predict(X_val_kbest)),
        },
    ]
)
tuning_df.round(4).to_csv("outputs/part_c_tuning.csv", index=False)

param_by_search = {
    "grid_search": ordered_params(grid_search.best_params_),
    "randomized_search": ordered_params(random_search.best_params_),
    "default_forest_same_7_features": default_params,
}
tuning_df["f1_rounded"] = tuning_df["val_f1"].round(10)
tuning_df["tie_rank"] = tuning_df["search"].map(TIE_ORDER.index)
winner = tuning_df.sort_values(
    ["f1_rounded", "tie_rank"], ascending=[False, True]
).iloc[0]
final_params = param_by_search[winner["search"]]

final_model = Pipeline(
    [
        ("selector", SelectKBest(f_classif, k=N_FEATURES_KEPT)),
        ("model", RandomForestClassifier(random_state=42, **final_params)),
    ]
)
final_model.fit(X_train, y_train)

val_proba = final_model.predict_proba(X_val)[:, 1]
threshold_rows = []
for cutoff in THRESHOLDS:
    threshold_rows.append(
        {"threshold": cutoff, **score_row(y_val, (val_proba >= cutoff).astype(int))}
    )
threshold_df = pd.DataFrame(threshold_rows)
threshold_df.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)

meets_recall = threshold_df[threshold_df["recall"] >= 0.80]
if len(meets_recall) > 0:
    chosen_threshold = float(meets_recall["threshold"].max())
else:
    chosen_threshold = 0.1

test_proba = final_model.predict_proba(X_test)[:, 1]
final_test = pd.DataFrame(
    [
        {
            "threshold": chosen_threshold,
            **score_row(y_test, (test_proba >= chosen_threshold).astype(int)),
        }
    ]
)
final_test.round(4).to_csv("outputs/part_c_final_test.csv", index=False)

if final_params["max_depth"] is None:
    depth_label = "unlimited"
else:
    depth_label = final_params["max_depth"]

final_config = pd.DataFrame(
    [
        {
            "selector_k": N_FEATURES_KEPT,
            "n_estimators": final_params["n_estimators"],
            "max_depth": depth_label,
            "min_samples_leaf": final_params["min_samples_leaf"],
            "threshold": chosen_threshold,
            "chosen_from": winner["search"],
        }
    ]
)
final_config.to_csv("outputs/part_c_final_config.csv", index=False)

noise_share = feature_scores_df.loc[
    feature_scores_df["is_noise"], "rf_importance"
].sum()
print(models_df.to_string(index=False))
print(feature_scores_df.to_string(index=False))
print("noise importance share:", round(noise_share, 4))
print(selectk_df.to_string(index=False))
print(
    tuning_df.drop(columns=["f1_rounded", "tie_rank"]).round(4).to_string(index=False)
)
print(threshold_df.round(4).to_string(index=False))
print(final_test.round(4).to_string(index=False))
print(final_config.to_string(index=False))
