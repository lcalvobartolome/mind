import pandas as pd
import numpy as np

from itertools import combinations

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score
)


# PATHS

X_PATH = (
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/"
    "1_questions_testing/"
    "X_with_qwen38b_nuevas_features_nuevas_metricas.xlsx"
)

DATASET_PATH = (
    "/export/usuarios01/ivgomez/mind/pipeline_irina/"
    "1_questions_testing/"
    "rosie_mind_v3_annotated_second_round.xlsx"
)

OUTPUT_PATH = (
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/"
    "1_questions_testing/cuartas_pruebas/"
    "all_feature_combinations_auc.xlsx"
)



X = pd.read_excel(X_PATH)

df = pd.read_excel(DATASET_PATH)

y = df["total"]
'''
X = X_full.dropna(
    subset=[
        
    ])
'''
if "atomized" in X.columns and "jaccard_overlap" in X.columns:

    X["atomized_x_jaccard"] = (
        X["atomized"] *
        X["jaccard_overlap"]
    )

if "atomized" in X.columns and "Cosine_SBERT" in X.columns:

    X["atomized_x_cosine"] = (
        X["atomized"] *
        X["Cosine_SBERT"]
    )

print("X shape:", X.shape)
print("y shape:", y.shape)


# CROSS VALIDATION SETUP

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

base_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# TEST ALL COMBINATIONS

results = []

feature_names = list(X.columns)

for r in range(1, len(feature_names) + 1):

    for subset in combinations(feature_names, r):

        subset = list(subset)

        X_subset = X[subset]

        auc_scores = cross_val_score(
            estimator=base_model,
            X=X_subset,
            y=y,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1
        )

        results.append({
            "features": ", ".join(subset),
            "n_features": len(subset),
            "auc_mean": auc_scores.mean(),
            "auc_std": auc_scores.std()
        })


# RESULTS

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="auc_mean",
    ascending=False
)

print("TOP 20 COMBINATIONS")

print(
    results_df.head(20)
)

results_df.to_excel(
    OUTPUT_PATH,
    index=False
)


# BEST MODEL

best_row = results_df.iloc[0]

print("BEST COMBINATION")

print(
    f"Features: {best_row['features']}"
)

print(
    f"AUC mean: {best_row['auc_mean']:.4f}"
)

print(
    f"AUC std: {best_row['auc_std']:.4f}"
)