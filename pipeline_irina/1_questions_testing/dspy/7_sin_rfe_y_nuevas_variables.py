import pandas as pd
import numpy as np
import joblib
from submetrics_calculation import extract_features
from pathlib import Path
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ( RepeatedStratifiedKFold, cross_val_score)
from sklearn.dummy import DummyClassifier



DATASET_PATH = ("/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/rosie_mind_v3_annotated_sample2.xlsx")

OUTPUT_DIR = ("/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/terceras_pruebas")

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_excel(DATASET_PATH)


df = df.dropna(
    subset=[
        "question",
        "anchor_passage",
        #"grounding",
        #"answerable",
        #"contextualized",
        "total",
    ]
)

feature_rows = []

for _, row in df.iterrows():

    features = extract_features(row)

    feature_rows.append(features)

X = pd.DataFrame(feature_rows)

y = df["total"]

print("\nFeatures extracted.")
print(X.head())




X["grounding_x_2gram"] = (X["grounding"]* X["2gram_overlap"])

X["grounding_x_sbert"] = (X["grounding"]* X["Cosine_SBERT"])


print("\nFeatures available:")
print(list(X.columns))


print("DUMMY CLASSIFIER")

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)

dummy_mf = DummyClassifier(
    strategy="most_frequent"
)

dummy_str = DummyClassifier(
    strategy="stratified",
    random_state=42
)

auc_mf = cross_val_score(
    dummy_mf,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

auc_str = cross_val_score(
    dummy_str,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print(
    f"Most frequent AUC: "
    f"{auc_mf.mean():.4f} ± {auc_mf.std():.4f}"
)

print(
    f"Stratified AUC: "
    f"{auc_str.mean():.4f} ± {auc_str.std():.4f}"
)

print(" END DUMMY CLASSIFIER")







model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)




results = []

features = list(X.columns)

total_combinations = 0

for k in range(
    1,
    len(features) + 1
):
    total_combinations += len(
        list(combinations(features, k))
    )

print(
    f"\nTesting {total_combinations} combinations..."
)

counter = 0

for k in range(
    1,
    len(features) + 1
):

    for subset in combinations(
        features,
        k
    ):

        counter += 1

        if counter % 100 == 0:

            print(
                f"{counter}/{total_combinations}"
            )

        X_subset = X[
            list(subset)
        ]

        scores = cross_val_score(
            model,
            X_subset,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        results.append({
            "features": ", ".join(subset),
            "n_features": len(subset),
            "auc_mean": scores.mean(),
            "auc_std": scores.std()
        })



results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="auc_mean",
    ascending=False
)

print("TOP 20 COMBINATIONS")

print(
    results_df.head(20)
)

results_df.to_excel(
    f"{OUTPUT_DIR}/exhaustive_auc_search.xlsx",
    index=False
)




best_features = (
    results_df.iloc[0]["features"]
    .split(", ")
)

print("BEST FEATURE SET")

print(best_features)

print(
    f"AUC mean: {results_df.iloc[0]['auc_mean']:.4f}"
)

print(
    f"AUC std:  {results_df.iloc[0]['auc_std']:.4f}"
)



X_best = X[
    best_features
]

model.fit(
    X_best,
    y
)

weights = pd.DataFrame({
    "feature": best_features,
    "weight": model.coef_[0]
})

weights = weights.sort_values(
    by="weight",
    ascending=False
)

print("FINAL MODEL WEIGHTS")

print(weights)
