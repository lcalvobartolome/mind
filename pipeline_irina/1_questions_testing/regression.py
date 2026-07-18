import pandas as pd

import joblib

from itertools import combinations

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)


Y_PATH  = "/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/rosie_mind_v3_annotated_second_round.xlsx"

X_PATH  = "/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/qwen27b_merged.xlsx"

OUT_PATH  = "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/terceras_pruebas/results_27b_reduced.xlsx"



X = pd.read_excel(X_PATH)
'''
X = X.drop(
    columns=[
        "question",
        "anchor_passage"
    ]
)
'''
X = X[["subordinate", "answerable", "jaccard"]]

df = pd.read_excel(Y_PATH)

y = df["total"]

print(f"Samples: {len(X)}")
print(f"Features: {len(X.columns)}")

model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X, y)

joblib.dump(model, "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/definitivo/question_quality.pkl")
'''
interaction_pairs = [

    ("subordinate", "2gram_overlap"),
    ("atomized", "Cosine_SBERT"),
    ("subordinate", "Cosine_SBERT"),
    ("answerable", "Cosine_SBERT"),
    ("contextualized", "Cosine_SBERT"),
]

for feature1, feature2 in interaction_pairs:

    if feature1 in X.columns and feature2 in X.columns:

        X[f"{feature1}*{feature2}"] = (
            X[feature1] * X[feature2]
        )


print("\nFinal feature list:\n")

for col in X.columns:
    print(col)


model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced"
)


cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)


results = []

feature_names = list(X.columns)


for r in range(1, 7): #maximo grupos de 6 features

    print(f"\nTesting combinations of {r} features...")

    for subset in combinations(feature_names, r):

        subset = list(subset)

        X_subset = X[subset]

        scores = cross_validate(

            estimator=model,

            X=X_subset,

            y=y,

            cv=cv,

            scoring={

                "auc": "roc_auc",

                "f1": "f1"

            },

            n_jobs=-1,

            return_train_score=False

        )

        auc_mean = scores["test_auc"].mean()
        auc_std = scores["test_auc"].std()

        f1_mean = scores["test_f1"].mean()
        f1_std = scores["test_f1"].std()

        results.append({

            "features":
                ", ".join(subset),

            "n_features":
                len(subset),

            "auc_mean":
                auc_mean,

            "auc_std":
                auc_std,

            "f1_mean":
                f1_mean,

            "f1_std":
                f1_std,

            # útil para priorizar modelos estables
            "auc_minus_std":
                auc_mean - auc_std,

            # útil para priorizar modelos sencillos
            "complexity":
                len(subset)

        })

print("\nAll combinations evaluated.")

# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["auc_mean", "f1_mean"],
    ascending=False
)

results_df.to_excel(
    OUT_PATH,
    index=False
)

print("\n===================================================")
print("TOP 20 MODELS (ordered by mean AUC)")
print("===================================================\n")

print(
    results_df[
        [
            "features",
            "n_features",
            "auc_mean",
            "auc_std",
            "f1_mean",
            "f1_std"
        ]
    ].head(20)
)


# =========================================================
# BEST MODEL
# =========================================================

best = results_df.iloc[0]

print("\n===================================================")
print("BEST MODEL")
print("===================================================\n")

print(f"Features      : {best['features']}")
print(f"N features    : {best['n_features']}")
print(f"AUC           : {best['auc_mean']:.4f} ± {best['auc_std']:.4f}")
print(f"F1            : {best['f1_mean']:.4f} ± {best['f1_std']:.4f}")


# =========================================================
# MOST STABLE MODELS
# =========================================================

stable_df = results_df.sort_values(
    by="auc_minus_std",
    ascending=False
)

print("\n===================================================")
print("TOP 10 MOST STABLE MODELS")
print("===================================================\n")

print(
    stable_df[
        [
            "features",
            "auc_mean",
            "auc_std",
            "auc_minus_std",
            "f1_mean"
        ]
    ].head(10)
)


# =========================================================
# SIMPLE MODELS (<=3 FEATURES)
# =========================================================

simple_models = results_df[
    results_df["n_features"] <= 3
]

print("\n===================================================")
print("BEST SIMPLE MODELS (<=3 FEATURES)")
print("===================================================\n")

print(
    simple_models[
        [
            "features",
            "n_features",
            "auc_mean",
            "auc_std",
            "f1_mean"
        ]
    ].head(20)
)

print("\nResults saved")
'''
