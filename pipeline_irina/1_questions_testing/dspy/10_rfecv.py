import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression


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

OUTPUT_DIR = (
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/"
    "1_questions_testing/terceras_pruebas/"
    "rfecv_analysis"
)

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)


# LOAD DATA

X = pd.read_excel(X_PATH)

df = pd.read_excel(DATASET_PATH)

y = df["total"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFeatures:")
print(list(X.columns))

summary = pd.DataFrame({
    "mean": X.mean(),
    "std": X.std(),
    "min": X.min(),
    "max": X.max(),
    "median": X.median(),
    "n_unique": X.nunique()
})

print(summary)

base_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced"
)

rfecv = RFECV(
    estimator=base_model,
    step=1,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

rfecv.fit(
    X,
    y
)


# RESULTS

results = pd.DataFrame({
    "feature": X.columns,
    "selected": rfecv.support_,
    "ranking": rfecv.ranking_
})

results = results.sort_values(
    by="ranking"
)

print("RFECV RESULTS")

print(results)

print(
    "\nOptimal number of features:",
    rfecv.n_features_
)

results.to_excel(
    f"{OUTPUT_DIR}/rfecv_results.xlsx",
    index=False
)


# SELECTED FEATURES

selected_features = results[
    results["selected"]
]["feature"].tolist()

print("\nSelected features:")
print(selected_features)

pd.DataFrame({
    "selected_features": selected_features
}).to_excel(
    f"{OUTPUT_DIR}/selected_features.xlsx",
    index=False
)


# RFECV CURVE

plt.figure(figsize=(8,5))

plt.plot(
    range(
        1,
        len(rfecv.cv_results_["mean_test_score"]) + 1
    ),
    rfecv.cv_results_["mean_test_score"],
    marker="o"
)

plt.xlabel(
    "Number of selected features"
)

plt.ylabel(
    "Mean CV AUC"
)

plt.title(
    "RFECV Feature Selection"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/rfecv_curve.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# BEST SCORE

best_auc = max(
    rfecv.cv_results_["mean_test_score"]
)

print(
    "\nBest cross-validated AUC:",
    round(best_auc, 4)
)

