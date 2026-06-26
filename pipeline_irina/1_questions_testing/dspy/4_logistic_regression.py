
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from scipy.stats import spearmanr

from submetrics_calculation import extract_features


# =========================================================
# PATHS
# =========================================================

DATASET_PATH = (
    "/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/rosie_mind_v3_annotated_sample2.xlsx"
)

OUTPUT_DIR = (
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing"
)

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_excel(DATASET_PATH)

print("\nDataset loaded.")
print(df.head())

df = df.dropna(
    subset=[
        "question",
        "anchor_passage",
        "grounding",
        "answerable",
        "contextualized",
        "total",
    ]
)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

feature_rows = []

for _, row in df.iterrows():

    features = extract_features(row)

    feature_rows.append(features)

X = pd.DataFrame(feature_rows)

y = df["total"]

print("\nFeatures extracted.")
print(X.head())



# =========================================================
# CORRELATION HEATMAP
# =========================================================

analysis_df = X.copy()
analysis_df["total"] = y

corr = analysis_df.corr(
    method="spearman"
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f"
)

plt.title(
    "Spearman correlation matrix"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation4.jpg", #cambiar
    dpi=300,
    bbox_inches="tight"
)

plt.close()




print("\n=========================")
print("FEATURE CORRELATIONS")
print("=========================")

correlations = []

for col in X.columns:

    corr_value = spearmanr(
        X[col],
        y
    ).statistic

    correlations.append({
        "feature": col,
        "spearman": corr_value
    })

corr_df = pd.DataFrame(
    correlations
)

corr_df = corr_df.sort_values(
    by="spearman",
    ascending=False
)

print(corr_df)

plt.figure(figsize=(8, 5))

plt.barh(
    corr_df["feature"],
    corr_df["spearman"]
)

plt.xlabel(
    "Spearman correlation with target"
)

plt.title(
    "Feature correlation with target"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/logistic4.jpg", #cambiar
    dpi=300,
    bbox_inches="tight"
)

plt.close()



model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced"
)

#rfe = RFE( estimator=base_model, n_features_to_select=3)

rfe = RFECV(
    estimator=base_model,
    step=1,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


rfe.fit(
    X,
    y
)

rfe_results = pd.DataFrame({
    "feature": X.columns,
    "selected": rfe.support_,
    "ranking": rfe.ranking_
})

rfe_results = rfe_results.sort_values(
    by="ranking"
)

print("\n=========================")
print("RFE RESULTS")
print("=========================")

print(rfe_results)

print(
    "\nOptimal number of features:",
    rfe.n_features_
)

rfe_results.to_excel(
    f"{OUTPUT_DIR}/rfe_results4.xlsx", #cambiar
    index=False
)

selected_features = rfe_results[
    rfe_results["selected"]
]["feature"].tolist()

print(selected_features)

X_rfe = X[selected_features]

plt.figure(figsize=(8,5))

plt.plot(
    range(
        1,
        len(rfe.cv_results_["mean_test_score"]) + 1
    ),
    rfe.cv_results_["mean_test_score"]
)

plt.xlabel(
    "Number of selected features"
)

plt.ylabel(
    "Cross-validated F1"
)

plt.title(
    "RFECV feature selection"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/rfecv_curve.jpg", #cambiar

    dpi=300
)

plt.close()


X_train, X_test, y_train, y_test = train_test_split(
    X_rfe,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)



accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

print("\n=========================")
print("LOGISTIC REGRESSION")
print("=========================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/confusion_matrix2.jpg", #cmabiar
    dpi=300,
    bbox_inches="tight"
)

plt.close()


weights = pd.DataFrame({
    "feature": selected_features,
    "weight": model.coef_[0]
})

weights = weights.sort_values(
    by="weight",
    ascending=False
)

print("\n=========================")
print("LOGISTIC WEIGHTS")
print("=========================")

print(weights)

weights.to_excel(
    f"{OUTPUT_DIR}/logistic_weights4.xlsx", #cambiar
    index=False
)


joblib.dump(
    model,
    f"{OUTPUT_DIR}/logistic_metric4.pkl" #cambiar
)


print("\nModel saved.")
