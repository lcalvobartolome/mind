import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from submetrics_calculation import extract_features



DATASET_PATH = ("/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/rosie_mind_v3_annotated_second_round.xlsx")

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

X.to_excel('/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/X_with_qwen327b_nuevas_features_nuevas_metricas_new_prompt.xlsx', index=False)
y = df["total"]

print("\nFeatures extracted.")
print(X.head())


# RFECV
'''
base_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced"
)

rfecv = RFECV(
    estimator=base_model,
    step=1,
    cv=5,
    scoring="roc_auc",    #"roc_auc", "f1",
    n_jobs=-1
)

rfecv.fit(
    X,
    y
)

rfe_results = pd.DataFrame({
    "feature": X.columns,
    "selected": rfecv.support_,
    "ranking": rfecv.ranking_
})

rfe_results = rfe_results.sort_values(
    by="ranking"
)

print("RFECV RESULTS")

print(rfe_results)

print(
    "\nOptimal number of features:",
    rfecv.n_features_
)

rfe_results.to_excel(
    f"{OUTPUT_DIR}/rfecv_results.xlsx", #cambiar
    index=False
)

selected_features = rfe_results[
    rfe_results["selected"]
]["feature"].tolist()

print("\nSelected features:")
print(selected_features)

X_selected = X[selected_features]

'''
analysis_df = X.copy()
analysis_df["total"] = y

pearson_corr = analysis_df.corr(
    method="pearson"
)

plt.figure(figsize=(10,8))

sns.heatmap(
    pearson_corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/pearson_heatmap_todas_las_variables.jpg", #cambiar
    dpi=300,
    bbox_inches="tight"
)


summary = pd.DataFrame({
    "mean": X.mean(),
    "std": X.std(),
    "min": X.min(),
    "max": X.max(),
    "n_unique": X.nunique()
})
print(summary)
# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# FINAL MODEL

model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# PREDICTIONS

y_pred = model.predict(
    X_test
)

y_proba = model.predict_proba(
    X_test
)[:, 1]


# METRICS

auc = roc_auc_score(
    y_test,
    y_proba
)

balanced_acc = balanced_accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("LOGISTIC REGRESSION")

print(f"AUC:                {auc:.4f}")
print(f"Balanced Accuracy:  {balanced_acc:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1-score:           {f1:.4f}")


# THRESHOLD SEARCH

thresholds = np.arange(
    0.05,
    0.96,
    0.05
)

best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:

    preds = (
        y_proba >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_test,
        preds
    )

    if current_f1 > best_f1:

        best_f1 = current_f1
        best_threshold = threshold


print("THRESHOLD SEARCH")

print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    f"Best F1: {best_f1:.4f}"
)


# FEATURE WEIGHTS

weights = pd.DataFrame({
    "feature": X.columns,
    "weight": model.coef_[0]
})

weights = weights.sort_values(
    by="weight",
    ascending=False
)

print("LOGISTIC WEIGHTS")

print(weights)
'''
weights.to_excel(
    f"{OUTPUT_DIR}/logistic_weights.xlsx",
    index=False
)


# SAVE MODEL

joblib.dump(
    model,
    f"{OUTPUT_DIR}/logistic_metric.pkl"
)

joblib.dump(
    selected_features,
    f"{OUTPUT_DIR}/selected_features.pkl"
)

print("\nModel saved.")
'''