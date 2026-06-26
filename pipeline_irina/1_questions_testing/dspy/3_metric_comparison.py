import pandas as pd
import joblib

from sklearn.linear_model import (
    LinearRegression,
    RidgeCV,
    LassoCV
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from scipy.stats import pearsonr, spearmanr

from submetrics_calculation import extract_features


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_excel(
    "/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/dspy/rosie_mind_v3_annotated_sample.xlsx"
)

print("\nDataset loaded.")
print(df.head())


df = df.dropna(
    subset=[
        "question",
        "anchor_passage",
        "grounding",
        "correctness",
        "contextualized",
        "total",
    ]
)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

feature_rows = []

for _, row in df.iterrows():
    feature_rows.append(
        extract_features(row)
    )

X = pd.DataFrame(feature_rows)

y = df["total"]

print("\nFeatures extracted.")
print(X.head())


# =========================================================
# JUDGE ANALYSIS
# =========================================================

print("\n=========================")
print("JUDGES VS HUMAN")
print("=========================")

for feature in [
    "grounding",
    "correctness",
    "contextualized",
]:
    corr = spearmanr(
        X[feature],
        df[feature]
    )

    print(
        f"{feature:15s} "
        f"Spearman={corr.statistic:.4f} "
        f"p={corr.pvalue:.4f}"
    )


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# =========================================================
# MODELS
# =========================================================

models = {
    "Linear": LinearRegression(),

    "Ridge": RidgeCV(
        alphas=[0.01, 0.1, 1, 10, 100]
    ),

    "Lasso": LassoCV(
        cv=5,
        random_state=42,
        max_iter=10000
    ),
}


# =========================================================
# TRAIN + EVALUATE
# =========================================================

results = []

best_model = None
best_name = None
best_spearman = -999


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        preds
    )

    pearson = pearsonr(
        y_test,
        preds
    )[0]

    spearman = spearmanr(
        y_test,
        preds
    )[0]

    results.append({
        "model": name,
        "MAE": mae,
        "Pearson": pearson,
        "Spearman": spearman,
    })

    if spearman > best_spearman:
        best_spearman = spearman
        best_model = model
        best_name = name


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Spearman",
    ascending=False
)

print("\n=========================")
print("MODEL COMPARISON")
print("=========================")

print(results_df)


# =========================================================
# BEST MODEL
# =========================================================

print("\nBest model:", best_name)
print("Best Spearman:", best_spearman)


# =========================================================
# FEATURE WEIGHTS
# =========================================================

if hasattr(best_model, "coef_"):

    weights = pd.DataFrame({
        "feature": X.columns,
        "weight": best_model.coef_,
    })

    weights = weights.sort_values(
        by="weight",
        ascending=False
    )

    print("\n=========================")
    print("FEATURE WEIGHTS")
    print("=========================")

    print(weights)


# =========================================================
# SAVE BEST MODEL
# =========================================================

joblib.dump(
    best_model,
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/best_metric.pkl"
)

joblib.dump(
    list(X.columns),
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/feature_order.pkl"
)

print("\nModel saved.")