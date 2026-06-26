import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from submetrics_calculation  import extract_features



df = pd.read_excel("/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/rosie_mind_v3_annotated_sample2.xlsx")

print("\nDataset loaded.")
print(df.head())




df = df.dropna(subset=[
    "question",
    "anchor_passage",
    "grounding",
    "answerable",
    "contextualized",
    "total"
])



feature_rows = []

for _, row in df.iterrows():

    features = extract_features(row)

    feature_rows.append(features)

X = pd.DataFrame(feature_rows)

y = df["total"]


print("\nFeatures extracted.")
print(X.head())



# X = dataframe de features
# y = target

analysis_df = X.copy()
analysis_df["total"] = y

corr = analysis_df.corr(method="spearman")

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f"
)

plt.title("Objective metrics")
plt.tight_layout()
plt.savefig('/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/correlation_case_4.jpg')
plt.show()


'''
print(X[[ #llmasajudge
    "grounding",
    "correctness",
    "contextualized"
]].describe())
print(X[[
    "grounding",
    "correctness",
    "contextualized"
]].head(10))
'''

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = LassoCV(
    cv=5,
    random_state=42,
    max_iter=10000
)

model.fit(X_train, y_train)



predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

pearson = pearsonr(
    y_test,
    predictions
)[0]

spearman = spearmanr(
    y_test,
    predictions
)[0]


print("evaluation")

print(f"MAE:       {mae:.4f}")
print(f"Pearson:   {pearson:.4f}")
print(f"Spearman:  {spearman:.4f}")


# weights

weights = pd.DataFrame({
    "feature": X.columns,
    "weight": model.coef_
})

weights = weights.sort_values(
    by="weight",
    ascending=False
)

print("weights")


print(weights)




joblib.dump(
    model,
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/lasso_metric.pkl"
)

joblib.dump(
    list(X.columns),
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/feature_order.pkl"
)

#print("\nModel saved.")
'''
print("spearmarn grounding:")
print(
    spearmanr(
        X["grounding"],
        df["grounding"]
    )
)
print("spearmarn correctness:")

print(
    spearmanr(
        X["correctness"],
        df["correctness"]
    )
)
print("spearmarn contextualized:")

print(
    spearmanr(
        X["contextualized"],
        df["contextualized"]
    )
)
'''