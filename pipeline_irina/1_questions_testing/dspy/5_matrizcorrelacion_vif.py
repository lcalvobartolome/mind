import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

from submetrics_calculation import extract_features




DATASET_PATH = (
   "/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/qwen27b_merged.xlsx"
)

OUTPUT_DIR = (
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing"
)

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_excel(DATASET_PATH)

print("\nDataset loaded.")


df = df.dropna(
    subset=[
        "question",
        "anchor_passage",
        "total"
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

print(
    "\nSBERT cosine statistics:"
)

print(
    X["Cosine_SBERT"].describe()
)

features_df = X.copy()
features_df["total"] = y



summary = pd.DataFrame({
    "mean": X.mean(),
    "std": X.std(),
    "min": X.min(),
    "max": X.max(),
    "n_unique": X.nunique()
})

print("\n=========================")
print("FEATURE SUMMARY")
print("=========================")

print(summary)
'''
summary.to_excel(
    f"{OUTPUT_DIR}/feature_summary2.xlsx" #cambiar
)
'''

# PEARSON HEATMAP

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

plt.title("Pearson Correlation Matrix")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/final_correlation.jpg", #cambiar
    dpi=300,
    bbox_inches="tight"
)

'''
plt.close()

pairplot_df = analysis_df[
    [   "grounding",
        "answerable",
        "Cosine_SBERT",
        "BertSCORE",
        "entailment",
        "contradiction",
        "2gram_overlap",
        "total"
    ]
]
g = sns.pairplot(
    pairplot_df,
    hue="total",
    corner=True,
    height=4
) 

for ax in g.axes.flatten():

    if ax is not None:

        ax.set_xlabel(
            ax.get_xlabel(),
            fontsize=20
        )

        ax.set_ylabel(
            ax.get_ylabel(),
            fontsize=20
        )

        ax.tick_params(
            axis="both",
            labelsize=20
        )

# aumentar leyenda

if g._legend is not None:

    g._legend.set_title(
        "Class",
        prop={"size": 30}
    )

    for text in g._legend.texts:
        text.set_fontsize(30)



plt.savefig(
    f"{OUTPUT_DIR}/pairplot.jpg", #cambiar
    dpi=300
)


features_to_plot = [
    "grounding",
    "answerable",
    "Cosine_SBERT",
    "BertSCORE",
    "entailment",
    "contradiction",
    "2gram_overlap"
]


fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(15, 8)
)

axes = axes.flatten()

for i, feature in enumerate(features_to_plot):

    sns.boxplot(
        data=analysis_df,
        x="total",
        y=feature,
        ax=axes[i]
    )

    axes[i].set_title(
        feature,
        fontsize=20
    )

    axes[i].set_xlabel(
        "Class",
        fontsize=20
    )

    axes[i].set_ylabel(
        feature,
        fontsize=20
    )

# eliminar subplot vacío
for j in range(
    len(features_to_plot),
    len(axes)
):
    fig.delaxes(
        axes[j]
    )

plt.suptitle(
    "Feature distributions by class",
    fontsize=24
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/all_boxplots.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# FEATURE -> TARGET CORRELATIONS

correlations = []

for feature in X.columns:

    try:

        pearson_value = pearsonr(
            X[feature],
            y
        )[0]

    except:
        pearson_value = None

    try:

        spearman_value = spearmanr(
            X[feature],
            y
        ).statistic

    except:
        spearman_value = None

    correlations.append({
        "feature": feature,
        "pearson": pearson_value,
        "spearman": spearman_value
    })


corr_df = pd.DataFrame(
    correlations
)

corr_df = corr_df.sort_values(
    by="spearman",
    ascending=False
)

print("\n=========================")
print("FEATURE -> TARGET CORRELATIONS")
print("=========================")

print(corr_df)

#corr_df.to_excel(
#    f"{OUTPUT_DIR}/feature_target_correlations.xlsx",
#    index=False)


# CORRELATION BARPLOT

plt.figure(figsize=(8,5))

plt.barh(
    corr_df["feature"],
    corr_df["spearman"]
)

plt.xlabel(
    "Spearman correlation with target"
)

plt.title(
    "Feature Correlation with Target"
)

plt.tight_layout()

#plt.savefig(
#    f"{OUTPUT_DIR}/feature_target_correlation.jpg",
#    dpi=300,
#    bbox_inches="tight")

plt.close()


# VIF ANALYSIS

X_vif = X.copy()

constant_features = [
    col
    for col in X_vif.columns
    if X_vif[col].nunique() <= 1
]

if len(constant_features) > 0:

    print(
        "\nRemoving constant features from VIF:"
    )

    print(constant_features)

    X_vif = X_vif.drop(
        columns=constant_features
    )


vif_df = pd.DataFrame()

vif_df["feature"] = X_vif.columns

vif_df["VIF"] = [
    variance_inflation_factor(
        X_vif.values,
        i
    )
    for i in range(
        X_vif.shape[1]
    )
]

vif_df = vif_df.sort_values(
    by="VIF",
    ascending=False
)

print("VIF ANALYSIS")

print(vif_df)

#vif_df.to_excel(
#    f"{OUTPUT_DIR}/vif_analysis2.xlsx", #cambiar
#    index=False)

'''

