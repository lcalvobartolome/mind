import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path



X_PATH = (
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/"
    "X_with_qwen38b_nuevas_features_nuevas_metricas.xlsx"
)

DATASET_PATH = (
    "/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/"
    "rosie_mind_v3_annotated_second_round.xlsx"
)

OUTPUT_DIR = (
    "/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/terceras_pruebas"
)

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)



X = pd.read_excel(X_PATH)

df = pd.read_excel(DATASET_PATH)

y = df["total"]

print("Features loaded:")
print(X.shape)

print("Target loaded:")
print(y.shape)


analysis_df = X.copy()

analysis_df["total"] = y.values

corr_matrix = analysis_df.corr(
    method="pearson"
)

print("\nCorrelation matrix:")
print(corr_matrix)



corr_matrix.to_excel(
    f"{OUTPUT_DIR}/correlation_matrix.xlsx"
)



pairplot_df = analysis_df[
    [   "atomized",
        "answerable",
        "contextualized",
        "noun_missing_ratio",
        "lexical_overlap",
        "jaccard_overlap",
        "Cosine_SBERT",
        "BertSCORE",
        "entailment",
        "contradiction",
        "2gram_overlap",
        "3gram_overlap",
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