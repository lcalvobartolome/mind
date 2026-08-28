from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/"
    "final_code/statistics_output/cases/CASE_3_C.parquet"
)

OUTPUT_DIR = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/"
    "final_db/case_3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(INPUT_PATH)

print("\n" + "=" * 80)
print("DATASET")
print("=" * 80)

print(f"Total rows: {len(df):,}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("MAPPED LABEL DISTRIBUTION")
print("=" * 80)

print(
    df["mapped_label"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# AGGREGATE BY SOURCE DOCUMENT + QUESTION
# ============================================================

group_cols = [
    "source_chunk_id",
    "question",
]

question_stats = (
    df
    .groupby(group_cols)
    .agg(

        # Total number of retrieved entries
        n_retrieved=(
            "mapped_label",
            "size"
        ),

        # Number of CONTRADICTION
        n_CONTRADICTION=(
            "mapped_label",
            lambda x: (
                x == "CONTRADICTION"
            ).sum()
        ),

        # Number of NOT_ENOUGH_INFO
        n_NOT_ENOUGH_INFO=(
            "mapped_label",
            lambda x: (
                x == "NOT_ENOUGH_INFO"
            ).sum()
        ),

        # Labels present in the aggregation
        labels_present=(
            "mapped_label",
            lambda x: sorted(set(x))
        ),

    )
    .reset_index()
)


# ============================================================
# PROPORTIONS
# ============================================================

question_stats["prop_CONTRADICTION"] = (
    question_stats["n_CONTRADICTION"]
    / question_stats["n_retrieved"]
)

question_stats["prop_NOT_ENOUGH_INFO"] = (
    question_stats["n_NOT_ENOUGH_INFO"]
    / question_stats["n_retrieved"]
)


# ============================================================
# BASIC CHECKS
# ============================================================

print("\n" + "=" * 80)
print("AGGREGATED QUESTION + DOCUMENT PAIRS")
print("=" * 80)

print(
    f"Total aggregations: "
    f"{len(question_stats):,}"
)

print(
    f"Unique source documents: "
    f"{question_stats['source_chunk_id'].nunique():,}"
)

print(
    f"Unique question texts: "
    f"{question_stats['question'].nunique():,}"
)


# ============================================================
# CHECK COUNTS
# ============================================================

question_stats["check"] = (
    question_stats["n_CONTRADICTION"]
    + question_stats["n_NOT_ENOUGH_INFO"]
)


# ============================================================
# INVALID RETRIEVAL > 20
# ============================================================

invalid_retrieval = question_stats[
    question_stats["n_retrieved"] > 20
].copy()


print("\n" + "=" * 80)
print("INVALID AGGREGATIONS (> 20 RETRIEVED)")
print("=" * 80)

print(
    f"Number of invalid aggregations: "
    f"{len(invalid_retrieval):,}"
)

if len(invalid_retrieval) > 0:

    print(
        invalid_retrieval[
            [
                "source_chunk_id",
                "question",
                "n_retrieved",
                "n_CONTRADICTION",
                "n_NOT_ENOUGH_INFO",
            ]
        ].to_string(index=False)
    )


# ============================================================
# REMOVE > 20 FROM AGGREGATED ANALYSIS
# ============================================================

question_stats = question_stats[
    question_stats["n_retrieved"] <= 20
].copy()


print(
    f"\nAggregations remaining after filtering: "
    f"{len(question_stats):,}"
)


# ============================================================
# VALIDATE LABEL COUNTS
# ============================================================

invalid_check = (
    question_stats["check"]
    != question_stats["n_retrieved"]
)

print(
    f"Aggregations where C + NEI != n_retrieved: "
    f"{invalid_check.sum():,}"
)


# ============================================================
# GLOBAL COUNTS
# ============================================================

total_c = (
    df["mapped_label"]
    .eq("CONTRADICTION")
    .sum()
)

total_nei = (
    df["mapped_label"]
    .eq("NOT_ENOUGH_INFO")
    .sum()
)


print("\n" + "=" * 80)
print("GLOBAL COUNTS")
print("=" * 80)

print(
    f"Total retrieved entries: "
    f"{len(df):,}"
)

print(
    f"Total CONTRADICTION: "
    f"{total_c:,}"
)

print(
    f"Total NOT_ENOUGH_INFO: "
    f"{total_nei:,}"
)

print(
    f"Percentage CONTRADICTION: "
    f"{total_c / len(df) * 100:.2f}%"
)

print(
    f"Percentage NOT_ENOUGH_INFO: "
    f"{total_nei / len(df) * 100:.2f}%"
)


# ============================================================
# RETRIEVAL SIZE DISTRIBUTION
# ============================================================

retrieval_distribution = (
    question_stats["n_retrieved"]
    .value_counts()
    .sort_index()
    .rename_axis("n_retrieved")
    .reset_index(name="n_aggregations")
)


print("\n" + "=" * 80)
print("NUMBER OF RETRIEVED ENTRIES PER AGGREGATION")
print("=" * 80)

print(
    retrieval_distribution
    .to_string(index=False)
)


# ============================================================
# NEI DISTRIBUTION
# ============================================================

nei_distribution = (
    question_stats["n_NOT_ENOUGH_INFO"]
    .value_counts()
    .sort_index()
    .rename_axis("n_NOT_ENOUGH_INFO")
    .reset_index(name="n_aggregations")
)


print("\n" + "=" * 80)
print("NOT_ENOUGH_INFO PER AGGREGATION")
print("=" * 80)

print(
    nei_distribution
    .to_string(index=False)
)


# ============================================================
# CONTRADICTION DISTRIBUTION
# ============================================================

contradiction_distribution = (
    question_stats["n_CONTRADICTION"]
    .value_counts()
    .sort_index()
    .rename_axis("n_CONTRADICTION")
    .reset_index(name="n_aggregations")
)


print("\n" + "=" * 80)
print("CONTRADICTIONS PER AGGREGATION")
print("=" * 80)

print(
    contradiction_distribution
    .to_string(index=False)
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(
    question_stats[
        [
            "n_retrieved",
            "n_CONTRADICTION",
            "n_NOT_ENOUGH_INFO",
            "prop_CONTRADICTION",
            "prop_NOT_ENOUGH_INFO",
        ]
    ]
    .describe()
    .to_string()
)


# ============================================================
# BARPLOT 1:
# RETRIEVAL SIZE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    retrieval_distribution["n_retrieved"],
    retrieval_distribution["n_aggregations"]
)

plt.xlabel(
    "Number of retrieved entries per question-document pair"
)

plt.ylabel(
    "Number of question-document pairs"
)

plt.title(
    "Distribution of retrieved entries"
)

plt.xlim(9.5, 20.5)

plt.xticks(
    range(10, 21)
)

plt.tight_layout()

retrieval_plot_path = (
    OUTPUT_DIR
    / "retrieval_size_distribution.jpg"
)

plt.savefig(
    retrieval_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# BARPLOT 2:
# NUMBER OF NEI PER AGGREGATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    nei_distribution["n_NOT_ENOUGH_INFO"],
    nei_distribution["n_aggregations"]
)

plt.xlabel(
    "Number of NOT_ENOUGH_INFO entries "
    "per question-document pair"
)

plt.ylabel(
    "Number of question-document pairs"
)

plt.title(
    "Distribution of NOT_ENOUGH_INFO entries"
)

plt.xticks(
    nei_distribution["n_NOT_ENOUGH_INFO"]
)

plt.tight_layout()

nei_plot_path = (
    OUTPUT_DIR
    / "not_enough_info_distribution.jpg"
)

plt.savefig(
    nei_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# BARPLOT 3:
# NUMBER OF CONTRADICTIONS PER AGGREGATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    contradiction_distribution["n_CONTRADICTION"],
    contradiction_distribution["n_aggregations"]
)

plt.xlabel(
    "Number of CONTRADICTION entries "
    "per question-document pair"
)

plt.ylabel(
    "Number of question-document pairs"
)

plt.title(
    "Distribution of CONTRADICTION entries"
)

plt.xticks(
    contradiction_distribution["n_CONTRADICTION"]
)

plt.tight_layout()

contradiction_plot_path = (
    OUTPUT_DIR
    / "contradiction_distribution.jpg"
)

plt.savefig(
    contradiction_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE AGGREGATED TABLES
# ============================================================

question_stats.to_parquet(
    OUTPUT_DIR / "case_3_aggregated.parquet",
    index=False
)

question_stats.to_csv(
    OUTPUT_DIR / "case_3_aggregated.csv",
    index=False
)

retrieval_distribution.to_csv(
    OUTPUT_DIR / "retrieval_size_distribution.csv",
    index=False
)

nei_distribution.to_csv(
    OUTPUT_DIR / "not_enough_info_distribution.csv",
    index=False
)

contradiction_distribution.to_csv(
    OUTPUT_DIR / "contradiction_distribution.csv",
    index=False
)


# ============================================================
# CREATE FINAL CASE 3 PARQUET
# ============================================================
#
# IMPORTANT:
#
# Go back to the ORIGINAL dataframe and keep ONLY
# the original CONTRADICTION rows.
#
# NOT_ENOUGH_INFO rows are discarded.
#
# No original columns or values are modified.
# ============================================================

case_3_final = df.loc[
    df["mapped_label"] == "CONTRADICTION"
].copy()


# ============================================================
# CHECK FINAL DATASET
# ============================================================

print("\n" + "=" * 80)
print("CASE 3 FINAL")
print("=" * 80)

print(
    f"Original CASE 3 rows: "
    f"{len(df):,}"
)

print(
    f"CONTRADICTION rows selected: "
    f"{len(case_3_final):,}"
)

print(
    f"NOT_ENOUGH_INFO rows discarded: "
    f"{total_nei:,}"
)

print(
    f"Unique question-document pairs containing contradiction: "
    f"{case_3_final[['source_chunk_id', 'question']].drop_duplicates().shape[0]:,}"
)


# ============================================================
# VERIFY FINAL LABELS
# ============================================================

print("\nFinal label distribution:")

print(
    case_3_final["mapped_label"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# SAVE FINAL PARQUET
# ============================================================

final_path = (
    OUTPUT_DIR
    / "case_3_final.parquet"
)

case_3_final.to_parquet(
    final_path,
    index=False
)


# ============================================================
# FILES SAVED
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    OUTPUT_DIR / "case_3_aggregated.parquet"
)

print(
    retrieval_plot_path
)

print(
    nei_plot_path
)

print(
    contradiction_plot_path
)

print(
    final_path
)

print("\nProcess finished.")