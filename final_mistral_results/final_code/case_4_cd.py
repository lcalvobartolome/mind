from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/"
    "final_code/statistics_output/cases/CASE_4_CD.parquet"
)

OUTPUT_DIR = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/"
    "final_db/case_4"
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

        # Total number of retrieved chunks/results
        n_retrieved=(
            "mapped_label",
            "size"
        ),

        # Number of CULTURAL_DISCREPANCY
        n_CULTURAL_DISCREPANCY=(
            "mapped_label",
            lambda x: (
                x == "CULTURAL_DISCREPANCY"
            ).sum()
        ),

        # Number of NOT_ENOUGH_INFO
        n_NOT_ENOUGH_INFO=(
            "mapped_label",
            lambda x: (
                x == "NOT_ENOUGH_INFO"
            ).sum()
        ),

        # Labels appearing in this aggregation
        labels_present=(
            "mapped_label",
            lambda x: sorted(set(x))
        ),

    )
    .reset_index()
)


# ============================================================
# PROPORTION OF CULTURAL_DISCREPANCY
# ============================================================

question_stats["prop_CULTURAL_DISCREPANCY"] = (
    question_stats["n_CULTURAL_DISCREPANCY"]
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
# CHECK THAT COUNTS ADD UP
# ============================================================

question_stats["check"] = (
    question_stats["n_CULTURAL_DISCREPANCY"]
    + question_stats["n_NOT_ENOUGH_INFO"]
)


# ============================================================
# REMOVE AGGREGATIONS WITH MORE THAN 20 RETRIEVED ENTRIES
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
                "n_CULTURAL_DISCREPANCY",
                "n_NOT_ENOUGH_INFO",
            ]
        ].to_string(index=False)
    )


# Remove them from the aggregated statistics
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
    f"Aggregations where CD + NEI != n_retrieved: "
    f"{invalid_check.sum():,}"
)


# ============================================================
# GLOBAL COUNTS
# ============================================================

total_cd = (
    df["mapped_label"]
    .eq("CULTURAL_DISCREPANCY")
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
    f"Total CULTURAL_DISCREPANCY: "
    f"{total_cd:,}"
)

print(
    f"Total NOT_ENOUGH_INFO: "
    f"{total_nei:,}"
)

print(
    f"Percentage CULTURAL_DISCREPANCY: "
    f"{total_cd / len(df) * 100:.2f}%"
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
# CULTURAL_DISCREPANCY COUNT DISTRIBUTION
# ============================================================

cd_distribution = (
    question_stats["n_CULTURAL_DISCREPANCY"]
    .value_counts()
    .sort_index()
    .rename_axis("n_CULTURAL_DISCREPANCY")
    .reset_index(name="n_aggregations")
)


print("\n" + "=" * 80)
print("CULTURAL_DISCREPANCY PER AGGREGATION")
print("=" * 80)

print(
    cd_distribution
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
            "n_CULTURAL_DISCREPANCY",
            "n_NOT_ENOUGH_INFO",
            "prop_CULTURAL_DISCREPANCY",
        ]
    ]
    .describe()
    .to_string()
)


# ============================================================
# BARPLOT 1:
# NUMBER OF RETRIEVED ENTRIES
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
# NUMBER OF CULTURAL_DISCREPANCY ENTRIES
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    cd_distribution["n_CULTURAL_DISCREPANCY"],
    cd_distribution["n_aggregations"]
)

plt.xlabel(
    "Number of CULTURAL_DISCREPANCY entries "
    "per question-document pair"
)

plt.ylabel(
    "Number of question-document pairs"
)

plt.title(
    "Distribution of CULTURAL_DISCREPANCY entries"
)

plt.xticks(
    cd_distribution["n_CULTURAL_DISCREPANCY"]
)

plt.tight_layout()

cd_plot_path = (
    OUTPUT_DIR
    / "cultural_discrepancy_distribution.jpg"
)

plt.savefig(
    cd_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE AGGREGATED TABLES
# ============================================================

question_stats.to_parquet(
    OUTPUT_DIR / "case_4_aggregated.parquet",
    index=False
)

question_stats.to_csv(
    OUTPUT_DIR / "case_4_aggregated.csv",
    index=False
)

retrieval_distribution.to_csv(
    OUTPUT_DIR / "retrieval_size_distribution.csv",
    index=False
)

cd_distribution.to_csv(
    OUTPUT_DIR / "cultural_discrepancy_distribution.csv",
    index=False
)


# ============================================================
# CREATE FINAL CASE 4 PARQUET
# ============================================================
#
# IMPORTANT:
# We return to the ORIGINAL df.
#
# We keep ONLY the original rows classified as
# CULTURAL_DISCREPANCY.
#
# NOT_ENOUGH_INFO rows are discarded.
# No columns or values are modified.
# ============================================================

case_4_final = df.loc[
    df["mapped_label"] == "CULTURAL_DISCREPANCY"
].copy()


# ============================================================
# FINAL DATASET CHECK
# ============================================================

print("\n" + "=" * 80)
print("CASE 4 FINAL")
print("=" * 80)

print(
    f"Original rows in CASE 4: "
    f"{len(df):,}"
)

print(
    f"CULTURAL_DISCREPANCY rows selected: "
    f"{len(case_4_final):,}"
)

print(
    f"Unique question-document pairs: "
    f"{case_4_final[['source_chunk_id', 'question']].drop_duplicates().shape[0]:,}"
)

print(
    f"Rows removed (NOT_ENOUGH_INFO): "
    f"{len(df) - len(case_4_final):,}"
)


print("\nFinal label distribution:")

print(
    case_4_final["mapped_label"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# SAVE FINAL PARQUET
# ============================================================

final_path = (
    OUTPUT_DIR
    / "case_4_final.parquet"
)

case_4_final.to_parquet(
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
    OUTPUT_DIR / "case_4_aggregated.parquet"
)

print(
    retrieval_plot_path
)

print(
    cd_plot_path
)

print(
    final_path
)

print("\nProcess finished.")