from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/final_code/statistics_output/cases/CASE_1_ND.parquet"
)

OUTPUT_DIR = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/final_db/case_1"
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

        # Number of NO_DISCREPANCY
        n_NO_DISCREPANCY=(
            "mapped_label",
            lambda x: (x == "NO_DISCREPANCY").sum()
        ),

        # Number of NOT_ENOUGH_INFO
        n_NOT_ENOUGH_INFO=(
            "mapped_label",
            lambda x: (x == "NOT_ENOUGH_INFO").sum()
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
# PROPORTION OF NO_DISCREPANCY
# ============================================================

question_stats["prop_NO_DISCREPANCY"] = (
    question_stats["n_NO_DISCREPANCY"]
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
    question_stats["n_NO_DISCREPANCY"]
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
                "n_NO_DISCREPANCY",
                "n_NOT_ENOUGH_INFO",
            ]
        ].to_string(index=False)
    )


# Remove them from the aggregated dataset
question_stats = question_stats[
    question_stats["n_retrieved"] <= 20
].copy()


print(
    f"\nAggregations remaining after filtering: "
    f"{len(question_stats):,}"
)

invalid_check = (
    question_stats["check"]
    != question_stats["n_retrieved"]
)

print(
    f"Aggregations where ND + NEI != n_retrieved: "
    f"{invalid_check.sum():,}"
)


# ============================================================
# GLOBAL NO_DISCREPANCY COUNTS
# ============================================================

total_nd = (
    df["mapped_label"]
    .eq("NO_DISCREPANCY")
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

print(f"Total retrieved entries: {len(df):,}")
print(f"Total NO_DISCREPANCY: {total_nd:,}")
print(f"Total NOT_ENOUGH_INFO: {total_nei:,}")

print(
    f"Percentage NO_DISCREPANCY: "
    f"{total_nd / len(df) * 100:.2f}%"
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
# NO_DISCREPANCY COUNT DISTRIBUTION
# ============================================================

nd_distribution = (
    question_stats["n_NO_DISCREPANCY"]
    .value_counts()
    .sort_index()
    .rename_axis("n_NO_DISCREPANCY")
    .reset_index(name="n_aggregations")
)


print("\n" + "=" * 80)
print("NO_DISCREPANCY PER AGGREGATION")
print("=" * 80)

print(
    nd_distribution
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
            "n_NO_DISCREPANCY",
            "n_NOT_ENOUGH_INFO",
            "prop_NO_DISCREPANCY",
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
# NUMBER OF NO_DISCREPANCY ENTRIES
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    nd_distribution["n_NO_DISCREPANCY"],
    nd_distribution["n_aggregations"]
)

plt.xlabel(
    "Number of NO_DISCREPANCY entries per question-document pair"
)

plt.ylabel(
    "Number of question-document pairs"
)

plt.title(
    "Distribution of NO_DISCREPANCY entries"
)

plt.xticks(
    nd_distribution["n_NO_DISCREPANCY"]
)

plt.tight_layout()

nd_plot_path = (
    OUTPUT_DIR
    / "no_discrepancy_distribution.jpg"
)

plt.savefig(
    nd_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE TABLES
# ============================================================

question_stats.to_parquet(
    OUTPUT_DIR / "case_1_aggregated.parquet",
    index=False
)

question_stats.to_csv(
    OUTPUT_DIR / "case_1_aggregated.csv",
    index=False
)

retrieval_distribution.to_csv(
    OUTPUT_DIR / "retrieval_size_distribution.csv",
    index=False
)

nd_distribution.to_csv(
    OUTPUT_DIR / "no_discrepancy_distribution.csv",
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    OUTPUT_DIR / "case_1_aggregated.parquet"
)

print(
    retrieval_plot_path
)

print(
    nd_plot_path
)

# ============================================================
# CREATE FINAL CASE 1 PARQUET
# ============================================================
case_1_final = df.loc[
    df["mapped_label"] == "NO_DISCREPANCY"
].copy()


# ============================================================
# CHECK
# ============================================================

print("\n" + "=" * 80)
print("CASE 1 FINAL")
print("=" * 80)

print(
    f"Original rows: "
    f"{len(df):,}"
)

print(
    f"NO_DISCREPANCY rows selected: "
    f"{len(case_1_final):,}"
)

print(
    f"Unique question-document pairs: "
    f"{case_1_final[['source_chunk_id', 'question']].drop_duplicates().shape[0]:,}"
)

print("\nLabel distribution:")

print(
    case_1_final["mapped_label"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

final_path = (
    OUTPUT_DIR
    / "case_1_final.parquet"
)

case_1_final.to_parquet(
    final_path,
    index=False
)

print("\n" + "=" * 80)
print("CASE 1 FINAL SAVED")
print("=" * 80)

print(final_path)

print("\nProcess finished.")