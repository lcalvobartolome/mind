from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/"
    "final_code/statistics_output/cases/CASE_6_ND_CD.parquet"
)

OUTPUT_DIR = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/"
    "final_db/case_6"
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

        # Total retrieved entries
        n_retrieved=(
            "mapped_label",
            "size"
        ),

        # NO_DISCREPANCY
        n_NO_DISCREPANCY=(
            "mapped_label",
            lambda x: (
                x == "NO_DISCREPANCY"
            ).sum()
        ),

        # CULTURAL_DISCREPANCY
        n_CULTURAL_DISCREPANCY=(
            "mapped_label",
            lambda x: (
                x == "CULTURAL_DISCREPANCY"
            ).sum()
        ),

        # NOT_ENOUGH_INFO
        n_NOT_ENOUGH_INFO=(
            "mapped_label",
            lambda x: (
                x == "NOT_ENOUGH_INFO"
            ).sum()
        ),

        # Labels present
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

question_stats["prop_NO_DISCREPANCY"] = (
    question_stats["n_NO_DISCREPANCY"]
    / question_stats["n_retrieved"]
)

question_stats["prop_CULTURAL_DISCREPANCY"] = (
    question_stats["n_CULTURAL_DISCREPANCY"]
    / question_stats["n_retrieved"]
)


# ============================================================
# BASIC INFORMATION
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
    question_stats["n_NO_DISCREPANCY"]
    + question_stats["n_CULTURAL_DISCREPANCY"]
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
                "n_NO_DISCREPANCY",
                "n_CULTURAL_DISCREPANCY",
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
    f"Aggregations where ND + CD + NEI != n_retrieved: "
    f"{invalid_check.sum():,}"
)


# ============================================================
# GLOBAL COUNTS
# ============================================================

total_nd = (
    df["mapped_label"]
    .eq("NO_DISCREPANCY")
    .sum()
)

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
    f"Total NO_DISCREPANCY: "
    f"{total_nd:,}"
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
    f"Percentage NO_DISCREPANCY: "
    f"{total_nd / len(df) * 100:.2f}%"
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
# ND DISTRIBUTION
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
# CD DISTRIBUTION
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
            "n_NO_DISCREPANCY",
            "n_CULTURAL_DISCREPANCY",
            "n_NOT_ENOUGH_INFO",
            "prop_NO_DISCREPANCY",
            "prop_CULTURAL_DISCREPANCY",
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
# NO_DISCREPANCY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    nd_distribution["n_NO_DISCREPANCY"],
    nd_distribution["n_aggregations"]
)

plt.xlabel(
    "Number of NO_DISCREPANCY entries "
    "per question-document pair"
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
# BARPLOT 3:
# CULTURAL_DISCREPANCY
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
    OUTPUT_DIR / "case_6_aggregated.parquet",
    index=False
)

question_stats.to_csv(
    OUTPUT_DIR / "case_6_aggregated.csv",
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

cd_distribution.to_csv(
    OUTPUT_DIR / "cultural_discrepancy_distribution.csv",
    index=False
)


# ============================================================
# CREATE FINAL CASE 6 DATASETS
# ============================================================
#
# Split the original CASE 6 into:
#
# 1. NO_DISCREPANCY
# 2. CULTURAL_DISCREPANCY
#
# NEI entries are discarded.
#
# Original rows and columns are preserved.
# ============================================================

case_6_nd_final = df.loc[
    df["mapped_label"] == "NO_DISCREPANCY"
].copy()

case_6_cd_final = df.loc[
    df["mapped_label"] == "CULTURAL_DISCREPANCY"
].copy()


# ============================================================
# CHECK FINAL DATASETS
# ============================================================

print("\n" + "=" * 80)
print("CASE 6 FINAL SPLIT")
print("=" * 80)

print(
    f"Original CASE 6 rows: "
    f"{len(df):,}"
)

print(
    f"NO_DISCREPANCY rows: "
    f"{len(case_6_nd_final):,}"
)

print(
    f"CULTURAL_DISCREPANCY rows: "
    f"{len(case_6_cd_final):,}"
)

print(
    f"NOT_ENOUGH_INFO rows discarded: "
    f"{total_nei:,}"
)


# ============================================================
# UNIQUE AGGREGATIONS IN EACH FINAL DATASET
# ============================================================

print("\nUnique question-document pairs:")

print(
    f"ND: "
    f"{case_6_nd_final[['source_chunk_id', 'question']].drop_duplicates().shape[0]:,}"
)

print(
    f"CD: "
    f"{case_6_cd_final[['source_chunk_id', 'question']].drop_duplicates().shape[0]:,}"
)


# ============================================================
# VERIFY LABELS
# ============================================================

print("\nND final label distribution:")

print(
    case_6_nd_final["mapped_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nCD final label distribution:")

print(
    case_6_cd_final["mapped_label"]
    .value_counts(dropna=False)
    .to_string()
)


plt.figure(figsize=(8, 7))

plt.scatter(
    question_stats["n_NO_DISCREPANCY"],
    question_stats["n_CULTURAL_DISCREPANCY"],
    alpha=0.5
)

max_value = max(
    question_stats["n_NO_DISCREPANCY"].max(),
    question_stats["n_CULTURAL_DISCREPANCY"].max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel("Number of NO_DISCREPANCY entries")
plt.ylabel("Number of CULTURAL_DISCREPANCY entries")

plt.title(
    "NO_DISCREPANCY vs CULTURAL_DISCREPANCY per question-document pair"
)

plt.xticks(range(0, max_value + 1))
plt.yticks(range(0, max_value + 1))

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "nd_vs_cd_scatter.jpg",
    dpi=300
)

plt.close()

# ============================================================
# SAVE FINAL PARQUETS
# ============================================================

nd_final_path = (
    OUTPUT_DIR
    / "case_6_ND_final.parquet"
)

cd_final_path = (
    OUTPUT_DIR
    / "case_6_CD_final.parquet"
)

case_6_nd_final.to_parquet(
    nd_final_path,
    index=False
)

case_6_cd_final.to_parquet(
    cd_final_path,
    index=False
)


# ============================================================
# FILES SAVED
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    OUTPUT_DIR / "case_6_aggregated.parquet"
)

print(
    retrieval_plot_path
)

print(
    nd_plot_path
)

print(
    cd_plot_path
)

print(
    nd_final_path
)

print(
    cd_final_path
)

print("\nProcess finished.")