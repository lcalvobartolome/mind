from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/mt5/improved_prompt/results_topic_0_mapped_final_round.parquet"
)

OUTPUT_DIR = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/final_code/statistics_output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COLUMNS
# ============================================================

ANCHOR_COL = "source_chunk_id"
QUESTION_ID_COL = "question_id"
QUESTION_COL = "question"
COMPARISON_COL = "target_chunk_id"
LABEL_COL = "mapped_label"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_PATH)

print("Columns:")
print(df.columns.tolist())

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print(f"Total rows: {len(df):,}")
print(f"Unique question IDs: {df[QUESTION_ID_COL].nunique():,}")
print(f"Unique question texts: {df[QUESTION_COL].nunique():,}")
print(f"Unique anchors: {df[ANCHOR_COL].nunique():,}")
print(f"Unique comparison chunks: {df[COMPARISON_COL].nunique():,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    ANCHOR_COL,
    QUESTION_ID_COL,
    QUESTION_COL,
    COMPARISON_COL,
    LABEL_COL,
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# CHECK FINAL LABELS
# ============================================================

VALID_LABELS = {
    "CONTRADICTION",
    "NO_DISCREPANCY",
    "CULTURAL_DISCREPANCY",
    "NOT_ENOUGH_INFO",
}

print("\n========================================")
print("ORIGINAL MAPPED LABEL DISTRIBUTION")
print("========================================")

print(
    df[LABEL_COL]
    .value_counts(dropna=False)
)


invalid_mask = ~df[LABEL_COL].isin(
    VALID_LABELS
)

n_invalid = invalid_mask.sum()

print(
    f"\nInvalid mapped labels: "
    f"{n_invalid:,}"
)

if n_invalid > 0:

    print(
        df.loc[
            invalid_mask,
            LABEL_COL
        ].value_counts(dropna=False)
    )

    raise ValueError(
        "There are still invalid mapped labels. "
        "Finish label normalization before aggregation."
    )


# ============================================================
# MAP FINAL LABELS TO SHORT NAMES
# ============================================================

label_mapping = {
    "NOT_ENOUGH_INFO": "NEI",
    "NO_DISCREPANCY": "ND",
    "CONTRADICTION": "C",
    "CULTURAL_DISCREPANCY": "CD",
}

# IMPORTANT:
# We do NOT modify mapped_label.
# We create a new column for aggregation.

df["short_label"] = df[LABEL_COL].map(
    label_mapping
)


print("\n========================================")
print("SHORT LABEL DISTRIBUTION")
print("========================================")

print(
    df["short_label"]
    .value_counts(dropna=False)
)


# ============================================================
# CHECK NUMBER OF COMPARISONS PER QUESTION
# ============================================================

comparisons_per_question = (
    df.groupby(
        [
            ANCHOR_COL,
            QUESTION_COL,
        ],
        dropna=False
    )
    .size()
)

print("\n========================================")
print("COMPARISONS PER QUESTION")
print("========================================")

print(
    comparisons_per_question
    .value_counts()
    .sort_index()
)

print(
    f"\nMean comparisons/question: "
    f"{comparisons_per_question.mean():.2f}"
)

print(
    f"Min comparisons/question: "
    f"{comparisons_per_question.min()}"
)

print(
    f"Max comparisons/question: "
    f"{comparisons_per_question.max()}"
)


# ============================================================
# FUNCTION TO CLASSIFY EACH QUESTION
# ============================================================

def classify_case(labels):

    labels = set(labels)

    has_nei = "NEI" in labels
    has_nd = "ND" in labels
    has_c = "C" in labels
    has_cd = "CD" in labels

    # ========================================================
    # CASE 2
    # Only NEI
    # ========================================================

    if labels == {"NEI"}:
        return "CASE_2_ONLY_NEI"

    # ========================================================
    # CASE 1
    # ND + optional NEI
    # No C / CD
    # ========================================================

    if has_nd and not has_c and not has_cd:
        return "CASE_1_ND"

    # ========================================================
    # CASE 3
    # C + optional NEI
    # No ND / CD
    # ========================================================

    if has_c and not has_nd and not has_cd:
        return "CASE_3_C"

    # ========================================================
    # CASE 4
    # CD + optional NEI
    # No ND / C
    # ========================================================

    if has_cd and not has_nd and not has_c:
        return "CASE_4_CD"

    # ========================================================
    # CASE 5
    # C + CD
    # Optional NEI
    # No ND
    # ========================================================

    if has_c and has_cd and not has_nd:
        return "CASE_5_C_CD"

    # ========================================================
    # CASE 6
    # ND + CD
    # Optional NEI
    # No C
    # ========================================================

    if has_nd and has_cd and not has_c:
        return "CASE_6_ND_CD"

    # ========================================================
    # CASE 7
    # ND + C
    # Optional NEI
    # No CD
    # ========================================================

    if has_nd and has_c and not has_cd:
        return "CASE_7_ND_C"

    # ========================================================
    # CASE 8
    # ND + C + CD
    # Optional NEI
    # ========================================================

    if has_nd and has_c and has_cd:
        return "CASE_8_ND_C_CD"

    return "OTHER"


# ============================================================
# AGGREGATE AT QUESTION LEVEL
# ============================================================

question_stats = (
    df.groupby(
        [
            ANCHOR_COL,
            QUESTION_COL,
        ],
        dropna=False
    )
    .agg(

        # Number of rows/comparisons
        n_comparisons=(
            COMPARISON_COL,
            "size"
        ),

        # Number of unique comparison chunks
        n_unique_comparisons=(
            COMPARISON_COL,
            "nunique"
        ),

        # Label counts
        n_NEI=(
            "short_label",
            lambda x: (x == "NEI").sum()
        ),

        n_ND=(
            "short_label",
            lambda x: (x == "ND").sum()
        ),

        n_C=(
            "short_label",
            lambda x: (x == "C").sum()
        ),

        n_CD=(
            "short_label",
            lambda x: (x == "CD").sum()
        ),

        # Labels that appear for this question
        labels_present=(
            "short_label",
            lambda x: sorted(set(x))
        ),
    )
    .reset_index()
)


# ============================================================
# CHECK THAT LABEL COUNTS SUM TO TOTAL
# ============================================================

question_stats["label_sum"] = (
    question_stats["n_NEI"]
    + question_stats["n_ND"]
    + question_stats["n_C"]
    + question_stats["n_CD"]
)

invalid_sum = (
    question_stats["label_sum"]
    != question_stats["n_comparisons"]
)

if invalid_sum.any():

    print("\nWARNING:")
    print(
        f"{invalid_sum.sum():,} questions have "
        f"label counts that do not match "
        f"n_comparisons."
    )

else:

    print(
        "\nAll label counts correctly sum "
        "to n_comparisons."
    )


# ============================================================
# PROPORTIONS PER QUESTION
# ============================================================

for label in ["NEI", "ND", "C", "CD"]:

    question_stats[f"prop_{label}"] = (
        question_stats[f"n_{label}"]
        / question_stats["n_comparisons"]
    )


# ============================================================
# ASSIGN ONE OF THE 8 CASES
# ============================================================

question_stats["case"] = (
    question_stats["labels_present"]
    .apply(classify_case)
)


# ============================================================
# SAVE QUESTION-LEVEL RESULTS
# ============================================================

question_stats.to_parquet(
    OUTPUT_DIR / "question_level_stats0.parquet", #cambiar  
    index=False
)

question_stats.to_csv(
    OUTPUT_DIR / "question_level_stats0.csv", #cambiar
    index=False
)


print("\n========================================")
print("QUESTION LEVEL")
print("========================================")

print(
    f"Total aggregated questions: "
    f"{len(question_stats):,}"
)


# ============================================================
# CASE COUNTS
# ============================================================

case_counts = (
    question_stats["case"]
    .value_counts()
    .rename_axis("case")
    .reset_index(name="n_questions")
)


case_counts["percentage"] = (
    case_counts["n_questions"]
    / len(question_stats)
    * 100
)


# Order cases
case_order = [
    "CASE_1_ND",
    "CASE_2_ONLY_NEI",
    "CASE_3_C",
    "CASE_4_CD",
    "CASE_5_C_CD",
    "CASE_6_ND_CD",
    "CASE_7_ND_C",
    "CASE_8_ND_C_CD",
    "OTHER",
]

case_counts["case"] = pd.Categorical(
    case_counts["case"],
    categories=case_order,
    ordered=True
)

case_counts = (
    case_counts
    .sort_values("case")
    .reset_index(drop=True)
)


print("\n========================================")
print("CASE DISTRIBUTION")
print("========================================")

print(
    case_counts.to_string(
        index=False
    )
)


case_counts.to_csv(
    OUTPUT_DIR / "case_distribution0.csv", #cambiar
    index=False
)


# ============================================================
# AGGREGATE AT ANCHOR / SOURCE CHUNK LEVEL
# ============================================================

anchor_stats = (
    question_stats
    .groupby(
        ANCHOR_COL,
        dropna=False
    )
    .agg(

        n_questions=(
            QUESTION_COL,
            "size"
        ),

        mean_NEI=(
            "n_NEI",
            "mean"
        ),

        mean_ND=(
            "n_ND",
            "mean"
        ),

        mean_C=(
            "n_C",
            "mean"
        ),

        mean_CD=(
            "n_CD",
            "mean"
        ),

        mean_prop_NEI=(
            "prop_NEI",
            "mean"
        ),

        mean_prop_ND=(
            "prop_ND",
            "mean"
        ),

        mean_prop_C=(
            "prop_C",
            "mean"
        ),

        mean_prop_CD=(
            "prop_CD",
            "mean"
        ),
    )
    .reset_index()
)

# ============================================================
# NUMBER OF QUESTIONS OF EACH CASE PER ANCHOR
# ============================================================

case_per_anchor = (
    question_stats
    .pivot_table(
        index=ANCHOR_COL,
        columns="case",
        values=QUESTION_COL,
        aggfunc="count",
        fill_value=0,
        observed=False
    )
    .reset_index()
)

case_per_anchor.columns.name = None



anchor_stats = anchor_stats.merge(
    case_per_anchor,
    on=ANCHOR_COL,
    how="left"
)


# ============================================================
# PROP OF ONLY-NEI QUESTIONS PER ANCHOR
# ============================================================

if "CASE_2_ONLY_NEI" in anchor_stats.columns:

    anchor_stats[
        "prop_questions_only_NEI"
    ] = (
        anchor_stats["CASE_2_ONLY_NEI"]
        / anchor_stats["n_questions"]
    )

else:

    anchor_stats[
        "prop_questions_only_NEI"
    ] = 0.0


# ============================================================
# SAVE ANCHOR LEVEL
# ============================================================

anchor_stats.to_parquet(
    OUTPUT_DIR / "anchor_level_stats0.parquet", #cambiar
    index=False
)

anchor_stats.to_csv(
    OUTPUT_DIR / "anchor_level_stats0.csv", #cambiar
    index=False
)


# ============================================================
# GLOBAL MEANS
# ============================================================

print("\n========================================")
print("MEAN LABEL COUNTS PER QUESTION")
print("========================================")

print(
    question_stats[
        [
            "n_NEI",
            "n_ND",
            "n_C",
            "n_CD",
        ]
    ].mean()
)


print("\n========================================")
print("MEAN LABEL PROPORTIONS PER QUESTION")
print("========================================")

print(
    question_stats[
        [
            "prop_NEI",
            "prop_ND",
            "prop_C",
            "prop_CD",
        ]
    ].mean()
)
# ============================================================
# SPLIT ORIGINAL PARQUET BY CASE
# ============================================================

CASES_OUTPUT_DIR = OUTPUT_DIR / "cases"

CASES_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ADD CASE TO ORIGINAL DATAFRAME
# ============================================================
#
# question_stats has one row per:
# source_chunk_id + question
#
# We merge the assigned case back into the original dataframe
# so that all original comparison rows are preserved.
# ============================================================

df_with_case = df.merge(
    question_stats[
        [
            ANCHOR_COL,
            QUESTION_COL,
            "case",
        ]
    ],
    on=[
        ANCHOR_COL,
        QUESTION_COL,
    ],
    how="left",
    validate="many_to_one"
)


# ============================================================
# CHECK MERGE
# ============================================================

missing_case = df_with_case["case"].isna().sum()

print("\n========================================")
print("CASE MERGE CHECK")
print("========================================")

print(
    f"Original rows: "
    f"{len(df):,}"
)

print(
    f"Rows after case merge: "
    f"{len(df_with_case):,}"
)

print(
    f"Rows without assigned case: "
    f"{missing_case:,}"
)

if len(df_with_case) != len(df):
    raise ValueError(
        "The merge changed the number of rows."
    )

if missing_case > 0:
    raise ValueError(
        "Some rows could not be assigned to a case."
    )


# ============================================================
# CASE NAMES
# ============================================================

CASE_NAMES = [
    "CASE_1_ND",
    "CASE_2_ONLY_NEI",
    "CASE_3_C",
    "CASE_4_CD",
    "CASE_5_C_CD",
    "CASE_6_ND_CD",
    "CASE_7_ND_C",
    "CASE_8_ND_C_CD",
]


# ============================================================
# SAVE ONE PARQUET PER CASE
# ============================================================

print("\n========================================")
print("PARQUET SIZE BY CASE")
print("========================================")

total_rows_cases = 0
total_questions_cases = 0


for case_name in CASE_NAMES:

    # --------------------------------------------------------
    # Filter original rows belonging to this case
    # --------------------------------------------------------

    case_df = df_with_case[
        df_with_case["case"] == case_name
    ].copy()

    # --------------------------------------------------------
    # Number of rows
    # --------------------------------------------------------

    n_rows = len(case_df)

    # --------------------------------------------------------
    # Number of aggregated questions
    # --------------------------------------------------------

    n_questions = (
        case_df[
            [
                ANCHOR_COL,
                QUESTION_COL,
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------------
    # Number of anchors
    # --------------------------------------------------------

    n_anchors = (
        case_df[ANCHOR_COL]
        .nunique()
    )

    # --------------------------------------------------------
    # Save parquet
    # --------------------------------------------------------

    output_file = (
        CASES_OUTPUT_DIR
        / f"{case_name}.parquet"
    )

    case_df.to_parquet(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Accumulate totals
    # --------------------------------------------------------

    total_rows_cases += n_rows
    total_questions_cases += n_questions

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print(
        f"\n{case_name}"
    )

    print(
        f"  Rows/comparisons : {n_rows:,}"
    )

    print(
        f"  Questions        : {n_questions:,}"
    )

    print(
        f"  Anchors          : {n_anchors:,}"
    )

    print(
        f"  File             : {output_file}"
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("\n========================================")
print("CASE SPLIT SUMMARY")
print("========================================")

print(
    f"Original parquet rows : "
    f"{len(df):,}"
)

print(
    f"Sum of case rows      : "
    f"{total_rows_cases:,}"
)

print(
    f"Aggregated questions  : "
    f"{len(question_stats):,}"
)

print(
    f"Sum of case questions : "
    f"{total_questions_cases:,}"
)


# ============================================================
# VALIDATE TOTAL ROWS
# ============================================================

if total_rows_cases == len(df):

    print(
        "\nOK: all original rows are contained "
        "in exactly one case."
    )

else:

    print(
        "\nWARNING: the sum of rows across cases "
        "does not match the original parquet."
    )


# ============================================================
# VALIDATE TOTAL QUESTIONS
# ============================================================

if total_questions_cases == len(question_stats):

    print(
        "OK: all aggregated questions are contained "
        "in exactly one case."
    )

else:

    print(
        "WARNING: the sum of questions across cases "
        "does not match question_stats."
    )

# ============================================================
# GLOBAL ANCHOR INFORMATION
# ============================================================

print("\n========================================")
print("ANCHOR LEVEL")
print("========================================")

print(
    f"Total anchors: "
    f"{len(anchor_stats):,}"
)

print(
    f"Mean questions per anchor: "
    f"{anchor_stats['n_questions'].mean():.2f}"
)

print(
    f"Mean proportion of ONLY-NEI questions "
    f"per anchor: "
    f"{anchor_stats['prop_questions_only_NEI'].mean():.4f}"
)


print("\n========================================")
print("FILES SAVED")
print("========================================")

print(
    OUTPUT_DIR / "question_level_stats0.parquet"
)

print(
    OUTPUT_DIR / "question_level_stats0.csv"
)

print(
    OUTPUT_DIR / "case_distribution0.csv"
)

print(
    OUTPUT_DIR / "anchor_level_stats0.parquet"
)

print(
    OUTPUT_DIR / "anchor_level_stats0.csv"
)

print("\nProcess finished.")