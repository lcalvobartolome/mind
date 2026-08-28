import argparse
import os
import pandas as pd


# ============================================================
# VALID LABELS
# ============================================================

VALID_LABELS = {
    "CONTRADICTION",
    "NO_DISCREPANCY",
    "CULTURAL_DISCREPANCY",
    "NOT_ENOUGH_INFO",
}


def final_normalize(row):
    """
    Final normalization round.

    Rules:
    1. If mapped_label is already valid -> keep it unchanged.
    2. If the unresolved original label contains "while"
       -> CONTRADICTION.
    3. Any remaining unresolved label -> NOT_ENOUGH_INFO.
    """

    mapped_label = row["mapped_label"]

    # ========================================================
    # 1. ALREADY VALID -> KEEP
    # ========================================================

    if mapped_label in VALID_LABELS:
        return mapped_label

    # ========================================================
    # 2. GET ORIGINAL LABEL
    # ========================================================

    raw_label = row["label"]

    if not pd.isna(raw_label) and isinstance(raw_label, str):

        text = raw_label.strip().lower()

        # ====================================================
        # WHILE -> CONTRADICTION
        # ====================================================

        if "while" in text:
            return "CONTRADICTION"

    # ========================================================
    # 3. EVERYTHING ELSE -> NOT_ENOUGH_INFO
    # ========================================================

    return "NOT_ENOUGH_INFO"


def normalize_parquet(i_path: str, o_path: str):

    # ========================================================
    # LOAD
    # ========================================================

    df = pd.read_parquet(i_path)

    print("Columns:")
    print(df.columns.tolist())

    assert "label" in df.columns, (
        "Column 'label' not found in parquet."
    )

    assert "mapped_label" in df.columns, (
        "Column 'mapped_label' not found in parquet."
    )

    print(f"\nTotal rows: {len(df):,}")

    # ========================================================
    # STATUS BEFORE FINAL ROUND
    # ========================================================

    valid_before_mask = df["mapped_label"].isin(
        VALID_LABELS
    )

    unresolved_before_mask = ~valid_before_mask

    n_valid_before = valid_before_mask.sum()
    n_unresolved_before = unresolved_before_mask.sum()

    print("\n========================================")
    print("BEFORE FINAL ROUND")
    print("========================================")

    print(
        f"Already valid labels: "
        f"{n_valid_before:,}"
    )

    print(
        f"Unresolved labels: "
        f"{n_unresolved_before:,}"
    )

    # ========================================================
    # COUNT "WHILE" BEFORE MODIFYING
    # ========================================================

    unresolved_text = (
        df.loc[unresolved_before_mask, "label"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    while_mask = unresolved_text.str.contains(
        "while",
        regex=False
    )

    n_while = while_mask.sum()

    print("\n========================================")
    print("FINAL RULES")
    print("========================================")

    print(
        f'Unresolved labels containing "while": '
        f"{n_while:,}"
    )

    print(
        f"Remaining labels that will become "
        f"NOT_ENOUGH_INFO: "
        f"{n_unresolved_before - n_while:,}"
    )

    # ========================================================
    # SAVE OLD LABEL FOR COMPARISON
    # ========================================================

    old_mapped_label = df["mapped_label"].copy()

    # ========================================================
    # APPLY FINAL NORMALIZATION
    # ========================================================

    df["mapped_label"] = df.apply(
        final_normalize,
        axis=1
    )

    # ========================================================
    # IDENTIFY CHANGES
    # ========================================================

    changed_mask = (
        ~old_mapped_label.isin(VALID_LABELS)
    )

    contradiction_recovered_mask = (
        changed_mask
        & (df["mapped_label"] == "CONTRADICTION")
    )

    nei_assigned_mask = (
        changed_mask
        & (df["mapped_label"] == "NOT_ENOUGH_INFO")
    )

    n_contradiction = contradiction_recovered_mask.sum()
    n_nei = nei_assigned_mask.sum()

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n========================================")
    print("FINAL ROUND RESULTS")
    print("========================================")

    print(
        f'Assigned to CONTRADICTION using "while": '
        f"{n_contradiction:,}"
    )

    print(
        f"Assigned to NOT_ENOUGH_INFO: "
        f"{n_nei:,}"
    )

    # ========================================================
    # FINAL DISTRIBUTION
    # ========================================================

    print("\n========================================")
    print("FINAL LABEL DISTRIBUTION")
    print("========================================")

    final_distribution = (
        df["mapped_label"]
        .value_counts(dropna=False)
    )

    print(final_distribution)

    # ========================================================
    # CHECK THAT EVERYTHING IS VALID
    # ========================================================

    invalid_mask = ~df["mapped_label"].isin(
        VALID_LABELS
    )

    n_invalid = invalid_mask.sum()

    print("\n========================================")
    print("FINAL VALIDATION")
    print("========================================")

    print(
        f"Remaining invalid labels: "
        f"{n_invalid:,}"
    )

    if n_invalid == 0:
        print(
            "All rows have a valid mapped label."
        )
    else:
        print(
            "WARNING: Some rows still have invalid labels."
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.dirname(o_path)

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # SAVE FINAL REPORT
    # ========================================================

    report_path = os.path.join(
        output_dir,
        "label_mapping_final_report4.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("FINAL LABEL MAPPING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Total rows: {len(df):,}\n"
        )

        f.write(
            f"Already valid before final round: "
            f"{n_valid_before:,}\n"
        )

        f.write(
            f"Unresolved before final round: "
            f"{n_unresolved_before:,}\n"
        )

        f.write("\n")

        f.write(
            f'Assigned to CONTRADICTION '
            f'using "while": '
            f"{n_contradiction:,}\n"
        )

        f.write(
            f"Assigned to NOT_ENOUGH_INFO: "
            f"{n_nei:,}\n"
        )

        f.write(
            f"Remaining invalid labels: "
            f"{n_invalid:,}\n"
        )

        # ----------------------------------------------------
        # Final distribution
        # ----------------------------------------------------

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("FINAL LABEL DISTRIBUTION\n")
        f.write("=" * 80 + "\n\n")

        for label, count in final_distribution.items():

            f.write(
                f"{label}: {count:,}\n"
            )

        # ----------------------------------------------------
        # WHILE EXAMPLES
        # ----------------------------------------------------

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(
            'LABELS ASSIGNED TO CONTRADICTION '
            'BECAUSE THEY CONTAIN "while"\n'
        )
        f.write("=" * 80 + "\n\n")

        while_examples = df.loc[
            contradiction_recovered_mask,
            "label"
        ]

        while_counts = (
            while_examples
            .value_counts(dropna=False)
        )

        for label, count in while_counts.items():

            f.write(
                f"COUNT: {count}\n"
            )

            f.write(
                f"LABEL: {repr(label)}\n"
            )

            f.write(
                "-" * 80 + "\n"
            )

    print(
        f"\nFinal report saved to: "
        f"{report_path}"
    )

    # ========================================================
    # SAVE FINAL PARQUET
    # ========================================================

    df.to_parquet(
        o_path,
        index=False
    )

    print(
        f"\nFinal parquet saved to: "
        f"{o_path}"
    )

    print("Process finished.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_parquet",
        default=(
            "/export/usuarios01/ivgomez/mind/final_mistral_results/mt5/improved_prompt/results_topic_4_mapped_round2.parquet"
        )
    )

    parser.add_argument(
        "--output_parquet",
        default=(
            "/export/usuarios01/ivgomez/mind/final_mistral_results/mt5/improved_prompt/results_topic_4_mapped_final_round.parquet"
        )
    )

    args = parser.parse_args()

    normalize_parquet(
        i_path=args.input_parquet,
        o_path=args.output_parquet
    )


if __name__ == "__main__":
    main()