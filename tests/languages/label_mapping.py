import argparse
import pandas as pd
from tqdm import tqdm
from mind.pipeline.pipeline import MIND
import os

VALID_LABELS = {
    "CONTRADICTION",
    "NO_DISCREPANCY",
    "CULTURAL_DISCREPANCY",
    "NOT_ENOUGH_INFO",
}

LABEL_PROMPT = """
You are given a reasoning explaining the relationship between two answers
to the same question in different languages.

Based ONLY on the reasoning, choose exactly ONE of the following labels:

- CONTRADICTION
- NO_DISCREPANCY
- CULTURAL_DISCREPANCY
- NOT_ENOUGH_INFO

Reasoning:
\"\"\"
{reason}
\"\"\"

Return ONLY the label, nothing else.
"""



def normalize_label(raw_label: str) -> str | None:
    if not raw_label or not isinstance(raw_label, str):
        return None

    label = raw_label.strip().upper()

    if "CONTRADICT" in label:
        return "CONTRADICTION"

    if label == "NO" or "NO_DISCREP" in label:
        return "NO_DISCREPANCY"

    if "CULTURAL" in label:
        return "CULTURAL_DISCREPANCY"

    if "NOT" in label and "INFO" in label:
        return "NOT_ENOUGH_INFO"

    if label in VALID_LABELS:
        return label

    return None



def infer_label_from_reason(mind: MIND, reason: str) -> str:
    prompt = LABEL_PROMPT.format(reason=reason)

    response, _ = mind._prompter.prompt(
        question=prompt,
        dry_run=False
    )

    return normalize_label(response)




def postprocess_parquet(i_path: str, o_path: str, llm_model: str):
    df = pd.read_parquet(i_path)

    assert "final_label" in df.columns
    assert "reason" in df.columns

    mind = MIND(
        llm_model=llm_model,
        do_check_entailement=False,
        config_path="config/config_i.yaml"
    )

    new_labels = []
    raw_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        print(_)
        raw_label = row["final_label"]
        reason = row["reason"]

        label = normalize_label(raw_label)

        MAX_RETRIES = 3

        attempts = 0
        while (label is None or label not in VALID_LABELS) and attempts < MAX_RETRIES:
            label = infer_label_from_reason(mind, reason)
            attempts += 1

        if label not in VALID_LABELS:
            print(f'no valid label in line {_}')
    

        new_labels.append(label)
        raw_labels.append(raw_label)

    df["mapped_label"] = new_labels
    '''
    output_file_xlsx = os.path.join(o_path, "mistral_mapped_en.xlsx") #modify
    df.to_excel(output_file_xlsx, index=False)
    '''
    output_file_parquet = os.path.join(o_path, "ds_mapped_es.parquet")
    df.to_parquet(output_file_parquet)
    print("Process finished")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", default="/export/usuarios01/ivgomez/mind/t_data/outputs/ground_truth/deepseek/ds_es.parquet")
    parser.add_argument("--output_parquet", default="/export/usuarios01/ivgomez/mind/t_data/outputs/ground_truth/deepseek")
    parser.add_argument(
        "--llm_model",
        default="deepseek-r1:8b"   #"deepseek-r1:8b" "mistral:7b" "gemma3:4b"
    )

    args = parser.parse_args()

    postprocess_parquet(
        i_path=args.input_parquet,
        o_path=args.output_parquet,
        llm_model=args.llm_model
    )


if __name__ == "__main__":
    main()
