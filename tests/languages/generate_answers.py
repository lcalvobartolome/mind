import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style
from mind.pipeline.pipeline import MIND
from mind.pipeline.corpus import Chunk


def process_file(path_annotations: str) -> pd.DataFrame:
    """
    Process a single parquet file of questions and return a DataFrame of results.
    """

    df = pd.read_parquet(path_annotations)    # df expected columns: question_id, question, anchor_passage, comparison_passage, topic

    #modify
    llm_model = "deepseek-r1:8b"

    mind = MIND(
        llm_model=llm_model,
        do_check_entailement=True,
        config_path="config/config_i.yaml"
    )

    print(f"Initialized MIND with model {llm_model}")
    results = []

    for id_row, row in tqdm(df.iterrows(), total=len(df)):
        if id_row % 100 == 0:
            print(f"Processing row {id_row} with LLM {llm_model}")

        anchor_chunk = Chunk(id=f"anchor_{id_row}", text=row.anchor_passage)
        a_s, _ = mind._generate_answer(row.question, anchor_chunk)

        target_chunk = Chunk(id=f"target_{id_row}", text=row.comparison_passage)
        a_t, discrepancy_label, reason = mind._evaluate_pair(
            question=row.question,
            a_s=a_s,
            source_chunk=anchor_chunk,
            target_chunk=target_chunk,
            topic=row.get("topic", None),
            subquery=None,
            save=False,
        )

        results.append({
            "question_id": row.get("question_id", None),
            "question": row.question,
            "anchor_passage": anchor_chunk.text,
            "anchor_answer": a_s,
            "comparison_passage": target_chunk.text,
            "comparison_answer": a_t,
            "final_label": discrepancy_label,
            "reason": reason,
        })

    df_results = pd.DataFrame(results)
    return df_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_annotations",
        type=str,
        #modify
        default="/export/usuarios01/ivgomez/mind/t_data/inputs/dataset_topic_15_it_es_it.parquet",
        help="Path to parquet containing everything.",
    )
    parser.add_argument(
        "--path_save",
        type=str,
        #modify
        default="/export/usuarios01/ivgomez/mind/t_data/outputs/deepseek_r1_8b",
        help="Directory where outputs will be saved.",
    )

    args = parser.parse_args()
    path_save = args.path_save
    path_annotations = args.path_annotations

    print(f"Processing {path_annotations}")
    df_results = process_file(path_annotations=path_annotations)
    Path(path_save).mkdir(parents=True, exist_ok=True)

    #modify
    output_file_xlsx = os.path.join(path_save, "prompt_it_deepseek_r1_8b_it_es_it.xlsx")


    df_results.to_excel(output_file_xlsx, index=False)

    #output_file_parquet = os.path.join(path_save, "results.parquet")
    #df_results.to_parquet(output_file_parquet)
    print("The process is finished")


if __name__ == "__main__":
    main()
