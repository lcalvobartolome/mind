import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style
from mind.pipeline.pipeline import MIND
from mind.pipeline.corpus import Chunk


'''
ATENCIÓN: las columans question y anchor_passage cambian el nombre dependiendo de la db que lean.
'''


def process_file(path_annotations: str) -> pd.DataFrame:
    """
    Process a single parquet file of questions and return a DataFrame of results.
    """

    df = pd.read_parquet(path_annotations)    # df expected columns: question_id, question, anchor_passage, comparison_passage, topic

    #modify
    llm_model = "deepseek-r1:8b" #"mistral:7b" "gemma3:4b" "deepseek-r1:8b" "qwen3-vl:8b"

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

        MAX_LEN = 1200
        context_cut = row.anchor_context[:MAX_LEN]

        anchor_chunk = Chunk(id=f"anchor_{id_row}", text=row.anchor_passage, full_doc=context_cut, metadata=None) #_it
        a_s, _ = mind._generate_answer(row.question, anchor_chunk) #_it

        target_chunk = Chunk(id=f"target_{id_row}", text=row.comparison_passage, full_doc=row.comparison_context, metadata=None)
        a_t, discrepancy_label, reason = mind._evaluate_pair(
            question=row.question, #_it
            a_s=a_s,
            source_chunk=anchor_chunk,
            target_chunk=target_chunk,
            topic=row.get("topic", None),
            subquery=None,
            save=False,
        )

        results.append({
            "question_id": row.get("question_id", None),
            "question": row.question, #_it
            "anchor_passage": anchor_chunk.text,
            "anchor_full":anchor_chunk.full_doc,
            "anchor_answer": a_s,
            "comparison_passage": target_chunk.text,
            "comparison_full":target_chunk.full_doc,
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
        default="/export/usuarios01/ivgomez/mind/t_data/inputs/subsets/subset_231_contexto.parquet",   #subsets/subset_15_it.parquet
        help="Path to parquet containing everything.",
    )
    parser.add_argument(
        "--path_save",
        type=str,
        #modify gemma3_4b      qwen3vl_8b     mistral7b        deepseek_r1_8b
        default="/export/usuarios01/ivgomez/mind/t_data/outputs/ground_truth/deepseek", 
        help="Directory where outputs will be saved.",
    )

    args = parser.parse_args()
    path_save = args.path_save
    path_annotations = args.path_annotations

    print(f"Processing {path_annotations}")
    df_results = process_file(path_annotations=path_annotations)
    Path(path_save).mkdir(parents=True, exist_ok=True)

    
    output_file_xlsx = os.path.join(path_save, "ds_es.xlsx") #modify

    df_results.to_excel(output_file_xlsx, index=False)

    output_file_parquet = os.path.join(path_save, "ds_es.parquet")
    df_results.to_parquet(output_file_parquet)
    print("The process is finished yey!!!!")


if __name__ == "__main__":
    main()
