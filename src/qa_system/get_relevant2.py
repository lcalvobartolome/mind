import pandas as pd # type: ignore
from prompter import Prompter
import os

template_path = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/templates/test_relevance.txt"
with open(template_path, 'r') as file:
    template = file.read()
    
path_already_annotated = "GENERATIONS/outs_good_model_tpc11/relevant_check"

paths_ = os.listdir(path_already_annotated)
paths_ = [path_already_annotated + "/" + path for path in paths_]
for path_annotations in paths_:
    annotations = pd.read_parquet(path_annotations)
    for llm_model in ["qwen:32b", "llama3.3:70b"]: #["qwen2.5:7b-instruct"]: #["llama3:70b-instruct", "llama3.1:8b-instruct-q8_0"]:
        
        prompter = Prompter(
            model_type=llm_model,
        )
        print(f"Processing for LLM {llm_model}")
        annotations[f"relevance_{llm_model}"] = len(annotations) * [None]
        for id_row, row in annotations.iterrows():
            question = template.format(passage=row.all_results_content, question=row.question)
            response, _ = prompter.prompt(
                #system_prompt_template_path=system_template_path,
                question=question
            )
            relevance = 1 if "yes" in response.lower() else 0
            annotations.loc[id_row, f"relevance_{llm_model}"] = relevance
        
            if id_row % 100 == 0: 
                print(f"Processed {len(annotations) - id_row} / {len(annotations)} using LLM {llm_model}")
    
    try:        
        #annotations.to_parquet(path_annotations.replace("/relevant/", "/relevant_check/"), index=False)
        annotations.to_parquet(path_annotations.replace("relevant.parquet", "relevant_all_added.parquet"), index=False)
    except Exception as e:
        print(f"Error saving annotations for {path_annotations}. Error: {e}")
        import pdb; pdb.set_trace()