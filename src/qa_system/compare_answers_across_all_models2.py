import os
import pathlib
import pandas as pd
import re
from prompter import Prompter
from tqdm import tqdm
import ast

path_relevant = "GENERATIONS/outs_good_model_tpc15/relevant"
path_save = "GENERATIONS/outs_good_model_tpc15/answers"
METHOD_EVAL = "results_3_weighted" #"results_4_unweighted"
top_k = 5
nr_eval = 100

######################
# PATHS TO TEMPLATES #
######################
# 3. ANSWER GENERATION
_3_INSTRUCTIONS_PATH = "templates/question_answering.txt"
_4_INSTRUCTIONS_PATH = "templates/discrepancy_detection.txt"
RELEVANCE_PROMPT = "templates/test_relevance.txt"

def extend_to_full_sentence(
    text: str,
    num_words: int
) -> str:
    """Truncate text to a certain number of words and extend to the end of the sentence so it's not cut off.
    """
    text_in_words = text.split()
    truncated_text = " ".join(text_in_words[:num_words])
    
    # Check if there's a period after the truncated text
    remaining_text = " ".join(text_in_words[num_words:])
    period_index = remaining_text.find(".")
    
    # If period, extend the truncated text to the end of the sentence
    if period_index != -1:
        extended_text = f"{truncated_text} {remaining_text[:period_index + 1]}"
    else:
        extended_text = truncated_text
    
    # Clean up screwed up punctuations        
    extended_text = re.sub(r'\s([?.!,"])', r'\1', extended_text)
    
    return extended_text

PATH_SOURCE = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet")
raw = pd.read_parquet(PATH_SOURCE)

results_all = []
paths_ = os.listdir(path_relevant)
paths_ = [path for path in paths_ if path.endswith("thr__dynamic.parquet") and ("llama" in path)]#or "llama" in path or "llama" in path #"qwen" in path or 
keep = []#or "llama" in path or "gpt" in path  or "gpt" in path
for path_queries in paths_:
    df = pd.read_parquet(path_relevant + "/" + path_queries)
    df = df.drop_duplicates(subset=['question'], keep='first')#.head(nr_eval)

    if "qwen" in path_queries:
        llm_model = "qwen:32b"
    elif "llama" in path_queries:
        llm_model = "llama3.3:70b"
    elif "gpt" in path_queries:
        llm_model = "gpt-4o-2024-08-06"
                
    prompter = Prompter(
        model_type=llm_model,
        ollama_host="http://kumo01.tsc.uc3m.es:11434"
    )
    
    results = []
    for id_row, row in tqdm(df.iterrows(), total=len(df)):
        
        if id_row % 100 == 0:
            print(f"Processing row {id_row} with LLM {llm_model}")
        
        try:
            results_4_unweighted = row[METHOD_EVAL]
            flattened_list = [{'doc_id': entry['doc_id'], 'score': entry['score']} for subarray in results_4_unweighted for entry in subarray]
            top_docs = [el["doc_id"] for el in flattened_list][:top_k]
            
            for top_doc in top_docs:
            
                # -----------------------------------------------------------------#
                # 3. ANSWER GENERATION
                #------------------------------------------------------------------#
                with open(_3_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
                
                ######################################
                # GENERATE ANSWER IN SOURCE LANGUAGE 
                ######################################
                passage_s = row.passage
                full_doc_s = row.full_doc
                
                formatted_template = template.format(question=row.question, passage=passage_s,full_document=(extend_to_full_sentence(full_doc_s, 100)+ " [...]"))
                
                answer_s, _ = prompter.prompt(question=formatted_template)
                print("Answer S:", answer_s)
                
                ######################################
                # GENERATE ANSWER IN TARGET LANGUAGE #
                ######################################
                passage_t = raw[raw.doc_id == top_doc].text.iloc[0]
                full_doc_t = raw[raw.doc_id == top_doc].full_doc.iloc[0]
                
                ##############################################
                # CHECK RELEVANCE OF PASSAGE TO THE QUESTION #
                ##############################################
                with open(RELEVANCE_PROMPT, 'r') as file: template = file.read()
                formatted_template = template.format(passage=passage_t, question=row.question)
                
                response, _ = prompter.prompt(question=formatted_template)
                relevance = 1 if "yes" in response.lower() else 0
                
                if relevance == 0:
                    answer_t = "I cannot answer the question given the context."
                else:
                    with open(_3_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
                
                    formatted_template = template.format(question=row.question, passage=passage_t,full_document=(extend_to_full_sentence(full_doc_t, 100)+ " [...]"))
                    answer_t, _ = prompter.prompt(question=formatted_template)
                
                print("Answer T:", answer_t)
                
                #import pdb; pdb.set_trace()
                
                #import pdb; pdb.set_trace()
                
                if "cannot answer the question given the context" not in answer_t:
                    #--------------------------------------------------------------#
                    # 4. DISCREPANCY DETECTION
                    # ---------------------------------------------------------------#
                    with open(_4_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
                    
                    question = template.format(question=row.question, answer_1=answer_s, answer_2=answer_t)
                    
                    discrepancy, _ = prompter.prompt(question=question)
                    
                    label, reason = None, None
                    lines = discrepancy.splitlines()
                    for line in lines:
                        if line.startswith("DISCREPANCY_TYPE:"):
                            label = line.split("DISCREPANCY_TYPE:")[1].strip()
                        elif line.startswith("REASON:"):
                            reason = line.split("REASON:")[1].strip()
                    
            
                    if label is None or reason is None:
                        try:
                            discrepancy_split = discrepancy.split("\n")
                            reason = discrepancy_split[0].strip("\n").strip("REASON:").strip()
                            label = discrepancy_split[1].strip("\n").strip("DISCREPANCY_TYPE:").strip()
                        except:
                            label = discrepancy
                            reason = ""
                    print("Discrepancy:", label)
                    
                else:
                    if answer_t == "I cannot answer as the passage contains personal opinions.":
                        reason = "I cannot answer as the passage contains personal opinions."
                        label = "NOT_ENOUGH_INFO"
                    else:
                        reason = "I cannot answer given the context."
                        label = "NOT_ENOUGH_INFO"
                    
                    
                results.append({
                    "question_id": row.question_id,
                    "doc_id": top_doc,
                    "question": row.question,
                    "passage_s": passage_s,
                    "answer_s": answer_s,
                    "passage_t": passage_t,
                    "answer_t": answer_t,
                    "discrepancy": label,
                    "reason": reason
                })
        except Exception as e:
            print(f"Error with question {row.question_id}: {e}")
            continue
            
    df_results = pd.DataFrame(results)
    # save results for each model
    df_results.to_parquet(f"{path_save}/{path_queries}")
    
    # append to all results with the model name
    df_results["model"] = llm_model
    results_all.append(df_results)
    
    # save intermediate results
    df_results.to_parquet(f"{path_save}/{METHOD_EVAL}.parquet")

df_all = pd.concat(results_all)
df_all.to_parquet(f"{path_save}/all_models_eval_v6.parquet")
