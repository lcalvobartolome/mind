import pandas as pd

# Paths per model
path_results_mind = {
    "qwen:32b": "data/mind_runs/rosie/results/rosie_results_topic_15.parquet",
}
path_save = "data/ablations/qa/eval_tasks"
eval_per_model = 50

###############################################################################
print("Preparing QUESTION EVAL task...")
cols_keep = ['question', 'source_chunk']
dimensions_eval = ["Verifiability", "Passage Independence", "Clarity", "Terminology", "Self-Containment", "Naturalness"]

all_results = []
for model_name, path in path_results_mind.items():
    df = pd.read_parquet(path)
    
    # Drop duplicate questions if the column exists
    f = df.drop_duplicates(subset=['question'], keep='first')
    
    df = df[cols_keep]
    
    # Sample up to eval_per_model rows; use replacement if there aren't enough
    n_rows = len(df)
    if n_rows == 0:
        continue
    replace_flag = n_rows < eval_per_model
    df = df.sample(n=eval_per_model, replace=replace_flag, random_state=42)
    
    # Add model name
    df['model'] = model_name
    
    all_results.append(df)

df_all = pd.concat(all_results, ignore_index=True)
# create one column per dimension, initialized to empty string
df_all["row_id"] = range(len(df_all))
for col in dimensions_eval:
    df_all[col] = len(df_all) * [""]

# order the columns so it appears:
# id_row
# source_chunk
# question
# dimensions (Verifiability, Passage Independence, Clarity, Terminology, Self-Containment, Naturalness)
# model
ordered_cols = (
    ["row_id", "source_chunk", "question"]
    + dimensions_eval
    + ["model"]
)
df_all = df_all[ordered_cols]
df_all.to_excel(path_save + "/questions_eval.xlsx", index=False)
print(f"Saved {len(df_all)} rows to {path_save}/questions_eval.xlsx")

###############################################################################
print("Preparing ANSWER EVAL task...")
cols_keep = ['question', 'source_chunk', 'target_chunk', 'a_s', 'a_t']
# Evaluation dimensions
dims_eval = [
    "Faithfulness",
    "Passage Dependence",
    "Passage Reference Avoidance",
    "Structured Response",
    "Language Consistency",
]

all_results = []
for model_name, path in path_results_mind.items():
    df = pd.read_parquet(path)

    # Drop duplicate questions if the column exists
    f = df.drop_duplicates(subset=['question'], keep='first')

    df = df[cols_keep]

    # Sample up to eval_per_model rows; use replacement if there aren't enough
    n_rows = len(df)
    if n_rows == 0:
        continue
    replace_flag = n_rows < eval_per_model
    df = df.sample(n=eval_per_model, replace=replace_flag, random_state=42)

    # Add model name
    df['model'] = model_name

    all_results.append(df)

df_all = pd.concat(all_results, ignore_index=True)

# Add a row id
df_all["row_id"] = range(len(df_all))

for col in dims_eval:
    for corpus in ["s", "t"]:
        df_all[f"{col}_{corpus}"] = len(df_all) * [""]

# order the columns so it appears:
# id_row
# question
# source_chunk
# a_s
# dimensions for source (Faithfulness_s, Passage Dependence_s, Passage Reference Avoidance_s, Structured Response_s, Language Consistency_s)
# target_chunk
# a_t
# dimensions for target (Faithfulness_t, Passage Dependence_t, Passage Reference Avoidance_t, Structured Response_t, Language Consistency_t)
# model
ordered_cols = (
    ["row_id", "question", "source_chunk", "a_s"]
    + [f"{col}_s" for col in dims_eval]
    + ["target_chunk", "a_t"]
    + [f"{col}_t" for col in dims_eval]
    + ["model"]
)
df_all = df_all[ordered_cols]

# Create an identifier column for splitting
split_idx = len(df) // 2
df_part1 = df.iloc[:split_idx]
df_part2 = df.iloc[split_idx:]

# Save the files
df_part2.to_excel(path_save + "/answers_eval_part1.xlsx", index=False)
df_part2.to_excel(path_save + "/answers_eval_part2.xlsx", index=False)

print(f"Split completed: {len(df_part1)} rows in part1, {len(df_part2)} rows in part2.")