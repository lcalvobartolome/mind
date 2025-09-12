import pandas as pd

# Paths per model
path_results_mind = {
    "qwen:32b": "data/mind_runs/rosie/results/rosie_results_topic_15.parquet",
}
path_save = "data/ablations/qa/eval_tasks"

paths_ = [
    #"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/GENERATIONS/outs_good_model_tpc11/answers/all_models_eval_v6.parquet",
    #"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/GENERATIONS/outs_good_model_tpc15/answers/all_models_eval_v6.parquet"
    "GENERATIONS/outs_good_model_tpc15/answers/questions_topic_15_qwen:32b_100_seed_1234_results_model30tpc_thr__dynamic.parquet",
    "GENERATIONS/outs_good_model_tpc15/answers/questions_topic_15_gpt-4o-2024-08-06_100_seed_1234_results_model30tpc_thr__dynamic.parquet",
    "GENERATIONS/outs_good_model_tpc15/answers/questions_topic_15_llama3.3:70b_100_seed_1234_results_model30tpc_thr__dynamic.parquet",
    "GENERATIONS/outs_good_model_tpc11/answers/questions_topic_11_qwen:32b_100_seed_1234_results_model30tpc_thr__dynamic.parquet",
    "GENERATIONS/outs_good_model_tpc11/answers/questions_topic_11_llama3.3:70b_100_seed_1234_results_model30tpc_thr__dynamic.parquet",
    "GENERATIONS/outs_good_model_tpc11/answers/questions_topic_11_gpt-4o-2024-08-06_100_seed_1234_results_model30tpc_thr__dynamic.parquet",
]

all_disc = []

# Process original datasets
for path in paths_:
    df = pd.read_parquet(path)
    #print(len(df))
    df = df[~df.question.str.contains("Here are the YES/NO questions generated based on the PASSAGE:")]
    #print(len(df))

    if "gpt" in path:
        df["model"] = "gpt-4o"
    elif "llama" in path: 
        df["model"] = "llama3.3:70b"
    else:
        df["model"] = "qwen:32b"

    # Fix discrepancy values
    df.loc[df['answer_t'].str.contains("cannot answer the question given the context", na=False), 'discrepancy'] = "NOT_ENOUGH_INFO"
    df.loc[df["answer_t"] == "I cannot answer given the context.", ["discrepancy"]] = "NOT_ENOUGH_INFO"
    df["discrepancy"] = df["discrepancy"].str.replace("NO_ DISCREPANCY", "NO_DISCREPANCY")
    df["discrepancy"] = df["discrepancy"].str.replace("CULTURAL_ DISCREPANCY", "CULTURAL_DISCREPANCY")

    # Add source column
    df["source"] = path.split("_tpc")[-1].split("/")[0]

    # Keep only relevant rows
    df_filtered = df[df["discrepancy"].isin(["CONTRADICTION", "CULTURAL_DISCREPANCY"])]

    # Sample 25 rows with NO_DISCREPANCY
    no_discrepancy_sample = df[df["discrepancy"] == "NO_DISCREPANCY"].groupby("model", group_keys=False).sample(n=10, random_state=42)
    print(len(no_discrepancy_sample))

    # Concatenate filtered results
    df_final = pd.concat([df_filtered, no_discrepancy_sample], ignore_index=True)

    # Keep only necessary columns
    df_final = df_final[["question", "answer_t", "answer_s", "discrepancy", "source", "model", "reason"]]

    # Append to all_disc
    all_disc.append(df_final)

# Load FEVER-DPLACE-Q dataset
df_fever = pd.read_csv("FEVER-DPLACE-Q_v2_discp.csv")

# Sample 25 rows per unique label
#fever_samples = df_fever.groupby("label", group_keys=False).apply(lambda x: x.sample(n=min(30, len(x)), random_state=42))

# Rename columns
fever_samples = df_fever.copy()
fever_samples = fever_samples.rename(columns={"answer1": "answer_t", "answer2": "answer_s", "label": "discrepancy"})
fever_samples["model"] = "synthetic"
fever_samples["reason"] = "synthetic"
fever_samples["source"] = "FEVER-DPLACE-Q"
fever_samples["discrepancy"] = fever_samples["discrepancy"].str.replace("SUPPORTS", "NO_DISCREPANCY")
fever_samples["discrepancy"] = fever_samples["discrepancy"].str.replace("REFUTES", "CONTRADICTION")

# Keep only necessary columns
fever_samples = fever_samples[["question", "answer_t", "answer_s", "discrepancy", "source", "model", "reason"]]

# Append FEVER-DPLACE-Q sampled data to all_disc
all_disc.append(fever_samples)

# Create final dataframe
final_df = pd.concat(all_disc, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df["id_discr"] = range(len(final_df))

final_df = final_df[['id_discr','source','question', 'answer_t', 'answer_s', 'discrepancy', 'model', 'reason']]
final_df["label"] = [""] * len(final_df)
final_df = final_df.drop_duplicates(subset="question", keep="first").reset_index(drop=True)

final_df.to_excel("GENERATIONS/discrepancies_v4.xlsx")
final_df.drop(columns=["source", "model", "discrepancy", "reason"]).to_excel("GENERATIONS/discrepancies_v4_users.xlsx", index=False)