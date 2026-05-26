import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

path_true = "/export/usuarios01/ivgomez/mind/t_data/inputs/subsets/subset_231_contexto.parquet"
path_pred_a = "/export/usuarios01/ivgomez/mind/t_data/outputs/full/mistral/mistral_mapped_es.parquet"
path_pred_b = "/export/usuarios01/ivgomez/mind/t_data/outputs/full/mistral/mistral_mapped_it.parquet"


df_true = pd.read_parquet(path_true)
df_a = pd.read_parquet(path_pred_a)
df_b = pd.read_parquet(path_pred_b)



df = pd.DataFrame({
    "y_true": df_true["final_label"].values,
    "y_pred_a": df_a["mapped_label"].values,
    "y_pred_b": df_b["mapped_label"].values,
})



df["correct_a"] = (df["y_pred_a"] == df["y_true"]).astype(int)
df["correct_b"] = (df["y_pred_b"] == df["y_true"]).astype(int)


n11 = ((df.correct_a == 1) & (df.correct_b == 1)).sum()
n10 = ((df.correct_a == 1) & (df.correct_b == 0)).sum()
n01 = ((df.correct_a == 0) & (df.correct_b == 1)).sum()
n00 = ((df.correct_a == 0) & (df.correct_b == 0)).sum()

table = [[n11, n10],
         [n01, n00]]

print("McNemar contingency table:")
print(table)


result = mcnemar(table, exact=True)
print(f"Statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")



