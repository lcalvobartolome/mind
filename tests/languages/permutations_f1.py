from scipy.stats import permutation_test
from sklearn.metrics import f1_score
import pandas as pd

path_true = "/export/usuarios01/ivgomez/mind/t_data/inputs/subsets/subset_231_contexto.parquet"
path_pred_a = "/export/usuarios01/ivgomez/mind/t_data/outputs/full/mistral/mistral_mapped_it.parquet"
path_pred_b = "/export/usuarios01/ivgomez/mind/t_data/outputs/full/mistral/mistral_mapped_it_verdad.parquet"

df_true = pd.read_parquet(path_true)
df_a = pd.read_parquet(path_pred_a)
df_b = pd.read_parquet(path_pred_b)

y_true = df_true["final_label"].values
y_pred_a = df_a["mapped_label"].values
y_pred_b = df_b["mapped_label"].values

# 🔧 FIX: eliminar None
mask = (
    pd.notna(y_true) &
    pd.notna(y_pred_a) &
    pd.notna(y_pred_b)
)

y_true = y_true[mask]
y_pred_a = y_pred_a[mask]
y_pred_b = y_pred_b[mask]

def macro_f1_diff(preds1, preds2):
    f1_1 = f1_score(y_true, preds1, average="macro")
    f1_2 = f1_score(y_true, preds2, average="macro")
    return f1_1 - f1_2

result = permutation_test(
    data=(y_pred_a, y_pred_b),
    statistic=macro_f1_diff,
    permutation_type="samples",
    alternative="two-sided",
    n_resamples=10000,
    random_state=42
)

f1_original_a = f1_score(y_true, y_pred_a, average="macro")
f1_original_b = f1_score(y_true, y_pred_b, average="macro")

print("diff Macro-F1:", result.statistic)
print("p-value:", result.pvalue)
print(f1_original_a)
print(f1_original_b)
