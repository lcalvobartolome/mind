import pandas as pd

df = pd.read_parquet('/export/usuarios01/ivgomez/mind/tests/data_final/dataset_es_con_resumen_EA3_1.parquet')

df["summary_mt5"] = df["summary_mt5"].str.replace(
    "Resume en una sola frase de manera clara y concisa de qué trata el texto:",
    "",
    regex=False
)

df.to_parquet("/export/usuarios01/ivgomez/mind/tests/data_final/dataset_es_con_resumen_final_clean.parquet", index=False)
