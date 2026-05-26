import pandas as pd
import re

df = pd.read_parquet("/export/usuarios01/ivgomez/mind/tests/data_final/dataset_it_con_resumen_EA3.parquet")

#df["content"] = df["content"].str.replace(r"\[\s*\d+\s*\]", "", regex=True)


df["summary_it5_wiki"] = df["summary_it5_wiki"].str.replace(r'^[\'"]+|[\'"]+$', '', regex=True)

df["summary_it5_fanpage"] = df["summary_it5_fanpage"].str.replace(
    r'^Resume en una sola frase de manera clara y concisa de qué trata el texto:\s*',
    '',
    regex=True
)
df.to_parquet("/export/usuarios01/ivgomez/mind/tests/data_final/dataset_it_con_resumen_EA3_clean.parquet", index=False)
