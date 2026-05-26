import pandas as pd

import re
df = pd.read_parquet('/export/usuarios01/ivgomez/mind/tests/data_final/dataset_it_con_resumen_EA3.parquet')


def clean_text(text):
    if pd.isna(text):
        return text
    
    # 1. Normalizar comillas tipográficas a simples
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    
    # 2. Eliminar secuencias raras de comillas (2 o más seguidas)
    text = re.sub(r"[\"']{2,}", "", text)
    
    # 3. Eliminar comillas sueltas pegadas a palabras (inicio o final)
    text = re.sub(r"\b['\"]+", "", text)   # inicio palabra
    text = re.sub(r"['\"]+\b", "", text)   # final palabra
    
    # 4. Eliminar comillas aisladas que quedan colgando
    text = re.sub(r"[\"']", "", text)
    
    return text

df["summary_it5_wiki"] = df["summary_it5_wiki"].apply(clean_text)

df.to_parquet("/export/usuarios01/ivgomez/mind/tests/data_final/dataset_it_con_resumen_EA3_clean_deverdad2.parquet", index=False)
