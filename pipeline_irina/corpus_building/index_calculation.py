import pandas as pd

# ==========================
# CONFIG
# ==========================

INPUT_PARQUET = "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_it_translated.parquet"
OUTPUT_PARQUET = "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_it_translated_coded.parquet"

LANG = "IT"   # ES, EN, DE, IT...

# ==========================
# LOAD
# ==========================

df = pd.read_parquet(INPUT_PARQUET)

# ==========================
# FIX id_preproc
# ==========================

def adapt_id(id_preproc: str, lang: str) -> str:
    if id_preproc.startswith("T_"):
        return f"T_{lang}_{id_preproc[2:]}"
    else:
        return f"{lang}_{id_preproc}"

df["id_preproc"] = df["id_preproc"].astype(str).apply(
    lambda x: adapt_id(x, LANG)
)

# ==========================
# SAVE
# ==========================

df.to_parquet(OUTPUT_PARQUET, index=False)

print("Done.")
print(df["id_preproc"].head())
print(df["id_preproc"].tail())