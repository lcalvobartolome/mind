'''
Importante: pasar el modelo como parámetro, nunca cargarlo desde la función de traducción
'''
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import pipeline

i_path = "t_data/inputs/subsets/subset_231_contexto.parquet"
o_path = "t_data/inputs/subsets/subset312_translated_contexto.parquet"
model_name = "Helsinki-NLP/opus-mt-en-it"

def load_model():
    translator = pipeline(
        "translation",
        model=model_name,
        tokenizer=model_name
    )
    print("translation model loaded")
    return translator


def translate_column(df: pd.DataFrame, translator, text_col: str) -> pd.Series:

    ds = Dataset.from_pandas(df[[text_col]])

    def translate_batch(batch):
        outputs = translator(batch[text_col])
        batch["translated_text"] = [o["translation_text"] for o in outputs]
        return batch

    print(f"translating column '{text_col}'...")
    ds = ds.map(translate_batch, batched=True, batch_size=8) #para pasarle varias entradas y que vaya más rápido.

    return ds.to_pandas()["translated_text"]


def main():
    print("reading")
    df = pd.read_parquet(i_path)

    translator = load_model()

    columnas_a_traducir = ["question", "anchor_passage", "anchor_context_cut"] #falta contexto

    MAX_LEN = 1200

    df["anchor_context_cut"] = df["anchor_context"].str.slice(0, MAX_LEN)

    for col in columnas_a_traducir:
        df[col + "_it"] = translate_column(df, translator, col)

    df.to_parquet(o_path, index=False)

    print("\n process finished")
    

if __name__ == "__main__":
    main()
