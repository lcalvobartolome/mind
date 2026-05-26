import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


INPUT_PATH = "/export/usuarios01/ivgomez/mind/tests/data_final/dataset_es_sin_resumen.parquet"
OUTPUT_PATH = "/export/usuarios01/ivgomez/mind/tests/data_final/dataset_es_con_resumen.parquet"

BATCH_SIZE = 8
MAX_INPUT = 512
MAX_OUTPUT = 60


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


mt5_name = "ELiRF/mt5-base-dacsa-es"
mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_name, use_fast=False)
mt5_model = AutoModelForSeq2SeqLM.from_pretrained(mt5_name).to(device)

xlsum_name = "csebuetnlp/mT5_multilingual_XLSum"
xlsum_tokenizer = AutoTokenizer.from_pretrained(xlsum_name)
xlsum_model = AutoModelForSeq2SeqLM.from_pretrained(xlsum_name).to(device)


def resumir_mt5(textos):
    inputs = mt5_tokenizer(
        ["Resume en una sola frase de manera clara y concisa de qué trata el texto: " + t for t in textos],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT
    ).to(device)

    with torch.no_grad():
        outputs = mt5_model.generate(
            **inputs,
            max_length=MAX_OUTPUT,
            num_beams=4,
            early_stopping=True
        )

    return mt5_tokenizer.batch_decode(outputs, skip_special_tokens=True)


def resumir_xlsum(textos):
    inputs = xlsum_tokenizer(
        textos,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT
    ).to(device)

    with torch.no_grad():
        outputs = xlsum_model.generate(
            **inputs,
            max_length=MAX_OUTPUT,
            min_length=10,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            early_stopping=True
        )

    return xlsum_tokenizer.batch_decode(outputs, skip_special_tokens=True)



df = pd.read_parquet(INPUT_PATH)
df["content"] = df["content"].fillna("").astype(str)

res_mt5 = []
res_xlsum = []

total_batches = len(df) // BATCH_SIZE + 1
print(f"Total batches: {total_batches}")

for i in range(0, len(df), BATCH_SIZE):

    if i % (BATCH_SIZE * 50) == 0:
        print(f"Procesando batch {i // BATCH_SIZE}/{total_batches}")

    textos = df["content"].iloc[i:i+BATCH_SIZE].tolist()

    try:
        r1 = resumir_mt5(textos)
    except Exception as e:
        print(f"Error MT5 batch {i}: {e}")
        r1 = [""] * len(textos)

    try:
        r2 = resumir_xlsum(textos)
    except Exception as e:
        print(f"Error XLSum batch {i}: {e}")
        r2 = [""] * len(textos)

    res_mt5.extend(r1)
    res_xlsum.extend(r2)


df["summary_mt5"] = res_mt5
df["summary_xlsum"] = res_xlsum

df.to_parquet(OUTPUT_PATH, index=False)

