import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# CONFIG
# =========================

INPUT_PATH = "/export/usuarios01/ivgomez/mind/tests/data_final/dataset_it_sin_resumen.parquet"
OUTPUT_PATH = "/export/usuarios01/ivgomez/mind/tests/data_final/dataset_it_con_resumen.parquet"

BATCH_SIZE = 8
MAX_INPUT = 512
MAX_OUTPUT = 60

# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# =========================
# MODELOS
# =========================

print("Cargando modelos...")

# 🔹 Modelo 1: IT5 fanpage (abstractivo con prompt)
model1_name = "ARTeLab/it5-summarization-fanpage"
tokenizer1 = T5Tokenizer.from_pretrained(model1_name)
model1 = T5ForConditionalGeneration.from_pretrained(model1_name).to(device)

# 🔹 Modelo 2: IT5 wiki (generativo sin prompt)
model2_name = "it5/it5-large-wiki-summarization"
tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
model2 = AutoModelForSeq2SeqLM.from_pretrained(model2_name).to(device)

print("Modelos cargados")

# =========================
# FUNCIONES
# =========================

def resumir_it5_fanpage(textos):
    inputs = tokenizer1(
        ["Riassumi in una sola frase in modo chiaro e conciso di cosa tratta il testo: " + t for t in textos],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT
    ).to(device)

    with torch.no_grad():
        outputs = model1.generate(
            **inputs,
            max_length=MAX_OUTPUT,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer1.batch_decode(outputs, skip_special_tokens=True)


def resumir_it5_wiki(textos):
    inputs = tokenizer2(
        textos,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT
    ).to(device)

    with torch.no_grad():
        outputs = model2.generate(
            **inputs,
            max_length=MAX_OUTPUT,
            min_length=10,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            early_stopping=True
        )

    return tokenizer2.batch_decode(outputs, skip_special_tokens=True)

# =========================
# CARGA DATOS
# =========================

print("Leyendo datos...")
df = pd.read_parquet(INPUT_PATH)
df["content"] = df["content"].fillna("").astype(str)

# =========================
# PROCESAMIENTO
# =========================

res_fanpage = []
res_wiki = []

total_batches = len(df) // BATCH_SIZE + 1
print(f"Total batches: {total_batches}")

for i in range(0, len(df), BATCH_SIZE):

    if i % (BATCH_SIZE * 50) == 0:
        print(f"Procesando batch {i // BATCH_SIZE}/{total_batches}")

    textos = df["content"].iloc[i:i+BATCH_SIZE].tolist()

    try:
        r1 = resumir_it5_fanpage(textos)
    except Exception as e:
        print(f"Error FANPAGE batch {i}: {e}")
        r1 = [""] * len(textos)

    try:
        r2 = resumir_it5_wiki(textos)
    except Exception as e:
        print(f"Error WIKI batch {i}: {e}")
        r2 = [""] * len(textos)

    res_fanpage.extend(r1)
    res_wiki.extend(r2)

# =========================
# GUARDAR
# =========================

df["summary_it5_fanpage"] = res_fanpage
df["summary_it5_wiki"] = res_wiki

print("Guardando resultado...")
df.to_parquet(OUTPUT_PATH, index=False)

