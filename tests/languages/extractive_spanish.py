import pandas as pd
import torch
import numpy as np
import nltk
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn

nltk.download('punkt')
nltk.download('punkt_tab')
print('leyendo')
INPUT_PATH = "/export/usuarios01/ivgomez/mind/tests/data_final/dataset_es_con_resumen_EA.parquet"
OUTPUT_PATH = "/export/usuarios01/ivgomez/mind/tests/data_final/dataset_es_con_resumen_EA3_1.parquet"

BATCH_SIZE = 8
MAX_INPUT = 512


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


print('cargando modelo')
bert_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(bert_name)
model = AutoModel.from_pretrained(bert_name).to(device)
model.eval()


def get_sentence_embedding(sentences):
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()

def resumir_classifier_batch(textos):
    summaries = []

    for text in textos:
        try:
            if not text.strip():
                summaries.append("")
                continue

            sentences = nltk.sent_tokenize(text, language="spanish")

            if len(sentences) == 1:
                summaries.append(sentences[0])
                continue

            sent_embeddings = get_sentence_embedding(sentences)

            scores = np.linalg.norm(sent_embeddings, axis=1)

            best_idx = int(np.argmax(scores))
            summaries.append(sentences[best_idx])

        except Exception as e:
            print(f"Error classifier: {e}")
            summaries.append("")

    return summaries

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        scores = self.fc(out).squeeze(-1)
        return scores

lstm_model = SimpleLSTM(768).to(device)
lstm_model.eval()

def resumir_lstm_batch(textos):
    summaries = []

    for text in textos:
        try:
            if not text.strip():
                summaries.append("")
                continue

            sentences = nltk.sent_tokenize(text, language="spanish")

            if len(sentences) == 1:
                summaries.append(sentences[0])
                continue

            sent_embeddings = get_sentence_embedding(sentences)

            emb_tensor = torch.tensor(sent_embeddings).unsqueeze(0).to(device)

            with torch.no_grad():
                scores = lstm_model(emb_tensor).cpu().numpy().flatten()

            best_idx = int(np.argmax(scores))
            summaries.append(sentences[best_idx])

        except Exception as e:
            print(f"Error lstm: {e}")
            summaries.append("")

    return summaries

def resumir_extractivo_batch(textos):
    summaries = []

    for text in textos:
        try:
            if not text.strip():
                summaries.append("")
                continue

            lang = "spanish"
            sentences = nltk.sent_tokenize(text, language=lang)

            if len(sentences) == 1:
                summaries.append(sentences[0])
                continue

            sent_embeddings = get_sentence_embedding(sentences)

            doc_embedding = np.mean(sent_embeddings, axis=0).reshape(1, -1)

            scores = [
                cosine_similarity([emb], doc_embedding)[0][0] / (len(sentences[i].split()) + 1)
                for i, emb in enumerate(sent_embeddings)
            ]
            best_idx = int(np.argmax(scores))
            best_sentence = sentences[best_idx]

            summaries.append(best_sentence)

        except Exception as e:
            print(f"Error en texto: {e}")
            summaries.append("")

    return summaries



df = pd.read_parquet(INPUT_PATH)
df["content"] = df["content"].fillna("").astype(str)

#res_bert = []
res_classifier = []
res_lstm = []

total_batches = len(df) // BATCH_SIZE + 1
print(f"Total batches: {total_batches}")

for i in range(0, len(df), BATCH_SIZE):

    if i % (BATCH_SIZE * 50) == 0:
        print(f"Procesando batch {i // BATCH_SIZE}/{total_batches}")

    textos = df["content"].iloc[i:i+BATCH_SIZE].tolist()

    try:
        #r = resumir_extractivo_batch(textos)
        r_clf = resumir_classifier_batch(textos)
        r_lstm = resumir_lstm_batch(textos)
    except Exception as e:
        print(f"Error batch {i}: {e}")
        #r = [""] * len(textos)
        r_clf = [""] * len(textos)
        r_lstm = [""] * len(textos)

    #res_bert.extend(r)
    res_classifier.extend(r_clf)
    res_lstm.extend(r_lstm)



#df["summary_bert_extractive"] = res_bert
df["summary_classifier"] = res_classifier
df["summary_lstm"] = res_lstm

print('guardando...')
df.to_parquet(OUTPUT_PATH, index=False)

print('listo')