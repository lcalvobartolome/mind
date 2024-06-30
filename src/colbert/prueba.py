import pandas as pd
from colbert import ColBERT, Indexer
import json
import pathlib

# Read parquets with topic information
out_es = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/POLI_FILTERED_AL/rosie_1_20/df_graph_es.parquet")
out_en = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/POLI_FILTERED_AL/rosie_1_20/df_graph_en.parquet")

df_es = pd.read_parquet(out_es)
df_en = pd.read_parquet(out_en)

import json

df_index = df_es[df_es.main_topic==16][['id_preproc', 'text']]

corpus_path = 'path_to_corpus.jsonl'

with open(corpus_path, 'w') as file:
    for _, row in df_index.iterrows():
        doc = {"id": str(row['id_preproc']), "text": row['text']}
        file.write(json.dumps(doc) + '\n')

# Initialize ColBERT model
colbert = ColBERT.from_pretrained('bert-base-uncased')

# Path to the corpus and the index
corpus_path = 'path_to_corpus.jsonl'
index_path = 'index'

# Function to read the corpus
def read_corpus(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)

# Initialize Indexer
indexer = Indexer(colbert, index_path=index_path, overwrite=True)

# Read and index documents
for document in read_corpus(corpus_path):
    indexer.index([(document['id'], document['text'])])