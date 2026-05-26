import sys
sys.path.append("/export/usuarios01/ivgomez/mind/src")

from pathlib import Path
import pandas as pd
from mind.pipeline.corpus import Corpus
from mind.pipeline.pipeline import MIND


DATA_PATH = "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_subes.parquet"
TEXT_COL = "content"
ID_COL = "id_preproc"
FULL_DOC_COL = "full_doc"


corpus = Corpus.from_parquet_and_thetas(
    path_parquet=DATA_PATH,
    id_col=ID_COL,
    passage_col=TEXT_COL,
    full_doc_col=FULL_DOC_COL,
    language_filter="ES",  
    load_thetas=False,
    config_path="config/config_i.yaml"  
)
llm_model = "mistral:7b" #"mistral:7b" "gemma3:4b" "deepseek-r1:8b" "qwen3-vl:8b"

mind = MIND(
    llm_model=llm_model,  
    source_corpus=corpus,
    multilingual=False,
    dry_run=False,
    config_path="config/config_i.yaml"
)

results = []

chunks = list(corpus.chunks_with_topic(topic_id=0, sample_size=50))

for chunk in chunks:
    text = chunk.text
    summary = chunk.metadata.get("summary", "")

    questions, _ = mind._generate_questions(chunk)

    results.append({
        "passage": text,
        "summary": summary,
        "questions": questions
    })

df_out = pd.DataFrame(results)
df_out.to_excel("/export/usuarios01/ivgomez/mind/outputs_pipeline/1_questions_testing/spanish_only/summary_mt5.xlsx", index=False)

print(df_out.head())