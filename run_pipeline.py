# main 
from pathlib import Path
from src.mind.pipeline import MIND
import pandas as pd

if __name__ == "__main__":
    
    llm_model = "qwen:32b"
    num_topics = 30
    
    model_folder = f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/28_jan/poly_rosie_1_{num_topics}"
    corpus_path = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet"
    
    source_corpus = {
        "corpus_path": corpus_path,
        "thetas_path": f"{model_folder}/mallet_output/thetas_EN.npz",
        "id_col": "doc_id",
        "passage_col": "text",
        "full_doc_col": "full_doc",
        "language_filter": "EN",
        "load_thetas": True
    }
    
    index_path = f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/rosie/index_corpus_train/{num_topics}k"
    
    target_corpus = {
        "corpus_path": corpus_path,
        "thetas_path": f"{model_folder}/mallet_output/thetas_ES.npz",
        "id_col": "doc_id",
        "passage_col": "text",
        "full_doc_col": "full_doc",
        "language_filter": "ES",
        "load_thetas": True,
        "index_path": index_path
    }
    
    mind = MIND(
        llm_model=llm_model,#"qwen:32b",
        source_corpus=source_corpus,
        target_corpus=target_corpus,
        retrieval_method="TB-ENN",
        multilingual=True,
        config_path=Path("config/config.yaml"),
        logger=None,
        dry_run=False
    )
    
    topic = 3 # Pediatric Healthcare
    print(f"Running pipeline for topic {topic}")
    sample_size = 1000
    mind.run_pipeline([topic], sample_size = sample_size)
    # save results as df
    results = pd.DataFrame(mind.results)
    results.to_parquet(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/rosie/results/{num_topics}k_sample_size_{sample_size}_results.parquet", index=False)
    import pdb; pdb.set_trace()