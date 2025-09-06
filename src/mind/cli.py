import pandas as pd
from mind.pipeline.pipeline import MIND


if __name__ == "__main__":
    
    # Example usage
    
    mind = MIND(
        llm_model="qwen:32b",#"qwen:32b",
        source_corpus={
            "corpus_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet", "thetas_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/26_jan_no_dup/poly_rosie_1_30/mallet_output/thetas_EN.npz",
            "id_col": "doc_id",
            "passage_col": "text",
            "full_doc_col": "full_doc",
            "language_filter": "EN",
            },
        target_corpus={
            "corpus_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet", "thetas_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/26_jan_no_dup/poly_rosie_1_30/mallet_output/thetas_ES.npz",
            "id_col": "doc_id",
            "passage_col": "text",
            "full_doc_col": "full_doc",
            "language_filter": "ES",
            "index_path": "test",
        },
        dry_run=False,
        do_check_entailement=True
    )
    
    topic = 15
    mind.run_pipeline([topic], sample_size=100)
    
    results = pd.DataFrame(mind.results)
    discarded = pd.DataFrame(mind.discarded)
    discarded.to_parquet("/polylingual_discarded.parquet", index=False)
    import pdb; pdb.set_trace()
    
    
    