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
    mind.run_pipeline(
        topics=[topic], path_save="data/mind_runs/rosie")
    
    # mind = MIND(
    #     llm_model="qwen:32b",#"qwen:32b",
    #     source_corpus={
    #         "corpus_path": "/export/usuarios_ml4ds/lbartolome/Repos/alonso_mind/Data/polylingual_df.parquet", 
    #         "thetas_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/models/wiki/ende/poly_en_de_05_09_30/mallet_output/thetas_EN.npz",
    #         "id_col": "doc_id",
    #         "passage_col": "text",
    #         "full_doc_col": "full_doc",
    #         "language_filter": "EN",
    #         },
    #     target_corpus={
    #         "corpus_path": "/export/usuarios_ml4ds/lbartolome/Repos/alonso_mind/Data/polylingual_df.parquet",
    #         "thetas_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/models/wiki/ende/poly_en_de_05_09_30/mallet_output/thetas_DE.npz",
    #         "id_col": "doc_id",
    #         "passage_col": "text",
    #         "full_doc_col": "full_doc",
    #         "language_filter": "DE",
    #         "index_path": "/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/mind_runs/ende/indexes",
    #     },
    #     dry_run=False,
    #     do_check_entailement=True
    # )
    
    # topic = 17
    # mind.run_pipeline(topics=[topic], sample_size=300, path_save="data/mind_runs/ende/results")
    
    
    
    