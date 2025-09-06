from mind.corpus_building.preprocessing import DataPreparer

if __name__ == "__main__":

    prep = DataPreparer(
        path_folder="/export/usuarios_ml4ds/lbartolome/Repos/alonso_mind/Data/1.5_trans_data/en_gen",
        name_anchor="en_2025-09-05_segm_trans.parquet",
        name_target="de_2025-09-05_segm_trans.parquet",
        storing_path="/export/usuarios_ml4ds/lbartolome/Repos/alonso_mind/Data",
        preproc_script="externals/NLPipe/src/nlpipe/cli.py",
        config_path="config.json",
        stw_path="externals/NLPipe/src/nlpipe/stw_lists",
        spacy_models={"EN":"en_core_web_sm","DE":"de_core_news_sm","ES":"es_core_news_sm"},
        schema={
        "chunk_id": "id_preproc",
        "doc_id": "id",
        "text": "text",
        "full_doc": "summary",
        "lang": "lang",
        "title": "title",
        "url": "url",
        "equivalence": "equivalence",
    },
    )

prep.format_dataframes() 
