import pandas as pd


#df = pd.read_parquet("/export/usuarios01/ivgomez/mind/tests/data_final/dataset_it_con_resumen_EA3_clean_deverdad2.parquet")
#df = pd.read_parquet("/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/splits/segmented_it_translated_0.parquet")
#df = pd.read_excel("/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/mind_runs/rosie/v2/results/annotated/rosie_mind_v3_annotated.xlsx")
#df_sample = df.head(250)
#df_sample.to_excel("/export/usuarios01/ivgomez/mind/pipeline_irina/1_questions_testing/rosie_mind_v3_annotated_sample.xlsx", index=False)

#df_sample.to_parquet("/export/usuarios01/ivgomez/mind/tests/data_final/dataset_es_final_clean_EA3.parquet", index=False)

df = pd.read_parquet("/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_es_translated_coded.parquet")
df_sample = df.tail(20)
df_sample.to_excel("/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_es_translated_coded_tail.xlsx", index=False)