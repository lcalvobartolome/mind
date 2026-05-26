import pandas as pd

df = pd.read_parquet('/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_subes.parquet')
df.to_excel('/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_subes1.xlsx', index=False)