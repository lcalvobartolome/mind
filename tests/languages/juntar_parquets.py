import pandas as pd

# Leer los tres parquets
df1 = pd.read_parquet("/export/usuarios01/ivgomez/mind/scrapy_data/nostrofiglio_coded.parquet")
df2 = pd.read_parquet("/export/usuarios01/ivgomez/mind/scrapy_data/iss_az_coded.parquet")
df3 = pd.read_parquet("/export/usuarios01/ivgomez/mind/scrapy_data/wiki_it_coded.parquet")

# Unirlos
df_final = pd.concat([df1, df2, df3], ignore_index=True)

# Guardar el resultado
df_final.to_parquet("/export/usuarios01/ivgomez/mind/tests/data_final/dataset_it_sin_resumen.parquet", index=False)