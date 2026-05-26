import pandas as pd


df = pd.read_parquet("/export/usuarios01/ivgomez/mind/scrapy_data/iss_az_dropped.parquet")

prefix = "WIT"


df["id_num"] = range(1, len(df) + 1)

df["codigo"] = prefix + df["id_num"].astype(str).str.zfill(3)

df.drop(columns=["id_num"], inplace=True)


df.to_parquet("/export/usuarios01/ivgomez/mind/scrapy_data/iss_az_coded.parquet", index=False)