import pandas as pd
from pathlib import Path

input_path = "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/segmented_it.parquet"

output_dir = "/export/usuarios01/ivgomez/mind/outputs_pipeline/corpus_building/splits"

chunk_size = 40000

Path(output_dir).mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(input_path)

for i in range(0, len(df), chunk_size):

    chunk = df.iloc[i:i + chunk_size]

    output_path = f"{output_dir}/segmented_it_part_{i//chunk_size}.parquet"

    chunk.to_parquet(output_path, compression="gzip")

    print(f"Saved: {output_path} ({len(chunk)} rows)")