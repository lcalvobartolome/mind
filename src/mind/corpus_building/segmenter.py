import argparse
from pathlib import Path
import re

import pandas as pd
from mind.utils.utils import init_logger
from tqdm import tqdm


class Segmenter():
    def __init__(
        self,
        config_path: Path = Path("config/config_i.yaml"),
        logger=None
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)

    def segment(
        self,
        path_df: Path,
        path_save: Path,
        text_col: str = "text",
        min_length: int = 100,
        sep: str = "\n"  # se mantiene pero ya no es crítico
    ):

        self._logger.info(f"Loading dataframe from {path_df}")
        df = pd.read_parquet(path_df)

        self._logger.info(
            f"Loaded {len(df)} rows. Starting segmentation on column '{text_col}'...")

        orig_cols = list(df.columns)
        new_rows = []

        import time
        start_time = time.time()

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting paragraphs"):
            full_doc_text = str(row[text_col])

            # 🔥 NUEVO: segmentación robusta
            paragraphs = [
                p.strip() for p in re.split(r'\n+|\.\s+', full_doc_text)
                if p and len(p.strip()) > min_length
            ]

            # 🔥 fallback: si no se segmenta, mantener texto original
            if len(paragraphs) == 0:
                paragraphs = [full_doc_text]

            for idx, p in enumerate(paragraphs):
                entry = {col: row.get(col, None) for col in orig_cols}

                # texto segmentado
                entry[text_col] = p

                # documento completo
                entry['full_doc'] = full_doc_text

                # 🔥 FIX CLAVE: usar codigo o id
                base_id = row.get('codigo', row.get('id', row.name))
                entry['id_preproc'] = f"{base_id}_{idx}"

                new_rows.append(entry)

        elapsed = time.time() - start_time
        self._logger.info(f"Segmentation took {elapsed:.2f} seconds.")

        seg_df = pd.DataFrame(new_rows)

        # nuevo id incremental
        seg_df['id'] = range(len(seg_df))

        self._logger.info(
            f"Segmented into {len(seg_df)} passages. Saving to {path_save}")

        seg_df.to_parquet(path_save, compression="gzip")

        self._logger.info(f"Saved segmented dataframe to {path_save}")

        return path_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Segmenter to split documents into segments.")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--min_length", type=int, default=100)
    parser.add_argument("--separator", type=str, default="\n")

    args = parser.parse_args()

    segmenter = Segmenter()

    result_path = segmenter.segment(
        path_df=Path(args.input),
        path_save=Path(args.output),
        text_col=args.text_col,
        min_length=args.min_length,
        sep=args.separator
    )

    result_df = pd.read_parquet(result_path)

    print(f"Segmentation complete. Saved to {args.output}. Rows: {len(result_df)}")