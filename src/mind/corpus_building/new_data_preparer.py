"""
Builds a polylingual dataset for the PLTM wrapper, starting from two input parquet files:
    - One file per language: anchor and comparison.
    - Each row represents a chunk/passage with its original metadata.
    - 'lemmas' contains the lemmatized text from NLPipe (run once per language).
    - 'lemmas_tr' contains lemmas from the cross-language translation (from the other language's NLPipe output).

Assumptions:
    - Two input parquet files, one for the anchor language and one for the comparison language.
    - Chunk IDs follow patterns like:
        Originals:     EN_<doc>_<chunk>   /  DE_<doc>_<chunk> / ES_...
        Translations:  T_EN_<doc>_<chunk> / T_DE_<doc>_<chunk> ...
      IMPORTANT (confirmed against real data): the language tag in a translation
      row (T_<lang>_...) refers to the SOURCE document's language, not the
      actual language of the text in that row. The text itself is a machine
      translation INTO the other file's language.
      Example: 'ES_FYS001_0' (in the ES/anchor file) is a native Spanish chunk.
      'T_ES_FYS001_0' (also stored in the ES/anchor file) is its translation
      INTO ITALIAN — the row lives next to its Spanish source for bookkeeping,
      but the text itself must be processed with the Italian NLPipe/spaCy model,
      not the Spanish one. Symmetrically, 'T_IT_<doc>_<chunk>' rows stored in
      the IT/comparison file contain Spanish text (translations of IT originals).
      The cross-lingual pairing key is the '<doc>_<chunk>' suffix, stripped of
      any language prefix, which is identical between a source row and its
      translation regardless of which file each lives in.

Output columns (plus any extra metadata preserved):
    - chunk_id   (from schema)
    - doc_id     (from schema)
    - full_doc   (from summary-like field if available)
    - text       (original chunk text)
    - lang       (UPPER, e.g., EN/DE/ES)
    - lemmas     (from NLPipe run on this language)
    - lemmas_tr  (from the other language's NLPipe run over translations)
    - title, url, equivalence (if present)

------------------------------------------------------------------------------
NOTA DE DEPURACIÓN (historial del problema con lemmas_tr):
1ª causa descartada: no era un problema de emparejamiento de IDs (eso funcionaba).
2ª causa, la real: NLPipe procesaba "un fichero = un idioma" (--lang es para todo
el fichero anchor), pero las filas T_ES_ dentro del fichero ES en realidad
contienen texto en ITALIANO (son la traducción al italiano del documento ES
correspondiente). El filtro de detección de idioma de NLPipe las descartaba
correctamente como "no son español", perdiendo el 99.9% de las traducciones.
Fix: en vez de procesar por fichero, se agrupan las filas por IDIOMA REAL del
texto (deducido del chunk_id) antes de llamar a NLPipe: un lote con todo el
texto realmente en anchor_lang (originales + traducciones T_<comp_lang>_) y
otro lote con todo el texto realmente en comp_lang (originales + traducciones
T_<anchor_lang>_). Los logs [DIAGNÓSTICO] muestran el tamaño de cada lote y de
los mapas de traducción resultantes para poder verificar el resultado.
------------------------------------------------------------------------------
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import pandas as pd
from mind.utils.utils import init_logger


class DataPreparer:
    def __init__(
        self,
        preproc_script: Optional[str] = None,
        config_path: Optional[str] = None,
        stw_path: Optional[str] = None,
        python_exe: str = "python3",
        spacy_models: Optional[Dict[str, str]] = None,
        schema: Optional[Dict[str, str]] = None,
        config_logger_path: Path = Path("config/config.yaml"),
        logger=None
    ):
        self._logger = logger if logger else init_logger(
            config_logger_path, __name__)

        # configure NLPipe
        self.preproc_script = Path(preproc_script) if preproc_script else None
        self.config_path = Path(config_path) if config_path else None
        self.stw_path = Path(stw_path) if stw_path else None
        self.python_exe = python_exe
        self.spacy_models = {k.upper(): v for k, v in (
            spacy_models or {}).items()}

        # Schema mapping: user must provide all required fields; optional fields are preserved automatically
        required_fields = ['chunk_id', 'text', 'lang', 'full_doc', 'doc_id']
        msg_fails = (
            "You must provide a schema mapping all required fields ('chunk_id', 'text', 'lang', 'full_doc', 'doc_id') to column names:\n"
            "- 'chunk_id': unique id for each chunk/passage\n"
            "- 'text': text content of the chunk\n"
            "- 'lang': language code (e.g., EN, ES)\n"
            "- 'full_doc': full document before chunking\n"
            "- 'doc_id': document id for each chunk\n"
            "Any other columns in your input file will be preserved as extra metadata.\n\n"
            "Example: schema = {\n"
            "    'chunk_id': 'id_preproc',\n"
            "    'text': 'chunk_text',\n"
            "    'lang': 'language',\n"
            "    'full_doc': 'summary',\n"
            "    'doc_id': 'id'\n"
            "}\n"
        )
        if schema is None or not all(f in schema for f in required_fields):
            raise ValueError(msg_fails)
        self.schema = schema

    @staticmethod
    def _upper_lang(x) -> str:
        return (x if isinstance(x, str) else str(x)).strip().upper()

    def _starts_with(self, series: pd.Series, prefix: str) -> pd.Series:
        return series.fillna("").astype(str).str.startswith(prefix)

    def _normalize(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rename required columns according to schema and preserve any other metadata that was in the original dataframe.
        """
        df = df.copy()

        # Required fields
        c_chunk = self.schema.get("chunk_id")
        c_text = self.schema.get("text")
        c_lang = self.schema.get("lang")
        c_full = self.schema.get("full_doc")
        c_doc = self.schema.get("doc_id")

        if not c_chunk or not c_text or not c_lang or not c_full:
            raise ValueError(
                "Schema must provide 'chunk_id', 'text', 'lang', and 'full_doc' column names.")
        for col in [c_chunk, c_text, c_lang, c_full]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' specified in schema is missing from DataFrame.")

        # Build normalized frame
        out = pd.DataFrame({
            "chunk_id": df[c_chunk].astype(str),
            "text": df[c_text],
            "full_doc": df[c_full],
            "doc_id": df[c_doc].astype(str),
        })

        # lang
        lang_val = df[c_lang].iloc[0] if df[c_lang].notna().any() else "XX"
        if lang_val == "XX":
            raise ValueError(
                f"Language column '{c_lang}' has no valid entries. Please check your data."
            )
        out["lang"] = self._upper_lang(lang_val)

        # Keep any extra columns from input that are not mapped by schema
        mapped_cols = set([c_chunk, c_text, c_lang, c_full, c_doc])
        extras = [c for c in df.columns if c not in mapped_cols]
        for c in extras:
            out[c] = df[c]

        return out

    def _spacy_model_for(self, lang_upper: str) -> str:
        """Get the spaCy model name for a specific language."""
        if not self.spacy_models or self._upper_lang(lang_upper) not in self.spacy_models:
            raise ValueError(f"No spaCy model configured for '{lang_upper}'. "
                             f"Provide spacy_models like {{'en':'en_core_web_sm'}}.")
        return self.spacy_models[self._upper_lang(lang_upper)]

    def _preprocess_df(
        self,
        df: pd.DataFrame,
        lang_upper: str,
        tag: str,
        path_save: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Run NLPipe on a temporary parquet file containing only the required columns:
        - id_preproc (from df['chunk_id'])
        - text (from df['text'])
        - lang (from df['lang'])
        After NLPipe processes the file and adds a 'lemmas' column, merge these lemmas back into the normalized DataFrame using 'chunk_id' as the key.
        """

        # normalized columns must exist
        required = {"chunk_id", "text", "lang"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"_preprocess_df expects normalized df with columns {required}. "
                f"Missing: {missing}. Did _normalize() set 'chunk_id' from your schema?"
            )

        tmp_dir = (Path(path_save).parent / "_tmp_preproc")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # temp parquet for nlpipe
        work = pd.DataFrame({
            "id_preproc": df["chunk_id"].astype(str),
            "text":  df["text"],
            "lang": df["lang"].astype(str).str.lower(),
        })
        tmp_parq = tmp_dir / f"{tag}_{lang_upper}.parquet"
        work.to_parquet(tmp_parq, compression="gzip")

        # --- DIAGNÓSTICO: cuántas filas entran a NLPipe y cuántas son traducciones ---
        n_before = len(work)
        n_trans_before = self._starts_with(work["id_preproc"], "T_").sum()
        self._logger.info(
            f"[DIAGNÓSTICO] _preprocess_df({tag}/{lang_upper}): entran {n_before} filas a NLPipe "
            f"({n_trans_before} con prefijo 'T_')."
        )

        # run NLPipe
        if self.preproc_script and self.config_path and self.stw_path:
            cmd = [
                self.python_exe, str(self.preproc_script),
                "--source_path", str(tmp_parq),
                "--source_type", "parquet",
                "--source", "mind",
                "--destination_path", str(tmp_parq),
                "--lang", lang_upper.lower(),
                "--spacy_model", self._spacy_model_for(lang_upper),
                "--config_file", str(self.config_path),
                "--stw_path", str(self.stw_path),
            ]
            print("Running NLPipe:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            print(f"✓ Preprocessed (lang={lang_upper})")
        else:
            print("Preprocessing skipped (not configured).")
            return df

        # read back NLPipe output and merge lemmas back by id_preproc - chunk_id
        proc = pd.read_parquet(tmp_parq)
        if "id_preproc" not in proc.columns or "lemmas" not in proc.columns:
            raise RuntimeError(
                f"NLPipe output missing id_preproc/lemmas; got: {list(proc.columns)}")

        # --- DIAGNÓSTICO: cuántas filas salen de NLPipe y cuántas se perdieron ---
        n_after = len(proc)
        n_lemmas_ok = proc["lemmas"].notna().sum()
        self._logger.info(
            f"[DIAGNÓSTICO] _preprocess_df({tag}/{lang_upper}): salen {n_after} filas de NLPipe "
            f"({n_lemmas_ok} con 'lemmas' no nulo)."
        )
        if n_after < n_before:
            ids_in = set(work["id_preproc"].astype(str))
            ids_out = set(proc["id_preproc"].astype(str))
            lost_ids = ids_in - ids_out
            n_lost_trans = sum(1 for i in lost_ids if str(i).startswith("T_"))
            self._logger.warning(
                f"[DIAGNÓSTICO] NLPipe perdió {n_before - n_after} filas en ({tag}/{lang_upper}); "
                f"de ellas, {n_lost_trans} eran filas de traducción (prefijo 'T_'). "
                "Si este número es alto, el filtro de detección de idioma de NLPipe "
                "(paso 'Detecting language...' en su log) probablemente está descartando "
                "las traducciones porque su idioma detectado no coincide con --lang. "
                "Esto explicaría que lemmas_tr salga vacío."
            )

        merged = df.merge(
            proc[["id_preproc", "lemmas"]],
            left_on="chunk_id",
            right_on="id_preproc",
            how="left",
            validate="one_to_one",
        )

        to_drop = [c for c in merged.columns if c.startswith("id_preproc")]
        merged = merged.drop(columns=to_drop)

        # remove tmp file
        tmp_parq.unlink(missing_ok=True)

        return merged

    @staticmethod
    def _pair_key_from_chunk_id(chunk_id: str) -> str:
        """
        Returns the '<doc>_<chunk>' suffix shared between an original chunk and
        its cross-lingual translation, stripping any leading language tag.

        The language tag in a translation row (T_<lang>_...) denotes the
        DESTINATION language (the file it lives in), not the source language,
        so it carries no useful pairing information and must be discarded.

        Examples:
          ES_FYS001_0     -> "FYS001_0"
          T_ES_FYS001_0   -> "FYS001_0"
          IT_FYS001_0     -> "FYS001_0"
          T_IT_FYS001_0   -> "FYS001_0"
        """
        s = str(chunk_id)
        m = re.match(r"^T_[A-Za-z]{2}_(.+)$", s)
        if m:
            return m.group(1)
        m2 = re.match(r"^[A-Za-z]{2}_(.+)$", s)
        if m2:
            return m2.group(1)
        return s

    def format_dataframes(
        self,
        anchor_path: Path,
        comparison_path: Path,
        path_save: Optional[Path] = None
    ) -> None:
        """
        Formats the anchor and target dataframes according to the specified schema, runs NLPipe preprocessing on both languages, and merges the results to produce a unified dataframe with lemmas from both the original and translated texts.
        """

        self._logger.info("Starting format_dataframes process...")

        anchor_df = pd.read_parquet(anchor_path)
        comparison_df = pd.read_parquet(comparison_path)

        # --- DIAGNÓSTICO: ¿hay filas de traducción (prefijo 'T_') en los ficheros de entrada? ---
        c_chunk_raw = self.schema["chunk_id"]
        n_anchor_trans = self._starts_with(anchor_df[c_chunk_raw], "T_").sum()
        n_comp_trans = self._starts_with(comparison_df[c_chunk_raw], "T_").sum()
        self._logger.info(
            f"[DIAGNÓSTICO] anchor_path: {len(anchor_df)} filas totales, "
            f"{n_anchor_trans} con prefijo 'T_'."
        )
        self._logger.info(
            f"[DIAGNÓSTICO] comparison_path: {len(comparison_df)} filas totales, "
            f"{n_comp_trans} con prefijo 'T_'."
        )
        if n_anchor_trans:
            sample_a = anchor_df.loc[
                self._starts_with(anchor_df[c_chunk_raw], "T_"), c_chunk_raw
            ].head(5).tolist()
            self._logger.info(f"[DIAGNÓSTICO] Ejemplos T_ en anchor: {sample_a}")
        if n_comp_trans:
            sample_c = comparison_df.loc[
                self._starts_with(comparison_df[c_chunk_raw], "T_"), c_chunk_raw
            ].head(5).tolist()
            self._logger.info(f"[DIAGNÓSTICO] Ejemplos T_ en comparison: {sample_c}")
        if n_anchor_trans == 0 and n_comp_trans == 0:
            self._logger.warning(
                "[DIAGNÓSTICO] No hay NINGUNA fila con prefijo 'T_' en ninguno de los dos "
                "ficheros de entrada. lemmas_tr quedará vacío para todas las filas porque "
                "el script no tiene traducciones que cruzar. Revisa cómo generas/nombras "
                "los chunk_id de las traducciones ANTES de seguir depurando el resto del pipeline."
            )

        # Use lang column directly from schema
        anchor_lang = self._upper_lang(anchor_df[self.schema["lang"]].iloc[0])
        comp_lang = self._upper_lang(comparison_df[self.schema["lang"]].iloc[0])
        self._logger.info(
            f"Anchor language: {anchor_lang}, Comparison language: {comp_lang}")

        # No decontamination or stable ordering needed after translation
        anc = anchor_df.copy()
        comp = comparison_df.copy()

        # Normalize columns to a common schema
        self._logger.info("Normalizing columns to a common schema...")
        anc_norm = self._normalize(anc)
        comp_norm = self._normalize(comp)

        # --- Determinar el idioma REAL de cada fila a partir del chunk_id ---
        # Las filas con prefijo 'T_<idioma_del_propio_fichero>_' NO están en ese
        # idioma: son la traducción de ese documento al idioma DEL OTRO fichero.
        # P.ej. 'T_ES_FYS001_0' vive en el fichero ES pero su texto está en italiano
        # (es la traducción de 'ES_FYS001_0' al italiano).
        is_anc_trans = anc_norm["chunk_id"].astype(
            str).str.startswith(f"T_{anchor_lang}_")
        is_comp_trans = comp_norm["chunk_id"].astype(
            str).str.startswith(f"T_{comp_lang}_")

        self._logger.info(
            f"[DIAGNÓSTICO] anchor: {(~is_anc_trans).sum()} filas con texto real en {anchor_lang} "
            f"(originales), {is_anc_trans.sum()} filas con texto real en {comp_lang} (traducciones)."
        )
        self._logger.info(
            f"[DIAGNÓSTICO] comparison: {(~is_comp_trans).sum()} filas con texto real en {comp_lang} "
            f"(originales), {is_comp_trans.sum()} filas con texto real en {anchor_lang} (traducciones)."
        )

        # Lote homogéneo de texto REALMENTE en anchor_lang: originales ES + T_IT_ (cuyo texto es ES)
        batch_anchor_lang = pd.concat(
            [
                anc_norm[~is_anc_trans].assign(lang=anchor_lang),
                comp_norm[is_comp_trans].assign(lang=anchor_lang),
            ],
            ignore_index=True,
        )
        # Lote homogéneo de texto REALMENTE en comp_lang: originales IT + T_ES_ (cuyo texto es IT)
        batch_comp_lang = pd.concat(
            [
                comp_norm[~is_comp_trans].assign(lang=comp_lang),
                anc_norm[is_anc_trans].assign(lang=comp_lang),
            ],
            ignore_index=True,
        )

        self._logger.info(
            f"[DIAGNÓSTICO] Lote {anchor_lang} (texto real {anchor_lang}) para NLPipe: "
            f"{len(batch_anchor_lang)} filas."
        )
        self._logger.info(
            f"[DIAGNÓSTICO] Lote {comp_lang} (texto real {comp_lang}) para NLPipe: "
            f"{len(batch_comp_lang)} filas."
        )

        # Run preprocessing per ACTUAL text language (not per input file)
        self._logger.info(
            "Running NLPipe preprocessing per actual text language...")
        proc_anchor_lang = self._preprocess_df(
            batch_anchor_lang, anchor_lang, tag="lang", path_save=path_save)
        proc_comp_lang = self._preprocess_df(
            batch_comp_lang, comp_lang, tag="lang", path_save=path_save)

        # Pairing key = sufijo '<doc>_<chunk>', sin idioma
        proc_anchor_lang["pair_key"] = proc_anchor_lang["chunk_id"].astype(
            str).apply(self._pair_key_from_chunk_id)
        proc_comp_lang["pair_key"] = proc_comp_lang["chunk_id"].astype(
            str).apply(self._pair_key_from_chunk_id)

        # Las traducciones de documentos ANCHOR (T_<anchor_lang>_) se procesaron
        # dentro del lote comp_lang (su texto real es comp_lang); por eso ahí es
        # donde hay que buscar el lemmas_tr de los originales anchor.
        anc_trans_proc = proc_comp_lang[
            proc_comp_lang["chunk_id"].astype(
                str).str.startswith(f"T_{anchor_lang}_")
        ]
        # Las traducciones de documentos COMPARISON (T_<comp_lang>_) se procesaron
        # dentro del lote anchor_lang.
        comp_trans_proc = proc_anchor_lang[
            proc_anchor_lang["chunk_id"].astype(
                str).str.startswith(f"T_{comp_lang}_")
        ]
        map_lemmas_tr_for_anchor = dict(
            zip(anc_trans_proc["pair_key"], anc_trans_proc["lemmas"]))
        map_lemmas_tr_for_comp = dict(
            zip(comp_trans_proc["pair_key"], comp_trans_proc["lemmas"]))

        self._logger.info(
            f"[DIAGNÓSTICO] Traducciones T_{anchor_lang}_ con lemmas (texto real {comp_lang}, "
            f"para lemmas_tr de originales {anchor_lang}): {len(map_lemmas_tr_for_anchor)} "
            f"de {is_anc_trans.sum()} filas."
        )
        self._logger.info(
            f"[DIAGNÓSTICO] Traducciones T_{comp_lang}_ con lemmas (texto real {anchor_lang}, "
            f"para lemmas_tr de originales {comp_lang}): {len(map_lemmas_tr_for_comp)} "
            f"de {is_comp_trans.sum()} filas."
        )
        if len(map_lemmas_tr_for_anchor) == 0 and len(map_lemmas_tr_for_comp) == 0:
            self._logger.warning(
                "[DIAGNÓSTICO] Ambos mapas de traducción siguen vacíos tras corregir el "
                "idioma real. Revisa si is_anc_trans/is_comp_trans detectan correctamente "
                "los prefijos, y si _pair_key_from_chunk_id extrae el mismo sufijo en ambos lados."
            )

        # lemmas propios: cada original busca sus lemmas en el lote de SU idioma real
        lemmas_anchor_by_chunk = dict(
            zip(proc_anchor_lang["chunk_id"], proc_anchor_lang["lemmas"]))
        lemmas_comp_by_chunk = dict(
            zip(proc_comp_lang["chunk_id"], proc_comp_lang["lemmas"]))

        anc_orig = anc_norm[~is_anc_trans].copy()
        anc_orig["pair_key"] = anc_orig["chunk_id"].astype(
            str).apply(self._pair_key_from_chunk_id)
        anc_orig["lemmas"] = anc_orig["chunk_id"].map(lemmas_anchor_by_chunk)
        anc_orig["lemmas_tr"] = anc_orig["pair_key"].map(
            map_lemmas_tr_for_anchor)

        comp_orig = comp_norm[~is_comp_trans].copy()
        comp_orig["pair_key"] = comp_orig["chunk_id"].astype(
            str).apply(self._pair_key_from_chunk_id)
        comp_orig["lemmas"] = comp_orig["chunk_id"].map(lemmas_comp_by_chunk)
        comp_orig["lemmas_tr"] = comp_orig["pair_key"].map(
            map_lemmas_tr_for_comp)

        anc_orig.drop(columns=["pair_key"], inplace=True)
        comp_orig.drop(columns=["pair_key"], inplace=True)

        # final stack
        final_df = pd.concat([anc_orig, comp_orig], ignore_index=True)

        # drop all rows where lemmas is None
        final_df = final_df[~final_df.lemmas.isnull()]
        # replace None in lemmas_tr with empty string
        final_df["lemmas_tr"] = final_df["lemmas_tr"].fillna("")

        # no duplicate chunk_ids
        assert not final_df["chunk_id"].duplicated().any(), (
            f"Duplicate chunk_id values found in final dataframe: "
            f"{final_df[final_df['chunk_id'].duplicated(keep=False)]['chunk_id'].tolist()}"
        )
        # all lemmas and lemmas_tr are non-null
        assert not final_df["lemmas"].isnull().any(), (
            f"Null lemmas found in final dataframe for chunk_ids: "
            f"{final_df[final_df['lemmas'].isnull()]['chunk_id'].tolist()}"
        )
        assert not final_df["lemmas_tr"].isnull().any(), (
            f"Null lemmas_tr found in final dataframe for chunk_ids: "
            f"{final_df[final_df['lemmas_tr'].isnull()]['chunk_id'].tolist()}"
        )

        # the number of rows in anchor_lang is equals to that of comparison_lang
        n_anc = len(final_df[final_df["lang"] == anchor_lang])
        n_comp = len(final_df[final_df["lang"] == comp_lang])
        self._logger.warning(f"Anchor rows: {n_anc}, comparison rows: {n_comp}")

        # --- DIAGNÓSTICO: cuántas filas finales se quedaron sin lemmas_tr ---
        n_empty_tr = (final_df["lemmas_tr"] == "").sum()
        self._logger.warning(
            f"[DIAGNÓSTICO] Filas finales con lemmas_tr vacío: {n_empty_tr} de {len(final_df)}."
        )

        # Save unified parquet
        if path_save:
            final_df.to_parquet(path_save)
            self._logger.info(f"Saved: {path_save}")

        return final_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run DataPreparer to build a polylingual dataset.")
    parser.add_argument("--anchor", type=str, required=True,
                        help="Path to anchor language parquet file.")
    parser.add_argument("--comparison", type=str, required=True,
                        help="Path to comparison language parquet file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output parquet file.")
    parser.add_argument("--schema", type=str, required=True,
                        help="JSON string or path to schema mapping required columns.")
    args = parser.parse_args()

    try:
        if args.schema.endswith('.json'):
            with open(args.schema, 'r') as f:
                schema = json.load(f)
        else:
            schema = json.loads(args.schema)
    except Exception as e:
        raise ValueError(f"Failed to load schema: {e}")

    preparer = DataPreparer(python_exe = sys.executable,
        preproc_script="externals/NLPipe/src/nlpipe/cli.py",
        config_path="externals/NLPipe/config.json",
        stw_path="externals/NLPipe/src/nlpipe/stw_lists",
        spacy_models={
            "es": "es_core_news_sm",
            "it": "it_core_news_sm",
        },schema=schema)

    final_df = preparer.format_dataframes(
        anchor_path=Path(args.anchor),
        comparison_path=Path(args.comparison),
        path_save=Path(args.output)
    )
    print(
        f"Polylingual dataset created and saved to {args.output}. Rows: {len(final_df)}")