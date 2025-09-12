from typing import Dict, Optional, Tuple, List
import subprocess
import re
from pathlib import Path
import pandas as pd
import os
import logging
from datetime import date
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from datasets import Dataset


class Segmenter():
    '''
    Class to carry out the segmentation of the dataset
    generated with wiki-retriever

    Parameters
    -------------
    in_directory: str
        Directory where the input file can be found
            /export/users/DATA
    file_name: str
        Name of the file itself
            mydata.parquet.gzip
    '''

    def __init__(self,
                 in_directory: str,
                 file_name: str,
                 out_directory: str,
                 input_df: pd.DataFrame = None,
                 segmented_df: pd.DataFrame = None,
                 logger: logging.Logger = None
                 ):

        self.in_directory = in_directory
        self.file_name = file_name
        self.out_directory = out_directory

        self.en_df = None
        self.es_df = None

        self.input_df = input_df if input_df is not None else pd.DataFrame(columns=["title", "summary", "lang", "url",  "id",  "equivalence",   "id_preproc"])

        self.segmented_df = segmented_df if segmented_df is not None else pd.DataFrame(columns=["title",  "summary", "text", "lang", "url",  "id",
         "equivalence",  "id_preproc"])
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Segmenter')
            # Add a console handler to output logs to the console
            console_handler = logging.StreamHandler()
            # Set handler level to INFO or lower
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    def read_dataframe(self) -> None:
        '''
        Checks the in_directory path and
        saves the file as a PD dataframe (input_df)
        '''

        # if not os.path.exists(self.in_directory):
        #  raise Exception('Path not found, check again')

        # elif not os.path.isfile(os.path.join(self.in_directory, self.file_name)):
        #  raise Exception('File not found, check again')

        # else:
        # self.input_df = pd.read_parquet(os.path.join(self.in_directory, self.file_name))
        if not Path(self.file_name).is_file():
            print(f"File {self.file_name} does not exist")
        self.input_df = pd.read_parquet(self.file_name)
        self._logger.info("File read sucessfully!")
        return

    def segment(self) -> None:
        '''
        Iterates over the input dataset and creates a second one in which each
        entry represents a paragraph of the original
        '''
        self._logger.info("Starting segmentation!")
        for i, row in tqdm(self.input_df.iterrows(), total=len(self.input_df), desc="Processing rows"):

            # Separates each text over the paragraph
            split_text = row['text'].split("\n")
            # Filters section names and blanks
            filtered_text = filter(
                lambda x: x != '' and len(x) > 100, split_text)

            paragraphs = list(filtered_text)

            for _, p in enumerate(paragraphs):
                # Add the new data
                self.segmented_df.loc[len(self.segmented_df)] = [
                    row['title'],
                    row['summary'],
                    p,
                    row['lang'],
                    row['url'],
                    len(self.segmented_df),
                    row["equivalence"],
                    row['id_preproc']+"_"+str(_)
                ]

        langs = self.segmented_df.lang.unique()

        self.en_df = self.segmented_df[self.segmented_df['lang'] == langs[0]]
        self.es_df = self.segmented_df[self.segmented_df['lang'] == langs[1]]

        # TODO: implement machine translation functionality

        file_lang0 = self.save_to_parquet(self.en_df, langs[0])
        file_lang1 = self.save_to_parquet(self.es_df, langs[1])

        return file_lang0, file_lang1

    def save_to_parquet(self, df: pd.DataFrame, lang: str, collab: bool = False) -> None:

        date_name = str(date.today())

        file_name = f"{lang}_{date_name}_segmented_dataset.parquet.gzip"

        save_path = os.path.join(self.out_directory, file_name)
        Path(self.out_directory).mkdir(parents=True, exist_ok=True)

        if collab:
            if "drive" not in os.listdir("/content"):
                from google.colab import drive
                drive.mount('/content/drive')

            df.to_parquet(path=save_path, compression="gzip")
            print(f"Saving in Drive: {save_path}")

        else:
            df.to_parquet(path=save_path, compression="gzip")
            print(f"Saving in PC: {save_path}")

        return save_path


class Translator:
    """
    If langs are {'en','de'}:
      - Translate every EN doc to DE and append to de_df
      - Translate every DE doc to EN and append to en_df
    Also supports {'en','es'} the same way.
    """

    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame):
        # Infer languages deterministically
        lang_a = df_a.lang.unique()[0]
        lang_b = df_b.lang.unique()[0]
        if lang_a == lang_b:
            raise ValueError(
                f"Both dataframes claim {lang_a}. Need two different languages.")

        self.src_langs = (lang_a, lang_b)
        self.langs = {lang_a, lang_b}

        # Keep dfs by their language string for clarity
        self.df_by_lang = {lang_a: df_a.reset_index(drop=True),
                           lang_b: df_b.reset_index(drop=True)}

        # Build direction → model/tokenizer maps
        self.models = {}
        self.tokenizers = {}

        def add_pair(src, tgt, repo):
            self.models[(src, tgt)] = pipeline("translation", model=repo)
            self.tokenizers[(src, tgt)] = AutoTokenizer.from_pretrained(repo)

        supported = {
            ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
            ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
            ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
            ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
        }

        for (src, tgt), repo in supported.items():
            if {src, tgt} == self.langs:
                add_pair(src, tgt, repo)

        # Sanity: ensure we have both directions for the chosen pair
        needed = [(lang_a, lang_b), (lang_b, lang_a)]
        for pair in needed:
            if pair not in self.models:
                raise Exception(f"Unsupported language pair: {self.langs}")

        # Work vars
        self.split_by_lang = {}        # lang -> split df (sentences)
        # src lang -> translated sentences (to tgt)
        self.trans_text_by_lang = {}
        self.translated_df_by_lang = {}  # lang -> final df with appended translations

    def _split(self, df: pd.DataFrame, src_lang: str, tgt_lang: str) -> pd.DataFrame:
        """
        Split each paragraph into sentences, filtering out too-long chunks
        based on the *translation* tokenizer for (src->tgt).
        """
        tok = self.tokenizers[(src_lang, tgt_lang)]
        model_max = getattr(tok, "model_max_length", 512)
        max_tokens = int(model_max * 0.9)

        def token_len(text: str) -> int:
            return len(tok.encode(text, truncation=False))

        rows = []
        # If a row yields no valid sentences, we drop that whole id_preproc
        dropped_ids = set()

        for _, row in df.iterrows():
            sentences = [s for s in row["text"].split(". ") if s]
            any_kept = False
            for j, s in enumerate(sentences):
                if token_len(s) < max_tokens:
                    rows.append([
                        row["title"], row["summary"], s, row["lang"], row["url"],
                        None,  # index placeholder
                        row["equivalence"], f"{row['id_preproc']}_{j}"
                    ])
                    any_kept = True
            if not any_kept:
                dropped_ids.add(row["id_preproc"])

        if dropped_ids:
            # Remove rows whose entire paragraph was too long
            df = df[~df["id_preproc"].isin(dropped_ids)].reset_index(drop=True)

        out = pd.DataFrame(
            rows,
            columns=["title", "summary", "text", "lang",
                     "url", "index", "equivalence", "id_preproc"]
        )
        out["index"] = range(len(out))
        return out

    def _translate_split(self, split_df: pd.DataFrame, src_lang: str, tgt_lang: str) -> pd.Series:
        ds = Dataset.from_pandas(split_df)
        model = self.models[(src_lang, tgt_lang)]

        def translate_batch(batch):
            out = model(batch["text"])
            batch["translated_text"] = [o["translation_text"] for o in out]
            return batch

        ds = ds.map(translate_batch, batched=True)
        return ds.to_pandas()["translated_text"]

    def _assemble(self, split_df: pd.DataFrame, translated_text: pd.Series,
                  src_lang: str, tgt_lang: str) -> pd.DataFrame:
        """
        Replace sentence text with translated sentences, reassemble per paragraph,
        and produce a tgt-lang dataframe mirroring the source metadata.
        """
        tmp = split_df.copy()
        tmp["text"] = translated_text.values

        tmp["aux_id"] = tmp["id_preproc"].str.rsplit("_", n=1).str[0]
        grouped = (tmp.groupby("aux_id")["text"]
                      .agg(lambda x: ' '.join(x.astype(str).str.strip()))
                      .reset_index()
                      .rename(columns={"text": "assembled_text"}))

        merged = (tmp.merge(grouped, on="aux_id", how="outer")
                  .drop(columns=["text", "id_preproc"])
                  .rename(columns={"aux_id": "id_preproc", "assembled_text": "text"})
                  .drop_duplicates(subset=["id_preproc"])
                  .assign(id_preproc=lambda x: "T_" + x["id_preproc"])
                  .assign(lang=tgt_lang)
                  .reset_index(drop=True))
        return merged

    def translate(self):
        # Determine deterministic pair
        lang_a, lang_b = self.src_langs

        # Split each side using the tokenizer for its translation direction
        split_a = self._split(self.df_by_lang[lang_a], lang_a, lang_b)
        split_b = self._split(self.df_by_lang[lang_b], lang_b, lang_a)
        self.split_by_lang[lang_a] = split_a
        self.split_by_lang[lang_b] = split_b

        # Translate each split to the *other* language
        trans_a_to_b = self._translate_split(split_a, lang_a, lang_b)
        trans_b_to_a = self._translate_split(split_b, lang_b, lang_a)
        self.trans_text_by_lang[lang_a] = trans_a_to_b
        self.trans_text_by_lang[lang_b] = trans_b_to_a

        # Reassemble translated docs per direction
        # EN→DE produces DE docs
        merged_b = self._assemble(split_a, trans_a_to_b, lang_a, lang_b)
        # DE→EN produces EN docs
        merged_a = self._assemble(split_b, trans_b_to_a, lang_b, lang_a)

        # Append to originals, per language
        self.translated_df_by_lang[lang_a] = pd.concat(
            [self.df_by_lang[lang_a], merged_a], ignore_index=True
        )
        self.translated_df_by_lang[lang_b] = pd.concat(
            [self.df_by_lang[lang_b], merged_b], ignore_index=True
        )

        return

    def save_dataframes(self, path: str):
        if not self.translated_df_by_lang:
            raise RuntimeError("Call translate() before saving.")

        today = str(date.today())
        for lang, df in self.translated_df_by_lang.items():
            fname = f"{lang}_{today}_segm_trans.parquet"
            save_path = os.path.join(path, fname)
            df.to_parquet(path=save_path, compression="gzip")
            print(f"Saved: {save_path}")


class DataPreparer:
    """
    Build a polylingual dataset:
      - One row per chunk/passage with original metadata
      - 'lemmas' = preprocessed original text (via NLPipe, once per language)
      - 'lemmas_tr' = lemmas of the cross-language translated counterpart

    Assumptions:
      - Two input parquet files, one per language (anchor/target).
      - Chunk ids follow patterns like:
          Originals:     EN_<doc>_<chunk>   /  DE_<doc>_<chunk> / ES_...
          Translations: T_EN_<doc>_<chunk>  / T_DE_<doc>_<chunk> ...
        (Adjust regex in `_pair_key_from_chunk_id` if needed.)

    Column agnosticism:
      - Pass a `schema` dict mapping logical fields to your column names.
      - If omitted, auto-detection tries common defaults.

    Output columns include (plus any extra metadata preserved):
      - chunk_id   (from schema)
      - doc_id     (from schema or derived from chunk_id)
      - full_doc   (from summary-like field if available)
      - text       (original chunk text)
      - lang       (UPPER, e.g., EN/DE/ES)
      - lemmas     (from NLPipe run on this language)
      - lemmas_tr  (from the *other* language's NLPipe run over translations)
      - title, url, equivalence (if present)
    """

    def __init__(
        self,
        path_folder: str,
        name_anchor: str,
        name_target: str,
        storing_path: str = "",
        # NLPipe config
        preproc_script: Optional[str] = None,    # path to your NLPipe CLI
        config_path: Optional[str] = None,       # JSON config path
        stw_path: Optional[str] = None,          # stopword lists dir
        python_exe: str = "python3",
        # {"EN":"en_core_web_sm", ...}
        spacy_models: Optional[Dict[str, str]] = None,
        # Column schema (optional)
        schema: Optional[Dict[str, str]] = None,
    ):
        self.path_folder = Path(path_folder)
        self.storing_path = Path(
            storing_path) if storing_path else self.path_folder
        self.name_anchor = name_anchor
        self.name_target = name_target

        # NLPipe
        self.preproc_script = Path(preproc_script) if preproc_script else None
        self.config_path = Path(config_path) if config_path else None
        self.stw_path = Path(stw_path) if stw_path else None
        self.python_exe = python_exe
        self.spacy_models = {k.upper(): v for k, v in (
            spacy_models or {}).items()}

        # Schema mapping
        # logical_key -> column name in input
        self.schema = schema or {}
        self._fill_default_schema()

        # DFs
        self.anchor_df: Optional[pd.DataFrame] = None
        self.target_df: Optional[pd.DataFrame] = None
        self.final_df: Optional[pd.DataFrame] = None

    # ---------- schema & detection ----------

    def _fill_default_schema(self):
        # Provide common fallbacks if user didn't map them
        defaults = {
            "chunk_id": ["id_preproc", "chunk_id", "chunkId"],
            "doc_id":   ["id", "doc_id", "docId"],
            "text":     ["text", "chunk_text", "raw_text"],
            "full_doc": ["summary", "full_doc", "document_text"],
            "lang":     ["lang", "language", "lng"],
            "title":    ["title"],
            "url":      ["url", "link"],
            "equivalence": ["equivalence", "pair_id", "match_id"],
        }
        self.schema = {k: self.schema.get(k) for k in defaults}
        self.defaults = defaults

    def _resolve_col(self, df: pd.DataFrame, key: str) -> Optional[str]:
        """
        Return the column name in df that corresponds to logical `key`.
        Uses user-provided mapping first; else fall back over defaults.
        """
        name = self.schema.get(key)
        if name and name in df.columns:
            return name
        for cand in self.defaults.get(key, []):
            if cand in df.columns:
                # remember for next time
                self.schema[key] = cand
                return cand
        # not strictly required for some keys (e.g., title/url/equivalence)
        return None

    # ---------- reading ----------

    def _read_one(self, fname: str) -> pd.DataFrame:
        p = self.path_folder / fname
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        df = pd.read_parquet(p)
        print(f"Read {fname} ({len(df)} rows).")
        return df

    def read_dataframes(self) -> None:
        if not self.path_folder.exists():
            raise FileNotFoundError(f"Path not found: {self.path_folder}")
        self.anchor_df = self._read_one(self.name_anchor)
        self.target_df = self._read_one(self.name_target)

    # ---------- language & ids ----------

    @staticmethod
    def _upper_lang(x) -> str:
        return (x if isinstance(x, str) else str(x)).strip().upper()

    def _infer_lang_code(self, df: pd.DataFrame) -> str:
        lang_col = self._resolve_col(df, "lang")
        if lang_col and df[lang_col].notna().any():
            return self._upper_lang(df[lang_col].dropna().iloc[0])
        # Fallback: guess from chunk_id patterns
        cid = self._resolve_col(df, "chunk_id")
        if cid:
            for v in df[cid].dropna().astype(str):
                m = re.match(r"^(?:T_)?([A-Za-z]{2})_", v)
                if m:
                    return m.group(1).upper()
        return "XX"

    def _derive_doc_id(self, row: pd.Series, doc_col: Optional[str], chunk_col: str) -> str:
        if doc_col and pd.notna(row.get(doc_col)):
            return str(row[doc_col])
        # Derive from chunk id by dropping final _<number>
        cid = str(row.get(chunk_col, ""))
        m = re.match(r"^(.*)_(\d+)$", cid)
        return m.group(1) if m else cid

    # ---------- decontamination & ordering ----------

    def _starts_with(self, series: pd.Series, prefix: str) -> pd.Series:
        return series.fillna("").astype(str).str.startswith(prefix)

    def _decontaminate(self, df: pd.DataFrame, self_lang: str, other_lang: str) -> pd.DataFrame:
        chunk_col = self._resolve_col(df, "chunk_id")
        raw_col = "raw_text" if "raw_text" in df.columns else None
        out = df.copy()
        if chunk_col:
            ip = out[chunk_col].astype(str)
            mask_self_trans = self._starts_with(ip, f"T_{self_lang}_")
            mask_other_stray = self._starts_with(ip, f"{other_lang}_")
            out = out[~(mask_self_trans | mask_other_stray)].copy()
        if raw_col:
            out = out[~out[raw_col].str.contains("isbn", case=False, na=False)]
        return out.reset_index(drop=True)

    def _stable_order(self, df: pd.DataFrame) -> pd.DataFrame:
        chunk_col = self._resolve_col(df, "chunk_id")
        if not chunk_col:
            return df
        ip = df[chunk_col].astype(str)
        is_translated = ip.str.startswith("T_")
        return pd.concat([df[is_translated], df[~is_translated]], ignore_index=True)

    # ---------- normalization ----------

    def _normalize(self, df: pd.DataFrame, force_lang: Optional[str] = None) -> pd.DataFrame:
        """
        Rename/mint the standard columns. Preserve extra metadata.
        """
        df = df.copy()
        c_chunk = self._resolve_col(df, "chunk_id")
        c_doc = self._resolve_col(df, "doc_id")
        c_text = self._resolve_col(df, "text")
        c_full = self._resolve_col(df, "full_doc")
        c_lang = self._resolve_col(df, "lang")
        c_title = self._resolve_col(df, "title")
        c_url = self._resolve_col(df, "url")
        c_equiv = self._resolve_col(df, "equivalence")

        # Ensure existence
        for need in [c_chunk, c_text]:
            if not need:
                raise ValueError("Input dataframe is missing required text/chunk_id columns."
                                 " Provide a schema={'chunk_id':..., 'text':...} mapping.")

        # Build normalized frame
        out = pd.DataFrame({
            "chunk_id": df[c_chunk].astype(str),
            "text": df[c_text],
        })

        # doc_id
        if c_doc and c_doc in df.columns:
            out["doc_id"] = df[c_doc].astype(str)
        else:
            out["doc_id"] = df.apply(
                lambda r: self._derive_doc_id(r, c_doc, c_chunk), axis=1)

        # full_doc (optional)
        out["full_doc"] = df[c_full] if c_full in df.columns else None

        # lang
        lang_val = force_lang or (
            df[c_lang].iloc[0] if c_lang in df.columns and df[c_lang].notna().any() else "XX")
        out["lang"] = self._upper_lang(lang_val)

        # extras
        if c_title in df.columns:
            out["title"] = df[c_title]
        if c_url in df.columns:
            out["url"] = df[c_url]
        if c_equiv in df.columns:
            out["equivalence"] = df[c_equiv]
            
        # Keep any extra columns from input
        known = set(out.columns)
        extras = [c for c in df.columns if c not in known]
        for c in extras:
            out[c] = df[c]

        return out


    def _spacy_model_for(self, lang_upper: str) -> str:
        if not self.spacy_models or lang_upper not in self.spacy_models:
            raise ValueError(f"No spaCy model configured for '{lang_upper}'. "
                             f"Provide spacy_models like {{'EN':'en_core_web_sm'}}.")
        return self.spacy_models[lang_upper]

    def _preprocess_df(self, df: pd.DataFrame, lang_upper: str, tag: str) -> pd.DataFrame:
        """
        Run NLPipe on a TEMP parquet that exposes exactly the columns it expects:
        id_preproc (← df['chunk_id']), raw_text (← df['text']), lang
        Then merge the returned 'lemmas' back onto the *normalized* df by chunk_id.
        """
        # 0) sanity: normalized columns must exist
        required = {"chunk_id", "text", "lang"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"_preprocess_df expects normalized df with columns {required}. "
                f"Missing: {missing}. Did _normalize() set 'chunk_id' from your schema?"
            )

        tmp_dir = (self.storing_path / "_tmp_preproc")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 1) minimal parquet for NLPipe
        work = pd.DataFrame({
            "id_preproc": df["chunk_id"].astype(str),
            "text":  df["text"],
            "lang":  df["lang"].astype(str),
        })
        tmp_parq = tmp_dir / f"{tag}_{lang_upper}.parquet"
        work.to_parquet(tmp_parq, compression="gzip")

        # 2) run NLPipe (logs to console)
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
            subprocess.run(cmd, check=True)  # no capture_output -> live console logs
            print(f"✓ Preprocessed (lang={lang_upper})")
        else:
            print("Preprocessing skipped (not configured).")
            return df

        # 3) read NLPipe output and merge lemmas back by id_preproc ↔ chunk_id
        proc = pd.read_parquet(tmp_parq)
        if "id_preproc" not in proc.columns or "lemmas" not in proc.columns:
            raise RuntimeError(f"NLPipe output missing id_preproc/lemmas; got: {list(proc.columns)}")

        merged = df.merge(
            proc[["id_preproc", "lemmas"]],
            left_on="chunk_id",
            right_on="id_preproc",
            how="left",
            validate="one_to_one",
        )
        
        to_drop = [c for c in merged.columns if c.startswith("id_preproc")]
        merged = merged.drop(columns=to_drop)

        return merged


    @staticmethod
    def _pair_key_from_chunk_id(chunk_id: str, row_lang: str) -> Tuple[str, str]:
        """
        Returns a stable key representing the *original source* chunk, regardless
        of whether the row is original or translated.

        Examples:
          EN_12_3     + row_lang=EN  -> ("EN", "12_3")
          T_EN_12_3   + row_lang=DE  -> ("EN", "12_3")
          DE_77_0     + row_lang=DE  -> ("DE", "77_0")
          T_DE_77_0   + row_lang=EN  -> ("DE", "77_0")
        """
        s = str(chunk_id)
        m = re.match(r"^T_([A-Za-z]{2})_(.+)$", s)
        if m:
            return (m.group(1).upper(), m.group(2))
        m2 = re.match(r"^([A-Za-z]{2})_(.+)$", s)
        if m2:
            return (m2.group(1).upper(), m2.group(2))
        # fallback: treat whole id as the "rest"
        return (row_lang.upper(), s)

    # ---------- main API ----------

    def format_dataframes(self) -> None:
        self.read_dataframes()

        # Infer language tags
        anchor_lang = self._infer_lang_code(self.anchor_df)
        target_lang = self._infer_lang_code(self.target_df)

        # Clean & order
        anc = self._decontaminate(self.anchor_df, anchor_lang, target_lang)
        tgt = self._decontaminate(self.target_df, target_lang, anchor_lang)
        anc = self._stable_order(anc)
        tgt = self._stable_order(tgt)

        # Normalize columns to a common schema
        anc_norm = self._normalize(anc, force_lang=anchor_lang)
        tgt_norm = self._normalize(tgt, force_lang=target_lang)

        # Save per-language temporaries, run NLPipe once per language to fill 'lemmas'
        tmp_dir = (self.storing_path / "_tmp_preproc")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        anc_parq = tmp_dir / f"anchor_{anchor_lang}.parquet"
        tgt_parq = tmp_dir / f"target_{target_lang}.parquet"

        anc_norm.to_parquet(anc_parq, compression="gzip")
        tgt_norm.to_parquet(tgt_parq, compression="gzip")

        # Run preprocessing per language
        anc_proc = self._preprocess_df(anc_norm, anchor_lang, tag="anchor")
        tgt_proc = self._preprocess_df(tgt_norm, target_lang, tag="target")

        # Build pairing keys representing the original source chunk
        def add_pair_key(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            keys: List[Tuple[str, str]] = [
                self._pair_key_from_chunk_id(cid, L)
                for cid, L in zip(df["chunk_id"].astype(str), df["lang"].astype(str))
            ]
            df["pair_src_lang"] = [k[0] for k in keys]
            df["pair_rest"] = [k[1] for k in keys]
            df["pair_key"] = df["pair_src_lang"] + ":" + df["pair_rest"]
            return df

        anc_proc = add_pair_key(anc_proc)
        tgt_proc = add_pair_key(tgt_proc)

        # For EN rows, lemmas_tr should come from DE's translations of EN originals: rows in target with id 'T_EN_...'
        # But thanks to pair_key, we can simply map by pair_key.
        # Build maps from each side's *translation rows* to their lemmas:
        def is_translation_of(lang_src: str, df: pd.DataFrame) -> pd.Series:
            # rows whose chunk_id starts with f"T_{lang_src}_"
            return df["chunk_id"].astype(str).str.startswith(f"T_{lang_src}_")

        anc_trans_map = anc_proc.loc[is_translation_of(target_lang, anc_proc), [
            "pair_key", "lemmas"]].dropna()
        tgt_trans_map = tgt_proc.loc[is_translation_of(anchor_lang, tgt_proc), [
            "pair_key", "lemmas"]].dropna()

        # translations of TARGET in ANCHOR
        map_from_anc = dict(
            zip(anc_trans_map["pair_key"], anc_trans_map["lemmas"]))
        # translations of ANCHOR in TARGET
        map_from_tgt = dict(
            zip(tgt_trans_map["pair_key"], tgt_trans_map["lemmas"]))

        # Fill lemmas_tr:
        # - For anchor originals (pair_key = ANCHOR:<rest>), find lemmas in target translations (map_from_tgt)
        # - For target originals, find lemmas in anchor translations (map_from_anc)
        anc_proc["lemmas_tr"] = anc_proc["pair_key"].map(map_from_tgt)
        tgt_proc["lemmas_tr"] = tgt_proc["pair_key"].map(map_from_anc)
        
        # keep only originals (no T_* rows) for the final output
        is_original = lambda df: ~df["chunk_id"].astype(str).str.startswith("T_")
        anc_orig = anc_proc[is_original(anc_proc)].copy()
        tgt_orig = tgt_proc[is_original(tgt_proc)].copy()
        
        # drop pairing helper columns
        cols_to_drop = ["pair_src_lang", "pair_rest", "pair_key"]
        for c in cols_to_drop:
            if c in anc_orig.columns: anc_orig.drop(columns=c, inplace=True)
            if c in tgt_orig.columns: tgt_orig.drop(columns=c, inplace=True)


        #n = min(len(anc_orig), len(tgt_orig))
        #anc_orig = anc_orig.iloc[:n].reset_index(drop=True)
        #tgt_orig = tgt_orig.iloc[:n].reset_index(drop=True)

        # final stack
        self.final_df = pd.concat([anc_orig, tgt_orig], ignore_index=True)
        
        # drop all rows where lemmas is None
        df = df[~df.lemmas.isnull()]
        # replace None in lemmas_tr with empty string
        df["lemmas_tr"] = df["lemmas_tr"].fillna("")

        # Save unified parquet
        out_path = self.storing_path / "polylingual_df.parquet"
        self.final_df.to_parquet(out_path, compression="gzip")
        print(f"Saved: {out_path.as_posix()}")

    def save_to_parquet(self) -> None:
        if self.final_df is None or self.final_df.empty:
            raise RuntimeError(
                "final_df is empty. Run format_dataframes() first.")
        out_path = self.storing_path / "polylingual_df.parquet"
        self.final_df.to_parquet(out_path, compression="gzip")
        print(f"Saved: {out_path.as_posix()}")
