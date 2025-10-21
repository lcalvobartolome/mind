import os
import pandas as pd

from utils import validate_and_get_dataset
from flask import Blueprint, request, jsonify
from mind.corpus_building.segmenter import Segmenter
from mind.corpus_building.translator import Translator
from mind.corpus_building.data_preparer import DataPreparer

preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')


@preprocessing_bp.route('/segmenter', methods=['POST'])
def segmenter():
    try:
        data = request.get_json()
        email = data.get("email")
        dataset = data.get("dataset")
        segmenter_data = data.get("segmenter_data")

        # Validate input & output datasets
        validation = validate_and_get_dataset(
            email=email,
            dataset=dataset,
            output=segmenter_data['output'],
            phase="1_Segmenter"
        )

        if isinstance(validation, tuple): dataset_path, output_dir = validation
        else: return validation

        # Segmenter
        print(f"Segmenting dataset {dataset}...")

        seg = Segmenter()
        seg.segment(
            path_df=dataset_path,
            path_save=output_dir,
            text_col=segmenter_data['text_col'],
            min_length=segmenter_data['min_length'],
            sep=segmenter_data['sep']
        )

        result = f"Dataset {dataset} segmented successfully."

        return jsonify({"status": "success", "message": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@preprocessing_bp.route('/translator', methods=['POST'])
def translator():
    try:
        data = request.get_json()
        email = data.get("email")
        dataset = data.get("dataset")
        translator_data = data.get("translator_data")

        # Validate input & output datasets
        validation = validate_and_get_dataset(
            email=email,
            dataset=dataset,
            output=translator_data['output'],
            phase="2_Translator"
        )

        if isinstance(validation, tuple): dataset_path, output_dir = validation
        else: return validation

        # Translator
        print(f"Translating dataset {dataset}...")

        trans = Translator()
        trans.translate(
            path_df=dataset_path,
            save_path=output_dir,
            src_lang=translator_data['src_lang'],
            tgt_lang=translator_data['tgt_lang'],
            text_col=translator_data['text_col'],
            lang_col=translator_data['lang_col'],
        )

        print(f"Translating dataset: {dataset}")

        result = f"Dataset {dataset} translated successfully."

        return jsonify({"status": "success", "message": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@preprocessing_bp.route('/preparer', methods=['POST'])
def preparer():
    try:
        data = request.get_json()
        email = data.get("email")
        dataset = data.get("dataset")
        preparer_data = data.get("preparer_data")

        # Validate input & output datasets
        validation = validate_and_get_dataset(
            email=email,
            dataset=dataset,
            output=preparer_data['output'],
            phase="3_Translator"
        )

        if isinstance(validation, tuple): dataset_path, output_dir = validation
        else: return validation

        # Data Preparer
        # print(f"Preparing dataset: {dataset}")

        # prep = DataPreparer(
        #     preproc_script="/backend/NLPipe/src/nlpipe/cli.py",
        #     config_path="config.json",
        #     stw_path="/backend/NLPipe/src/nlpipe/stw_lists",
        #     spacy_models={
        #         "en": "en_core_web_sm",
        #         "de": "de_core_news_sm",
        #         "es": "es_core_news_sm"},
        #     schema={
        #         "chunk_id": "id_preproc",
        #         "doc_id": "id",
        #         "text": "text",
        #         "full_doc": "summary",
        #         "lang": "lang",
        #         "title": "title",
        #         "url": "url",
        #         "equivalence": "equivalence",
        #     },
        # )

        # prep.format_dataframes(
        #     anchor_path=str(output_dir / f"{args.base_name}_trans_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip"),
        #     comparison_path=str(output_dir / f"{args.base_name}_trans_s{args.target_lang}_t{args.seed_lang}_n{args.ndocs}.parquet.gzip"),
        #     path_save=str(output_dir / f"{args.base_name}_polylingual_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip")
        # )

        # Result
        result = f"Dataset {dataset} prepared successfully."

        return jsonify({"status": "success", "message": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
