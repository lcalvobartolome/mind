import os
import re
import sys
import glob
import json
import string
import requests
import warnings
import threading
import pandas as pd

from multiprocessing import Event, Process
from mind.cli import comma_separated_ints
from flask import Blueprint, jsonify, request
from utils import get_TM_detection, obtain_langs_TM


detection_bp = Blueprint("detection", __name__)
MIND_FRONTEND_URL = os.getenv('MIND_FRONTEND_URL', 'http://frontend:5000')

active_processes = {}
MAX_USERS = 2
lock = threading.Lock()

class StreamForwarder:
    """
    Stream log to HTML terminal
    """
    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __init__(self, endpoint_url=None, log_file="/data/pipeline.log"):
        self.endpoint_url = endpoint_url
        self.log_file = log_file
        warnings.simplefilter('default')

        with open(self.log_file, "w") as f:
            f.write("")

    def clean_line(self, line: str) -> str:
        line = self.ANSI_ESCAPE.sub('', line)
        printable = set(string.printable)
        return ''.join([c for c in line if c in printable]).strip()

    def write(self, message: str):
        if not message.strip():
            return

        clean_msg = self.clean_line(message)

        if clean_msg:
            if "Warning" not in clean_msg and self.endpoint_url:
                try:
                    requests.post(self.endpoint_url, json={"log": clean_msg}, timeout=0.5)
                except Exception:
                    pass

            self._save_local(clean_msg)

        sys.__stdout__.write(message + "\n")
        sys.__stdout__.flush()

    def _save_local(self, message: str):
        try:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        except Exception:
            pass

    def flush(self):
        pass

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


@detection_bp.route('detection/topickeys', methods=['GET'])
def getTopicKeys():
    try:
        data = request.get_json()
        email = data.get("email")
        dataset_name = data.get("dataset_name")
        topic_model = data.get("topic_model")

        if not email or not dataset_name or not topic_model:
            return jsonify({"error": "Missing one of the mandatory arguments"}), 400
        
        path = f'/data/{email}/3_TopicModel/{topic_model}'

        # Read parquet to tm model
        if not os.path.exists(path):
            return jsonify({"error": f'Not existing Topic Model "{topic_model}"'}), 500
        
        topic_keys = {
            "TM_name": topic_model,
            "lang": []
        }

        # Mallet files
        mallet_topic_keys = f'{path}/mallet_output/keys_*.txt'
        for file in glob.glob(mallet_topic_keys):
            if not os.path.isfile(file):
                continue

            lang = file.split('keys_')[-1].replace('.txt', '')
            topic_keys["lang"].append(lang)
            
            # "k" : {"name": title, "EN": [words], "ES: [words]"}
            with open(file, 'r') as f:
                i = 1
                keys = {}
                for topic in f:
                    # TODO future work call LLM for a title in topics
                    keys[i] = topic.replace('\n', '').split(' ')
                    i += 1
                topic_keys[lang] = keys
        
        return jsonify(topic_keys), 200
    
    except Exception as e:
        print(str(e))
        return jsonify({"error": f"ERROR: {str(e)}"}), 500

def run_pipeline_process(cfg, run_kwargs, log_file):
    from mind.pipeline.pipeline import MIND
    try:
        with StreamForwarder(f'{MIND_FRONTEND_URL}/log_detection', log_file):
            mind = MIND(**cfg)
            print("MIND class created. Running pipeline...", file=sys.__stdout__)
            mind.run_pipeline(**run_kwargs)
    except Exception as e:
        print(f"[PIPELINE ERROR] {e}", file=sys.__stdout__)

@detection_bp.route('/detection/analyse_contradiction', methods=['POST'])
def analyse_contradiction():
    try:
        data = request.get_json()
        print(data)
        email = data.get("email")
        TM = data.get("TM")
        topics = data.get("topics")
        sample_size = data.get("sample_size")

        # First check if was analyse before
        path_results = f'/data/{email}/4_Detection/{TM}_contradiction/{topics}/mind_results.parquet'
        if os.path.exists(path_results):
            print('Results were done before.')
            return jsonify({"message": f"Pipeline done correctly"}), 200
        
        print('Analysing...')
        paths = get_TM_detection(email, TM)

        if isinstance(paths, tuple):
            pathTM, pathCorpus = paths[0], paths[1]
        else:
            raise Exception("Path TM failed")
        
        lang = obtain_langs_TM(pathTM)
        
        # =========================
        # =      CONFIG PART      =
        # =========================

        # source_corpus = {
        #     "corpus_path": pathCorpus,
        #     "thetas_path": f'{pathTM}/mallet_output/thetas_{lang[0]}.npz',
        #     "id_col": 'doc_id',
        #     "passage_col": 'text',
        #     "full_doc_col": 'full_doc',
        #     "language_filter": lang[0],
        #     "filter_ids": None,
        #     "load_thetas": True # Check
        # }

        source_corpus = {
            "corpus_path": pathCorpus,
            "thetas_path": f'{pathTM}/mallet_output/thetas_{lang[0]}.npz',
            "id_col": 'doc_id',
            "passage_col": 'lemmas',
            "full_doc_col": 'raw_text',
            "language_filter": lang[0],
            "filter_ids": None, # 
            "load_thetas": True,
        }

        # target_corpus = {
        #     "corpus_path": pathCorpus,
        #     "thetas_path": f'{pathTM}/mallet_output/thetas_{lang[1]}.npz',
        #     "id_col": 'doc_id',
        #     "passage_col": 'text',
        #     "full_doc_col": 'full_doc',
        #     "language_filter": lang[1],
        #     "filter_ids": None,
        #     "load_thetas": True # Check
        # }

        target_corpus = {
            "corpus_path": pathCorpus,
            "thetas_path": f'{pathTM}/mallet_output/thetas_{lang[1]}.npz',
            "id_col": 'doc_id',
            "passage_col": 'lemmas',
            "full_doc_col": 'raw_text',
            "language_filter": lang[1],
            "filter_ids": None,
            "load_thetas": True,
            'index_path': f'/data/{email}/3_TopicModel/{TM}/'
        }

        cfg = {
            "llm_model": "llama3:8b",
            "llm_server": "http://kumo02.tsc.uc3m.es:11434",
            "source_corpus": source_corpus,
            "target_corpus": target_corpus,
            # "dry_run": False,
            # "do_check_entailement": True,
            "config_path": '/src/config/config.yaml'
        }

        run_kwargs = {
            "topics": comma_separated_ints(topics),
            "sample_size": sample_size,
            "path_save": path_results,
            "previous_check": None
        }

        log_file = f'/data/{email}/pipeline-mind.log'
        global lock, active_processes
        with lock:
            if email not in active_processes and len(active_processes) >= MAX_USERS:
                return jsonify({"error": "Máximo de usuarios activos alcanzado"}), 429

            # Cancel on the other session
            if email in active_processes:
                prev_proc = active_processes[email]["process"]
                if prev_proc.is_alive():
                    print(f"Cancelling previous pipeline for {email}", file=sys.__stdout__)
                    prev_proc.terminate()
                    prev_proc.join()

            cancel_event = Event()
            p = Process(target=run_pipeline_process, args=(cfg, run_kwargs, log_file))
            p.start()

            active_processes[email] = {"process": p, "cancel_event": cancel_event}

        active_processes[email]["process"].join()

        with lock:
            del active_processes[email]

        return jsonify({"message": "Pipeline completed"}), 200
    
    except Exception as e:
        print(e)
        return jsonify({"error": f"ERROR: {str(e)}"}), 500
    
@detection_bp.route('/detection/result_mind', methods=['GET'])
def get_results_mind():
    try:
        data = request.get_json()
        print(data)
        email = data.get("email")
        TM = data.get("TM")
        topics = data.get("topics")

        df = pd.read_parquet(f'/data/{email}/4_Detection/{TM}_contradiction/{topics}/mind_results.parquet', engine='pyarrow')
        result_mind = df.to_dict(orient='records')
        result_columns = df.columns.tolist()

        columns_json = json.dumps([{"name": col} for col in df.columns])
        non_orderable_indices = json.dumps([i for i, col in enumerate(df.columns) if col in ['label', 'final_label']])

        return jsonify({"message": f"Results from MIND obtained correctly",
                        "result_mind": result_mind,
                        "result_columns": result_columns,
                        "columns_json": columns_json,
                        "non_orderable_indices": non_orderable_indices}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": f"ERROR: {str(e)}"}), 500
    
@detection_bp.route('/detection/update_results', methods=['POST'])
def update_result_mind():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        
        TM = request.form.get("TM")
        topics = request.form.get("topics")
        email = request.form.get("email")
        if not TM or not topics or not email:
            return jsonify({"error": "Missing parameters"}), 400

        df = pd.read_excel(file, engine='openpyxl')
        keys = []
        for key in df.keys():
            values = key.replace('\n', '').split(' ')
            if 'label' in values:
                keys.append('label')
            elif 'final_label' in values:
                keys.append('final_label')
            else:
                keys.append(values[0])

        df.columns = keys
        df.to_parquet(f'/data/{email}/4_Detection/{TM}_contradiction/{topics}/mind_results.parquet', engine='pyarrow')

        return jsonify({"message": f"Results from MIND saved correctly"}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": f"ERROR: {str(e)}"}), 500
