import os
import pandas as pd
from flask import jsonify


def validate_and_get_dataset(email: str, dataset: str, output: str, phase: str):
    """
    Verifica la existencia del dataset y que no exista ya el output.
    
    Parámetros:
        email (str): correo del usuario
        dataset (str): nombre del dataset
        output (str): nombre del output especificado por el usuario
        phase (str): nombre de la fase, ej. "1_Segmenter" o "2_Translator"

    Devuelve:
        tuple(dataset_path, output_dir) si todo es válido,
        o una respuesta Flask (jsonify, status_code) si hay error.
    """

    try:
        # 1️⃣ Buscar el dataset en el parquet principal
        parquet_path = "/data/datasets_stage_preprocess.parquet"
        if not os.path.exists(parquet_path):
            return jsonify({
                "status": "error",
                "message": f"Parquet file not found at {parquet_path}"
            }), 404

        df = pd.read_parquet(parquet_path)

        row = df[
            (df["Usermail"] == email) &
            (df["Dataset"] == dataset) &
            (df["Stage"] == 1)
        ]

        if row.empty:
            return jsonify({
                "status": "error",
                "message": f"No dataset found for user '{email}', dataset '{dataset}', stage 1."
            }), 404

        dataset_path = row.iloc[0]["Path"]

        # 2️⃣ Comprobar que no exista el output
        output_dir = f"/data/{email}/Stage_1/{dataset}/{phase}/{output}/dataset"
        if os.path.exists(output_dir):
            return jsonify({
                "status": "error",
                "message": f"Output already exists at {output_dir}. Please choose another output name."
            }), 400

        return dataset_path, output_dir

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Validation error: {e}"
        }), 500
    